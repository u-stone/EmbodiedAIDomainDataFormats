"""Low-level LeRobot v3.0 format inspector and read-only reader (PyArrow + PyAV, zero lerobot black-box).

Systems-level anatomy of LeRobot v3.0 (HuggingFace / OpenPI / SmolVLA / GR00T):
- Decoupled columnar + video storage:
    meta/info.json                                  -> global schema, fps, shard path templates
    meta/stats.json                                 -> feature normalization stats (min/max/mean/std/q01/q99)
    meta/tasks.parquet                              -> task_index -> natural language instruction
    meta/episodes/chunk-XXX/file-XXX.parquet        -> episode table with GLOBAL row bounds
                                                       (dataset_from_index, dataset_to_index) & video time slots
    data/chunk-XXX/file-XXX.parquet                 -> shared tabular shards (multiple episodes per file!)
    videos/<cam>/chunk-XXX/file-XXX.mp4             -> shared MP4 shards (multiple episodes concatenated!)
- Key systems pitfalls exposed directly in this reader:
  1. Global -> Local row index translation: `dataset_from_index` is GLOBAL across all shards;
     when indexing a specific `data/chunk-c/file-f.parquet`, you MUST subtract that shard's
     starting global row (`local_row = global_row - shard_global_start`).
  2. Video GOP & Seek cost: random frame access requires `av.open -> container.seek(pts) -> decode
     forward from keyframe`. That's why LeRobot enforces tiny GOP (`g=2`, `bf=0`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

from embodied_formats.common import (
    BaseFormatReader,
    EpisodeMeta,
    FieldSchema,
    InspectReport,
    StepSample,
    WindowSample,
    dir_disk_usage,
    format_bytes,
)


class LeRobotV3FormatReader(BaseFormatReader):
    """Direct PyArrow + PyAV inspector and reader for LeRobot v3.0 datasets."""

    format_name = "lerobot_v3"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        info_path = self.path / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Not a LeRobot dataset (missing {info_path})")

        self._info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
        stats_path = self.path / "meta" / "stats.json"
        self._stats: dict[str, Any] = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

        self._fps = float(self._info.get("fps", 10.0))
        self._video_keys = [
            k for k, v in self._info.get("features", {}).items() if v.get("dtype") == "video"
        ]

        # Load tasks lookup if present
        self._tasks: dict[int, str] = {}
        tasks_pq = self.path / "meta" / "tasks.parquet"
        if tasks_pq.exists():
            t_tbl = pq.read_table(tasks_pq)
            cols = t_tbl.column_names
            if "task_index" in cols:
                idxs = t_tbl["task_index"].to_pylist()
                # task text may be in 'task' column or pandas index column
                text_col = "task" if "task" in cols else next((c for c in cols if c != "task_index"), None)
                if text_col is not None:
                    for tid, txt in zip(idxs, t_tbl[text_col].to_pylist()):
                        self._tasks[int(tid)] = str(txt)

        # Load meta/episodes/*.parquet
        ep_files = sorted((self.path / "meta" / "episodes").rglob("*.parquet"))
        if not ep_files:
            raise FileNotFoundError(f"Missing meta/episodes/*.parquet under {self.path}")
        self._ep_rows: list[dict[str, Any]] = []
        for ef in ep_files:
            tbl = pq.read_table(ef)
            self._ep_rows.extend(tbl.to_pylist())
        self._ep_rows.sort(key=lambda r: int(r["episode_index"]))

        # Compute each data shard's starting GLOBAL row index so we can translate
        # global row -> local row within that Parquet file!
        self._shard_start_row: dict[tuple[int, int], int] = {}
        for row in self._ep_rows:
            key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
            g_from = int(row["dataset_from_index"])
            if key not in self._shard_start_row or g_from < self._shard_start_row[key]:
                self._shard_start_row[key] = g_from

        # Caches for open Parquet tables and PyAV containers
        self._parquet_cache: dict[Path, Any] = {}
        self._av_cache: dict[Path, av.container.InputContainer] = {}
        self._episodes: list[EpisodeMeta] | None = None

    def _data_shard_path(self, chunk_idx: int, file_idx: int) -> Path:
        tpl = self._info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        return self.path / tpl.format(chunk_index=chunk_idx, file_index=file_idx)

    def _video_shard_path(self, video_key: str, chunk_idx: int, file_idx: int) -> Path:
        tpl = self._info.get(
            "video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        )
        return self.path / tpl.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)

    def _get_parquet_table(self, pq_path: Path) -> Any:
        if pq_path not in self._parquet_cache:
            self._parquet_cache[pq_path] = pq.read_table(pq_path)
        return self._parquet_cache[pq_path]

    def _decode_video_frames(self, mp4_path: Path, target_timestamps_s: list[float]) -> list[np.ndarray]:
        """Seek and decode exact frame(s) by timestamp using PyAV (exposes GOP seek behavior)."""
        if mp4_path not in self._av_cache:
            self._av_cache[mp4_path] = av.open(str(mp4_path))
        container = self._av_cache[mp4_path]
        stream = container.streams.video[0]
        tb = float(stream.time_base)

        results: list[np.ndarray] = []
        # Sort timestamps to minimize redundant backward seeks when reading a window
        indexed_ts = sorted(enumerate(target_timestamps_s), key=lambda x: x[1])
        decoded_map: dict[int, np.ndarray] = {}

        # Seek once before the earliest requested timestamp
        earliest_ts = max(0.0, indexed_ts[0][1])
        seek_pts = int(earliest_ts / tb)
        container.seek(seek_pts, backward=True, stream=stream)

        frame_iter = container.decode(stream)
        last_rgb: np.ndarray | None = None
        last_ts = -1.0
        half_frame = 0.5 / max(self._fps, 1.0)

        for orig_i, req_ts in indexed_ts:
            if last_rgb is not None and abs(last_ts - req_ts) <= half_frame:
                decoded_map[orig_i] = last_rgb
                continue
            for frame in frame_iter:
                f_ts = float(frame.pts * tb) if frame.pts is not None else 0.0
                if f_ts + half_frame >= req_ts:
                    last_rgb = frame.to_ndarray(format="rgb24")
                    last_ts = f_ts
                    decoded_map[orig_i] = last_rgb
                    break
            if orig_i not in decoded_map and last_rgb is not None:
                decoded_map[orig_i] = last_rgb

        for i in range(len(target_timestamps_s)):
            if i not in decoded_map:
                raise RuntimeError(f"Failed to decode timestamp {target_timestamps_s[i]:.4f}s from {mp4_path}")
            results.append(decoded_map[i])
        return results

    def _inspect_video_gop(self, mp4_path: Path, max_packets: int = 2000) -> dict[str, Any]:
        """Demux MP4 packets via PyAV to measure true keyframe intervals (GOP size) and pix_fmt."""
        with av.open(str(mp4_path)) as container:
            stream = container.streams.video[0]
            keyframe_indices: list[int] = []
            pkt_count = 0
            for pkt in container.demux(stream):
                if pkt.dts is None:
                    continue
                if pkt.is_keyframe:
                    keyframe_indices.append(pkt_count)
                pkt_count += 1
                if pkt_count >= max_packets:
                    break
            gops = np.diff(keyframe_indices) if len(keyframe_indices) >= 2 else np.array([pkt_count])
            return {
                "file": mp4_path.name,
                "codec": stream.codec_context.name,
                "pix_fmt": stream.codec_context.pix_fmt,
                "resolution": f"{stream.width}x{stream.height}",
                "avg_rate": str(stream.average_rate),
                "packets_inspected": pkt_count,
                "keyframes": len(keyframe_indices),
                "min_gop": int(np.min(gops)) if len(gops) else 0,
                "mean_gop": round(float(np.mean(gops)), 2) if len(gops) else 0.0,
                "max_gop": int(np.max(gops)) if len(gops) else 0,
            }

    def list_episodes(self) -> list[EpisodeMeta]:
        if self._episodes is not None:
            return self._episodes
        eps: list[EpisodeMeta] = []
        for r in self._ep_rows:
            g_from = int(r["dataset_from_index"])
            g_to = int(r["dataset_to_index"])
            tasks_field = r.get("tasks", [])
            if isinstance(tasks_field, list) and tasks_field:
                lang = str(tasks_field[0])
            else:
                lang = ""
            eps.append(
                EpisodeMeta(
                    episode_id=int(r["episode_index"]),
                    num_steps=g_to - g_from,
                    fps=self._fps,
                    language_instruction=lang,
                    extra={
                        "dataset_from_index": g_from,
                        "dataset_to_index": g_to,
                        "data_shard": (int(r["data/chunk_index"]), int(r["data/file_index"])),
                    },
                )
            )
        self._episodes = eps
        return eps

    def inspect(self) -> InspectReport:
        disk_bytes, file_count = dir_disk_usage(self.path)
        episodes = self.list_episodes()

        schema_fields: list[FieldSchema] = []
        for fname, fmeta in self._info.get("features", {}).items():
            dtype = str(fmeta.get("dtype", ""))
            shape = str(tuple(fmeta.get("shape", [])))
            extra_info = fmeta.get("info", {}) or {}
            enc = str(extra_info.get("video.codec", "parquet")) if dtype == "video" else "parquet"
            details = f"names={fmeta.get('names')}" if fmeta.get("names") else ""
            if dtype == "video" and extra_info:
                details = f"pix_fmt={extra_info.get('video.pix_fmt')}, fps={extra_info.get('video.fps')}"
            schema_fields.append(FieldSchema(name=fname, dtype=dtype, shape=shape, encoding=enc, details=details))

        # Inspect first data Parquet file metadata
        data_files = sorted((self.path / "data").rglob("*.parquet"))
        pq_details: dict[str, Any] = {}
        if data_files:
            pf = pq.ParquetFile(data_files[0])
            rg0 = pf.metadata.row_group(0)
            col0_comp = rg0.column(0).compression if rg0.num_columns > 0 else "unknown"
            pq_details = {
                "data_parquet_shards": len(data_files),
                "shard0_rows": pf.metadata.num_rows,
                "shard0_row_groups": pf.metadata.num_row_groups,
                "shard0_compression": col0_comp,
            }

        # Inspect first MP4 per video camera
        video_gop_reports: dict[str, Any] = {}
        for vk in self._video_keys:
            mp4s = sorted((self.path / "videos" / vk).rglob("*.mp4"))
            if mp4s:
                video_gop_reports[vk] = self._inspect_video_gop(mp4s[0])

        # Audit stats.json briefly
        stats_audit: dict[str, Any] = {}
        for k, st in self._stats.items():
            if not isinstance(st, dict):
                continue
            has_q = "q01" in st and "q99" in st
            stats_audit[k] = {
                "keys": sorted(st.keys()),
                "has_quantiles_q01_q99": has_q,
            }

        total_steps = sum(ep.num_steps for ep in episodes)
        return InspectReport(
            format_name=self.format_name,
            path=str(self.path),
            disk_bytes=disk_bytes,
            file_count=file_count,
            num_episodes=len(episodes),
            total_steps=total_steps,
            avg_fps=self._fps,
            schema=schema_fields,
            storage_details={
                "codebase_version": self._info.get("codebase_version"),
                "robot_type": self._info.get("robot_type"),
                "parquet": pq_details,
                "video_gop_anatomy": video_gop_reports,
                "stats_json_audit": stats_audit,
            },
            notes=[
                "LeRobot v3.0 将多条轨迹合并到共享 Parquet 与 MP4 分片，彻底消除 v2.1 海量轨迹下的小文件爆炸问题。",
                "核心读取坑点：meta/episodes 中的 dataset_from_index 是跨分片全局行号，直接切片单个 file-XXX.parquet 时必须减去该分片起始行号转为局部索引。",
                "视频采用极小 GOP（通常 g=2）换取随机 Seek 性能；帧间压缩比 HDF5/Zarr 节省 80%–95% 空间，但随机抽帧仍存在 CPU 视频解码开销。",
            ],
        )

    def read_step(self, episode_id: int, step_index: int) -> StepSample:
        r = self._ep_rows[episode_id]
        c_idx, f_idx = int(r["data/chunk_index"]), int(r["data/file_index"])
        global_row = int(r["dataset_from_index"]) + step_index
        local_row = global_row - self._shard_start_row[(c_idx, f_idx)]

        tbl = self._get_parquet_table(self._data_shard_path(c_idx, f_idx))
        row_dict = {col: tbl[col][local_row].as_py() for col in tbl.column_names}

        state = (
            np.asarray(row_dict["observation.state"], dtype=np.float32)
            if "observation.state" in row_dict and row_dict["observation.state"] is not None
            else None
        )
        action = (
            np.asarray(row_dict["action"], dtype=np.float32)
            if "action" in row_dict and row_dict["action"] is not None
            else None
        )
        ts = float(row_dict.get("timestamp", step_index / max(self._fps, 1e-6)))

        images: dict[str, np.ndarray] = {}
        for vk in self._video_keys:
            vc = int(r[f"videos/{vk}/chunk_index"])
            vf = int(r[f"videos/{vk}/file_index"])
            v_from = float(r[f"videos/{vk}/from_timestamp"])
            mp4_path = self._video_shard_path(vk, vc, vf)
            target_ts = v_from + step_index / self._fps
            images[vk] = self._decode_video_frames(mp4_path, [target_ts])[0]

        ep = self.list_episodes()[episode_id]
        return StepSample(
            episode_id=episode_id,
            step_index=step_index,
            timestamp_s=ts,
            images=images,
            state=state,
            action=action,
            language_instruction=ep.language_instruction,
            extra={"global_row": global_row, "local_row": local_row, "data_shard": (c_idx, f_idx)},
        )

    def read_window(
        self,
        episode_id: int,
        step_index: int,
        obs_horizon: int = 2,
        action_horizon: int = 16,
    ) -> WindowSample:
        """Native LeRobot v3.0 window read: 1 Parquet slice + 1 video seek per camera."""
        r = self._ep_rows[episode_id]
        ep = self.list_episodes()[episode_id]
        n = ep.num_steps
        c_idx, f_idx = int(r["data/chunk_index"]), int(r["data/file_index"])
        ep_local_start = int(r["dataset_from_index"]) - self._shard_start_row[(c_idx, f_idx)]

        obs_indices = [max(0, min(n - 1, step_index - (obs_horizon - 1 - i))) for i in range(obs_horizon)]
        act_indices = [max(0, min(n - 1, step_index + i)) for i in range(action_horizon)]

        tbl = self._get_parquet_table(self._data_shard_path(c_idx, f_idx))
        obs_local = [ep_local_start + i for i in obs_indices]
        act_local = [ep_local_start + i for i in act_indices]

        timestamps = np.asarray(
            [float(tbl["timestamp"][i].as_py()) for i in obs_local]
            if "timestamp" in tbl.column_names
            else [i / self._fps for i in obs_indices],
            dtype=np.float64,
        )

        states = (
            np.stack([np.asarray(tbl["observation.state"][i].as_py(), dtype=np.float32) for i in obs_local], axis=0)
            if "observation.state" in tbl.column_names
            else None
        )
        action_chunk = (
            np.stack([np.asarray(tbl["action"][i].as_py(), dtype=np.float32) for i in act_local], axis=0)
            if "action" in tbl.column_names
            else None
        )

        images: dict[str, np.ndarray] = {}
        for vk in self._video_keys:
            vc = int(r[f"videos/{vk}/chunk_index"])
            vf = int(r[f"videos/{vk}/file_index"])
            v_from = float(r[f"videos/{vk}/from_timestamp"])
            mp4_path = self._video_shard_path(vk, vc, vf)
            req_ts = [v_from + idx / self._fps for idx in obs_indices]
            frames = self._decode_video_frames(mp4_path, req_ts)
            images[vk] = np.stack(frames, axis=0)

        return WindowSample(
            episode_id=episode_id,
            step_index=step_index,
            timestamps_s=timestamps,
            images=images,
            states=states,
            action_chunk=action_chunk,
            language_instruction=ep.language_instruction,
        )

    def close(self) -> None:
        for c in self._av_cache.values():
            c.close()
        self._av_cache.clear()
        self._parquet_cache.clear()
