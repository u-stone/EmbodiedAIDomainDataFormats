"""Low-level Zarr format inspector and read-only reader (Diffusion Policy ReplayBuffer convention).

Systems-level anatomy of Zarr (v2):
- Directory-tree chunked array store: every N-D chunk is stored as an independent file on disk
  (or object in S3/GCS) with a tiny JSON header (`.zarray`, `.zgroup`, `.zattrs`).
- Key characteristics:
  1. Lock-free multi-process / multi-thread concurrent random reads (each chunk is a separate file).
  2. Flat concatenated arrays (`data/img`, `data/state`, `data/action`) + `meta/episode_ends`
     integer boundary vector — zero-copy virtual episode slicing!
  3. Blosc (LZ4/Zstd + BitShuffle) gives fast decompression, but like HDF5 lacks inter-frame
     video compression, so high-res multi-camera datasets consume large disk space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

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


class ZarrFormatReader(BaseFormatReader):
    """Inspect and read Zarr v2 ReplayBuffer stores directly via `zarr`."""

    format_name = "zarr"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._root = zarr.open_group(str(self.path), mode="r")
        self._data_grp = self._root["data"] if "data" in self._root else self._root
        self._meta_grp = self._root["meta"] if "meta" in self._root else None

        if self._meta_grp is not None and "episode_ends" in self._meta_grp:
            self._episode_ends = np.asarray(self._meta_grp["episode_ends"][:], dtype=np.int64)
        else:
            # Fallback: treat entire array as 1 episode
            first_arr = next(iter(self._data_grp.arrays()))[1]
            self._episode_ends = np.asarray([first_arr.shape[0]], dtype=np.int64)

        self._cameras: dict[str, zarr.Array] = {}
        self._state_arr: zarr.Array | None = None
        self._action_arr: zarr.Array | None = None

        for name, arr in self._data_grp.arrays():
            nl = name.lower()
            if "img" in nl or "image" in nl or "rgb" in nl or (arr.ndim == 4 and arr.shape[-1] == 3):
                self._cameras[name] = arr
            elif self._state_arr is None and any(tok in nl for tok in ("state", "qpos", "pos", "obs")):
                self._state_arr = arr
            elif self._action_arr is None and "act" in nl:
                self._action_arr = arr

        self._episodes: list[EpisodeMeta] | None = None

    def _ep_bounds(self, episode_id: int) -> tuple[int, int]:
        start = 0 if episode_id == 0 else int(self._episode_ends[episode_id - 1])
        end = int(self._episode_ends[episode_id])
        return start, end

    def list_episodes(self) -> list[EpisodeMeta]:
        if self._episodes is not None:
            return self._episodes
        fps = float(self._root.attrs.get("fps", 10.0))
        lang = str(self._root.attrs.get("language_instruction", ""))
        eps: list[EpisodeMeta] = []
        prev = 0
        for idx, end in enumerate(self._episode_ends):
            end_i = int(end)
            eps.append(
                EpisodeMeta(
                    episode_id=idx,
                    num_steps=end_i - prev,
                    fps=fps,
                    language_instruction=lang,
                    extra={"global_start_idx": prev, "global_end_idx": end_i},
                )
            )
            prev = end_i
        self._episodes = eps
        return eps

    def inspect(self) -> InspectReport:
        disk_bytes, file_count = dir_disk_usage(self.path)
        episodes = self.list_episodes()

        schema_fields: list[FieldSchema] = []
        total_logical = 0
        total_stored = 0
        total_chunks_init = 0
        compressors: set[str] = set()

        def _walk(prefix: str, grp: zarr.Group) -> None:
            nonlocal total_logical, total_stored, total_chunks_init
            for name, arr in grp.arrays():
                full_name = f"{prefix}/{name}" if prefix else name
                logical = int(arr.nbytes)
                stored = int(arr.nbytes_stored)
                ratio = (logical / stored) if stored > 0 else 1.0
                comp_str = str(arr.compressor) if arr.compressor is not None else "none"
                compressors.add(comp_str)
                total_logical += logical
                total_stored += stored
                total_chunks_init += int(arr.nchunks_initialized)
                schema_fields.append(
                    FieldSchema(
                        name=full_name,
                        dtype=str(arr.dtype),
                        shape=str(arr.shape),
                        encoding=comp_str,
                        details=(
                            f"chunks={arr.chunks}, init_chunks={arr.nchunks_initialized}/{arr.nchunks}, "
                            f"logical={format_bytes(logical)}, stored={format_bytes(stored)} ({ratio:.2f}x)"
                        ),
                    )
                )
            for name, sub in grp.groups():
                _walk(f"{prefix}/{name}" if prefix else name, sub)

        _walk("", self._root)
        overall_ratio = (total_logical / total_stored) if total_stored > 0 else 1.0
        total_steps = sum(ep.num_steps for ep in episodes)
        avg_fps = float(np.mean([ep.fps for ep in episodes])) if episodes else 0.0

        return InspectReport(
            format_name=self.format_name,
            path=str(self.path),
            disk_bytes=disk_bytes,
            file_count=file_count,
            num_episodes=len(episodes),
            total_steps=total_steps,
            avg_fps=round(avg_fps, 2),
            schema=schema_fields,
            storage_details={
                "zarr_version": zarr.__version__,
                "total_logical_bytes": format_bytes(total_logical),
                "total_stored_bytes": format_bytes(total_stored),
                "overall_compression_ratio": f"{overall_ratio:.2f}x",
                "total_chunk_files": total_chunks_init,
                "compressors": sorted(compressors),
                "episode_index_strategy": "flat contiguous arrays + meta/episode_ends boundary vector",
            },
            notes=[
                "Zarr 把多维数组的每个 Chunk 存成独立文件（配合 .zarray 元数据），天然支持多进程/多线程无锁高并发随机读，也直接适配 S3/OSS 对象存储。",
                "Diffusion Policy 经典的 ReplayBuffer 布局将所有轨迹首尾拼接为扁平数组，仅用 meta/episode_ends 记录切分点，实现零拷贝切片与极速跨轨迹随机采样。",
                "缺点与 HDF5 类似：无帧间视频压缩，高清多视角图像靠 Blosc(LZ4/Zstd) 单帧压缩时体积依然偏大。",
            ],
        )

    def read_step(self, episode_id: int, step_index: int) -> StepSample:
        start, end = self._ep_bounds(episode_id)
        g_idx = start + step_index
        if g_idx >= end:
            raise IndexError(f"step_index {step_index} out of bounds for episode {episode_id} (len={end - start})")

        ep = self.list_episodes()[episode_id]
        images = {
            cam: np.asarray(arr[g_idx], dtype=np.uint8)
            for cam, arr in self._cameras.items()
        }
        state = np.asarray(self._state_arr[g_idx], dtype=np.float32) if self._state_arr is not None else None
        action = np.asarray(self._action_arr[g_idx], dtype=np.float32) if self._action_arr is not None else None

        return StepSample(
            episode_id=episode_id,
            step_index=step_index,
            timestamp_s=step_index / max(ep.fps, 1e-6),
            images=images,
            state=state,
            action=action,
            language_instruction=ep.language_instruction,
            extra={"global_flat_index": g_idx},
        )

    def read_window(
        self,
        episode_id: int,
        step_index: int,
        obs_horizon: int = 2,
        action_horizon: int = 16,
    ) -> WindowSample:
        """Native Zarr chunk-slice window read with episode boundary clamping."""
        start, end = self._ep_bounds(episode_id)
        ep = self.list_episodes()[episode_id]
        n = end - start

        obs_s = max(0, step_index - obs_horizon + 1)
        obs_e = min(n, step_index + 1)
        pad_left = obs_horizon - (obs_e - obs_s)

        act_s = step_index
        act_e = min(n, step_index + action_horizon)
        pad_right = action_horizon - (act_e - act_s)

        obs_idx = [0] * pad_left + list(range(obs_s, obs_e))
        timestamps = np.asarray(obs_idx, dtype=np.float64) / max(ep.fps, 1e-6)

        g_obs_s, g_obs_e = start + obs_s, start + obs_e
        g_act_s, g_act_e = start + act_s, start + act_e

        images: dict[str, np.ndarray] = {}
        for cam, arr in self._cameras.items():
            sl = np.asarray(arr[g_obs_s:g_obs_e], dtype=np.uint8)
            if pad_left > 0:
                sl = np.concatenate([np.repeat(sl[:1], pad_left, axis=0), sl], axis=0)
            images[cam] = sl

        states = None
        if self._state_arr is not None:
            st = np.asarray(self._state_arr[g_obs_s:g_obs_e], dtype=np.float32)
            if pad_left > 0:
                st = np.concatenate([np.repeat(st[:1], pad_left, axis=0), st], axis=0)
            states = st

        action_chunk = None
        if self._action_arr is not None:
            ac = np.asarray(self._action_arr[g_act_s:g_act_e], dtype=np.float32)
            if pad_right > 0:
                ac = np.concatenate([ac, np.repeat(ac[-1:], pad_right, axis=0)], axis=0)
            action_chunk = ac

        return WindowSample(
            episode_id=episode_id,
            step_index=step_index,
            timestamps_s=timestamps,
            images=images,
            states=states,
            action_chunk=action_chunk,
            language_instruction=ep.language_instruction,
        )
