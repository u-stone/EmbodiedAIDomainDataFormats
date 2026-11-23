"""Low-level MCAP format inspector and read-only reader.

Systems-level anatomy of MCAP (used by ROS 2, Foxglove, Rerun):
- Append-only binary container designed for low-latency real-time robot logging.
- Physical layout: [Magic][Header][Data Section: Chunk(compressed messages) / Message]
                   [Summary Section: Schema, Channel, ChunkIndex, Statistics, SummaryOffset]
                   [Footer][Magic]
- Key characteristics:
  1. Multi-channel asynchronous streams with nanosecond log_time / publish_time.
  2. NO native Episode boundary (a .mcap file is a continuous time-series recording).
  3. Reading a synchronized (image, state, action) step requires picking a primary clock
     (usually the main camera topic) and aligning higher-frequency topics via searchsorted.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from mcap.reader import make_reader
from PIL import Image

from embodied_formats.common import (
    BaseFormatReader,
    EpisodeMeta,
    FieldSchema,
    InspectReport,
    StepSample,
    dir_disk_usage,
)


def _decode_image_payload(payload: dict[str, Any]) -> np.ndarray:
    """Decode Foxglove-style JSON CompressedImage (base64 JPEG/PNG) into HxWx3 uint8 RGB."""
    raw = payload.get("data", "")
    img_bytes = base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)
    with Image.open(io.BytesIO(img_bytes)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


class MCAPFormatReader(BaseFormatReader):
    """Inspect and read `.mcap` file(s) using the official low-level `mcap` reader.

    Accepts either a single `.mcap` file or a directory containing multiple `.mcap` files
    (each `.mcap` file is exposed as one logical episode).
    """

    format_name = "mcap"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        if self.path.is_file():
            self._files = [self.path]
        else:
            self._files = sorted(self.path.glob("*.mcap"))
        if not self._files:
            raise FileNotFoundError(f"No .mcap files found under {self.path}")

        # Lazy per-episode parsed topic streams: ep_id -> topic -> (timestamps_ns, decoded_payloads)
        self._ep_cache: dict[int, dict[str, tuple[np.ndarray, list[dict[str, Any]]]]] = {}
        self._episodes: list[EpisodeMeta] | None = None
        self._topic_roles: dict[str, str] = {}

    def _detect_topic_roles(self, topics: list[str]) -> dict[str, str]:
        """Identify primary camera, extra cameras, state, and action topics by convention."""
        roles: dict[str, str] = {}
        cam_topics = [t for t in topics if "image" in t.lower() or "cam" in t.lower()]
        state_topics = [t for t in topics if "state" in t.lower() or "joint" in t.lower() or "odom" in t.lower()]
        action_topics = [t for t in topics if "action" in t.lower() or "cmd" in t.lower() or "command" in t.lower()]

        if cam_topics:
            roles["primary_camera"] = cam_topics[0]
            for idx, ct in enumerate(cam_topics):
                roles[f"camera_{idx}"] = ct
        if state_topics:
            roles["state"] = state_topics[0]
        if action_topics:
            roles["action"] = action_topics[0]
        return roles

    def _load_episode_streams(self, ep_id: int) -> dict[str, tuple[np.ndarray, list[dict[str, Any]]]]:
        if ep_id in self._ep_cache:
            return self._ep_cache[ep_id]

        mcap_path = self._files[ep_id]
        per_topic_ts: dict[str, list[int]] = {}
        per_topic_msg: dict[str, list[dict[str, Any]]] = {}

        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                per_topic_ts.setdefault(topic, []).append(message.log_time)
                if channel.message_encoding == "json":
                    payload = json.loads(message.data.decode("utf-8"))
                else:
                    # Fallback for binary/CDR/Protobuf payloads we don't have schema plugins for
                    payload = {"_raw_bytes_len": len(message.data), "_encoding": channel.message_encoding}
                per_topic_msg.setdefault(topic, []).append(payload)

        streams: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}
        for topic, ts_list in per_topic_ts.items():
            ts_arr = np.asarray(ts_list, dtype=np.int64)
            order = np.argsort(ts_arr, kind="stable")
            streams[topic] = (ts_arr[order], [per_topic_msg[topic][i] for i in order])

        if not self._topic_roles:
            self._topic_roles = self._detect_topic_roles(sorted(streams.keys()))

        self._ep_cache[ep_id] = streams
        return streams

    def list_episodes(self) -> list[EpisodeMeta]:
        if self._episodes is not None:
            return self._episodes

        episodes: list[EpisodeMeta] = []
        for idx, mcap_path in enumerate(self._files):
            with mcap_path.open("rb") as f:
                reader = make_reader(f)
                summary = reader.get_summary()
                stats = summary.statistics if summary else None
                channels = summary.channels if summary else {}

                topics = sorted(ch.topic for ch in channels.values())
                roles = self._detect_topic_roles(topics)
                if not self._topic_roles:
                    self._topic_roles = roles

                primary_topic = roles.get("primary_camera") or (topics[0] if topics else "")
                primary_ch_id = next((cid for cid, ch in channels.items() if ch.topic == primary_topic), None)

                num_steps = 0
                fps = 0.0
                if stats and primary_ch_id is not None:
                    num_steps = int(stats.channel_message_counts.get(primary_ch_id, 0))
                    dur_s = max(1e-6, (stats.message_end_time - stats.message_start_time) / 1e9)
                    fps = (num_steps - 1) / dur_s if num_steps > 1 else 0.0

                lang = ""
                if summary:
                    # Check MCAP metadata records for episode task description
                    for meta_idx in summary.metadata_indexes:
                        if meta_idx.name == "episode_info":
                            pass
                episodes.append(
                    EpisodeMeta(
                        episode_id=idx,
                        num_steps=num_steps,
                        fps=round(fps, 2),
                        language_instruction=lang,
                        extra={
                            "file": mcap_path.name,
                            "primary_clock_topic": primary_topic,
                            "total_messages_all_topics": stats.message_count if stats else 0,
                            "chunk_count": stats.chunk_count if stats else 0,
                        },
                    )
                )
        self._episodes = episodes
        return episodes

    def inspect(self) -> InspectReport:
        disk_bytes, file_count = dir_disk_usage(self.path)
        episodes = self.list_episodes()

        # Deep-inspect the first MCAP file's binary structure & timestamp jitter
        first_file = self._files[0]
        schema_fields: list[FieldSchema] = []
        chunk_compressions: set[str] = set()
        chunk_count_total = 0
        header_info: dict[str, Any] = {}
        jitter_stats: dict[str, dict[str, float]] = {}

        with first_file.open("rb") as f:
            reader = make_reader(f)
            header = reader.get_header()
            header_info = {"profile": header.profile, "library": header.library}
            summary = reader.get_summary()

            if summary:
                for cidx in summary.chunk_indexes:
                    chunk_compressions.add(cidx.compression or "none")
                chunk_count_total = len(summary.chunk_indexes)

                stats = summary.statistics
                for cid, ch in summary.channels.items():
                    sch = summary.schemas.get(ch.schema_id)
                    count = stats.channel_message_counts.get(cid, 0) if stats else 0
                    schema_fields.append(
                        FieldSchema(
                            name=ch.topic,
                            dtype=sch.name if sch else "unknown",
                            shape=f"messages={count}",
                            encoding=ch.message_encoding,
                            details=f"channel_id={cid}, schema_encoding={sch.encoding if sch else 'none'}",
                        )
                    )

        # Measure asynchronous inter-message timestamp jitter on first file
        streams = self._load_episode_streams(0)
        for topic, (ts_ns, _) in streams.items():
            if len(ts_ns) >= 2:
                dts_ms = np.diff(ts_ns) / 1e6
                jitter_stats[topic] = {
                    "hz": round(1000.0 / float(np.mean(dts_ms)), 2),
                    "mean_dt_ms": round(float(np.mean(dts_ms)), 3),
                    "std_dt_ms": round(float(np.std(dts_ms)), 3),
                    "max_dt_ms": round(float(np.max(dts_ms)), 3),
                }

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
                "header": header_info,
                "chunk_count_file0": chunk_count_total,
                "chunk_compressions": sorted(chunk_compressions),
                "topic_roles": self._topic_roles,
                "async_topic_timing": jitter_stats,
            },
            notes=[
                "MCAP 是面向真机低延迟追加写入的容器，原生保存纳秒级异构多通道异步时间戳，没有 Episode 边界概念。",
                "随机读取单个训练 Step 时，必须以主相机为基准时钟，并对高频状态/动作通道做时间戳对齐（searchsorted / 插值）。",
                "Chunk 级压缩（zstd/lz4）+ 文件尾部的 Summary ChunkIndex 支持按时间区间跳过无关 Chunk，但仍不适合直接用于 GPU 高并发随机采样训练。",
            ],
        )

    def read_step(self, episode_id: int, step_index: int) -> StepSample:
        streams = self._load_episode_streams(episode_id)
        primary_topic = self._topic_roles.get("primary_camera")
        if not primary_topic or primary_topic not in streams:
            raise RuntimeError(f"Primary camera topic not found in MCAP streams: {list(streams.keys())}")

        cam_ts, cam_msgs = streams[primary_topic]
        t_ns = int(cam_ts[step_index])
        t_s = float((t_ns - cam_ts[0]) / 1e9)

        images: dict[str, np.ndarray] = {}
        align_skew_ms: dict[str, float] = {}

        for role, topic in self._topic_roles.items():
            if not role.startswith("camera_") or topic not in streams:
                continue
            ts_arr, msgs = streams[topic]
            idx = int(np.clip(np.searchsorted(ts_arr, t_ns), 0, len(ts_arr) - 1))
            if idx > 0 and abs(ts_arr[idx - 1] - t_ns) < abs(ts_arr[idx] - t_ns):
                idx -= 1
            images[topic.strip("/").replace("/", "_")] = _decode_image_payload(msgs[idx])
            align_skew_ms[topic] = round(float(abs(ts_arr[idx] - t_ns) / 1e6), 3)

        state = None
        if "state" in self._topic_roles and self._topic_roles["state"] in streams:
            ts_arr, msgs = streams[self._topic_roles["state"]]
            idx = int(np.clip(np.searchsorted(ts_arr, t_ns), 0, len(ts_arr) - 1))
            if idx > 0 and abs(ts_arr[idx - 1] - t_ns) < abs(ts_arr[idx] - t_ns):
                idx -= 1
            raw_state = msgs[idx].get("position") or msgs[idx].get("state") or msgs[idx].get("qpos")
            if raw_state is not None:
                state = np.asarray(raw_state, dtype=np.float32)
            align_skew_ms[self._topic_roles["state"]] = round(float(abs(ts_arr[idx] - t_ns) / 1e6), 3)

        action = None
        if "action" in self._topic_roles and self._topic_roles["action"] in streams:
            ts_arr, msgs = streams[self._topic_roles["action"]]
            idx = int(np.clip(np.searchsorted(ts_arr, t_ns), 0, len(ts_arr) - 1))
            if idx > 0 and abs(ts_arr[idx - 1] - t_ns) < abs(ts_arr[idx] - t_ns):
                idx -= 1
            raw_act = msgs[idx].get("action") or msgs[idx].get("command") or msgs[idx].get("target")
            if raw_act is not None:
                action = np.asarray(raw_act, dtype=np.float32)
            align_skew_ms[self._topic_roles["action"]] = round(float(abs(ts_arr[idx] - t_ns) / 1e6), 3)

        return StepSample(
            episode_id=episode_id,
            step_index=step_index,
            timestamp_s=t_s,
            images=images,
            state=state,
            action=action,
            extra={"align_skew_ms": align_skew_ms},
        )
