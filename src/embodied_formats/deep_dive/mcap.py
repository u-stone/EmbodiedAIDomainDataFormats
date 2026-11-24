"""Ultra-deep systems anatomy & verification lab for ROS 2 bag / MCAP (ZERO ROS 2 install needed).

Designed for genuine ROS 2 production recordings (e.g. `RobotisAI/evButtonPush-260608-0-MCAP`):
  1. MCAP binary container & ChunkIndex anatomy (Footer -> Summary -> all 197 ChunkIndexes:
     chunk size distribution, time span per chunk, compression codec).
  2. Pure-Python ROS 2 CDR (Common Data Representation) wire decoder:
     - Parses 4-byte CDR encapsulation header (`00 01 00 00` = CDR_LE)
     - Decodes real `sensor_msgs/msg/CompressedImage` -> extracts embedded JPEG resolution!
     - Decodes real `sensor_msgs/msg/JointState` -> extracts exact joint names & float64 positions!
  3. Real multi-sensor asynchronous clock jitter & cross-topic alignment skew profiler:
     computes inter-message `dt` mean/std/p99/max for all 6 topics, then aligns all 5
     secondary topics against the primary camera clock to quantify real-world hardware skew!
  4. Sidecar metadata unification (`metadata.yaml` + `episode_info.json` + `robot.urdf` XML).
"""

from __future__ import annotations

import io
import json
import struct
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from mcap.reader import make_reader
from PIL import Image

from embodied_formats.common import format_bytes


def _find_mcap_file(root: Path) -> Path:
    if root.is_file():
        return root
    cands = sorted(root.glob("*.mcap"))
    if not cands:
        raise FileNotFoundError(f"No .mcap file found under {root}")
    return cands[0]


# ---------------------------------------------------------------------------
# Pure-Python ROS 2 CDR (CDR_LE) Binary Wire Reader
# ---------------------------------------------------------------------------
class CDRReader:
    """Minimal OMG CDR Little-Endian deserializer matching ROS 2 rmw_fastrtps/cyclonedds."""

    def __init__(self, data: bytes) -> None:
        self.buf = data
        if len(data) < 4:
            raise ValueError("CDR buffer too short for 4-byte encapsulation header")
        # Bytes 0..1: representation identifier (0x00 0x01 = CDR_LE, 0x00 0x00 = CDR_BE)
        self.endian = "<" if data[1] == 0x01 else ">"
        self.origin = 4
        self.pos = 4

    def _align(self, n: int) -> None:
        rel = self.pos - self.origin
        rem = rel % n
        if rem:
            self.pos += n - rem

    def read_u32(self) -> int:
        self._align(4)
        v = struct.unpack_from(f"{self.endian}I", self.buf, self.pos)[0]
        self.pos += 4
        return int(v)

    def read_i32(self) -> int:
        self._align(4)
        v = struct.unpack_from(f"{self.endian}i", self.buf, self.pos)[0]
        self.pos += 4
        return int(v)

    def read_string(self) -> str:
        n = self.read_u32()
        raw = self.buf[self.pos : self.pos + n]
        self.pos += n
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    def read_bytes_seq(self) -> bytes:
        n = self.read_u32()
        raw = self.buf[self.pos : self.pos + n]
        self.pos += n
        return raw

    def read_f64_seq(self) -> list[float]:
        n = self.read_u32()
        if n == 0:
            return []
        self._align(8)
        vals = list(struct.unpack_from(f"{self.endian}{n}d", self.buf, self.pos))
        self.pos += n * 8
        return vals


def decode_ros2_compressed_image_cdr(data: bytes) -> dict[str, Any]:
    """Decode `sensor_msgs/msg/CompressedImage` directly from ROS 2 CDR bytes."""
    r = CDRReader(data)
    sec = r.read_i32()
    nanosec = r.read_u32()
    frame_id = r.read_string()
    fmt = r.read_string()
    img_bytes = r.read_bytes_seq()
    with Image.open(io.BytesIO(img_bytes)) as im:
        w, h, mode = im.width, im.height, im.mode
    return {
        "cdr_header_hex": " ".join(f"{b:02X}" for b in data[:4]),
        "stamp_s": round(sec + nanosec * 1e-9, 6),
        "frame_id": frame_id,
        "format": fmt,
        "compressed_bytes": len(img_bytes),
        "decoded_resolution": f"{w}x{h} ({mode})",
        "magic_hex": " ".join(f"{b:02X}" for b in img_bytes[:4]),
    }


def decode_ros2_joint_state_cdr(data: bytes) -> dict[str, Any]:
    """Decode `sensor_msgs/msg/JointState` directly from ROS 2 CDR bytes."""
    r = CDRReader(data)
    sec = r.read_i32()
    nanosec = r.read_u32()
    frame_id = r.read_string()
    n_names = r.read_u32()
    names = [r.read_string() for _ in range(n_names)]
    positions = r.read_f64_seq()
    velocities = r.read_f64_seq()
    efforts = r.read_f64_seq()
    return {
        "cdr_header_hex": " ".join(f"{b:02X}" for b in data[:4]),
        "stamp_s": round(sec + nanosec * 1e-9, 6),
        "frame_id": frame_id,
        "joint_names": names,
        "positions_rad": [round(x, 4) for x in positions],
        "velocity_len": len(velocities),
        "effort_len": len(efforts),
    }


def analyze_mcap_chunks_and_cdr(mcap_path: Path) -> dict[str, Any]:
    """Analyze all ChunkIndexes + decode real ROS 2 CDR sample messages (Camera + JointState)."""
    with mcap_path.open("rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        chunks = summary.chunk_indexes if summary else []
        chunk_sizes = [c.compressed_size for c in chunks]
        chunk_spans_ms = [(c.message_end_time - c.message_start_time) / 1e6 for c in chunks]
        codecs = Counter(c.compression or "none" for c in chunks)

        cdr_samples: dict[str, Any] = {}
        for schema, channel, message in reader.iter_messages():
            if schema is None:
                continue
            if schema.name == "sensor_msgs/msg/CompressedImage" and channel.topic not in cdr_samples:
                cdr_samples[channel.topic] = decode_ros2_compressed_image_cdr(message.data)
            elif schema.name == "sensor_msgs/msg/JointState" and channel.topic not in cdr_samples:
                cdr_samples[channel.topic] = decode_ros2_joint_state_cdr(message.data)
            if len(cdr_samples) >= 5:
                break

    return {
        "file": mcap_path.name,
        "chunk_count": len(chunks),
        "chunk_codecs": dict(codecs),
        "chunk_size_min_median_max": [
            format_bytes(min(chunk_sizes)),
            format_bytes(float(np.median(chunk_sizes))),
            format_bytes(max(chunk_sizes)),
        ],
        "chunk_time_span_ms_min_median_max": [
            round(min(chunk_spans_ms), 2),
            round(float(np.median(chunk_spans_ms)), 2),
            round(max(chunk_spans_ms), 2),
        ],
        "decoded_ros2_cdr_samples": cdr_samples,
    }


def analyze_async_clocks_and_alignment(mcap_path: Path) -> dict[str, Any]:
    """Measure inter-message timestamp jitter across all topics & nearest-neighbor alignment skew."""
    topic_ts: dict[str, list[int]] = {}
    with mcap_path.open("rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages():
            topic_ts.setdefault(channel.topic, []).append(message.log_time)

    per_topic_jitter: dict[str, Any] = {}
    sorted_ts: dict[str, np.ndarray] = {}
    for topic, ts_list in sorted(topic_ts.items()):
        arr = np.sort(np.asarray(ts_list, dtype=np.int64))
        sorted_ts[topic] = arr
        dts_ms = np.diff(arr) / 1e6
        per_topic_jitter[topic] = {
            "count": len(arr),
            "effective_hz": round(1000.0 / max(float(np.mean(dts_ms)), 1e-9), 2) if len(dts_ms) else 0.0,
            "dt_mean_ms": round(float(np.mean(dts_ms)), 3) if len(dts_ms) else 0.0,
            "dt_std_ms": round(float(np.std(dts_ms)), 3) if len(dts_ms) else 0.0,
            "dt_p99_ms": round(float(np.percentile(dts_ms, 99)), 3) if len(dts_ms) else 0.0,
            "dt_max_ms": round(float(np.max(dts_ms)), 3) if len(dts_ms) else 0.0,
        }

    # Pick primary camera topic (prefer cam_top or first camera topic)
    cam_topics = [t for t in sorted_ts if "camera" in t.lower() or "image" in t.lower()]
    primary = next((t for t in cam_topics if "top" in t.lower()), cam_topics[0])
    ref_ts = sorted_ts[primary]

    alignment_skew: dict[str, Any] = {}
    for topic, arr in sorted_ts.items():
        if topic == primary:
            continue
        idxs = np.clip(np.searchsorted(arr, ref_ts), 0, len(arr) - 1)
        prev_idxs = np.clip(idxs - 1, 0, len(arr) - 1)
        d_cur = np.abs(arr[idxs] - ref_ts)
        d_prev = np.abs(arr[prev_idxs] - ref_ts)
        best_ms = np.minimum(d_cur, d_prev) / 1e6
        alignment_skew[topic] = {
            "skew_mean_ms": round(float(np.mean(best_ms)), 3),
            "skew_p95_ms": round(float(np.percentile(best_ms, 95)), 3),
            "skew_max_ms": round(float(np.max(best_ms)), 3),
        }

    return {
        "primary_clock_topic": primary,
        "primary_clock_frames": len(ref_ts),
        "per_topic_timing_jitter": per_topic_jitter,
        "alignment_skew_against_primary_camera": alignment_skew,
    }


def analyze_mcap_sidecars(root: Path) -> dict[str, Any]:
    """Parse ROS 2 metadata.yaml, episode_info.json, and robot.urdf XML kinematics."""
    out: dict[str, Any] = {}
    if not root.is_dir():
        return out

    meta_yaml = root / "metadata.yaml"
    if meta_yaml.exists():
        y = yaml.safe_load(meta_yaml.read_text(encoding="utf-8"))
        bag_info = y.get("rosbag2_bagfile_information", {})
        out["metadata_yaml"] = {
            "version": bag_info.get("version"),
            "storage_identifier": bag_info.get("storage_identifier"),
            "duration_s": round(bag_info.get("duration", {}).get("nanoseconds", 0) / 1e9, 3),
            "message_count": bag_info.get("message_count"),
            "num_topics": len(bag_info.get("topics_with_message_count", [])),
        }

    ep_json = root / "episode_info.json"
    if ep_json.exists():
        out["episode_info_json"] = json.loads(ep_json.read_text(encoding="utf-8"))

    urdf_path = root / "robot.urdf"
    if urdf_path.exists():
        el = ET.fromstring(urdf_path.read_text(encoding="utf-8"))
        joints = el.findall("joint")
        joint_types = Counter(j.attrib.get("type", "unknown") for j in joints)
        out["robot_urdf"] = {
            "robot_name": el.attrib.get("name", ""),
            "link_count": len(el.findall("link")),
            "joint_count": len(joints),
            "joint_types": dict(joint_types),
            "revolute_or_prismatic_joints": [
                j.attrib.get("name", "") for j in joints if j.attrib.get("type") in ("revolute", "prismatic", "continuous")
            ],
        }
    return out


def run_full_mcap_deep_study(root: Path) -> dict[str, Any]:
    mcap_path = _find_mcap_file(root)
    return {
        "dataset_path": str(root),
        "chunks_and_ros2_cdr_decode": analyze_mcap_chunks_and_cdr(mcap_path),
        "async_clocks_and_alignment": analyze_async_clocks_and_alignment(mcap_path),
        "sidecar_metadata_and_urdf": analyze_mcap_sidecars(root),
    }
