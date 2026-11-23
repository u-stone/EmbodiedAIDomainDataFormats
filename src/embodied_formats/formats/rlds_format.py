"""Low-level RLDS / TFRecord binary inspector and read-only reader (ZERO TensorFlow dependency).

Systems-level anatomy of TFRecord + RLDS (Open X-Embodiment / OpenVLA / Octo / RT-1-X):
- TFRecord binary framing per record:
    [uint64_le length]
    [uint32_le masked_crc32c(length_bytes)]
    [bytes     payload (length bytes: serialized tf.train.Example protobuf)]
    [uint32_le masked_crc32c(payload)]
  where masked_crc32c(x) = (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xFFFFFFFF.
- RLDS encodes ONE ENTIRE EPISODE as a single TFRecord `tf.train.Example` record whose
  feature map flattens nested `steps/*` fields into repeated lists:
    - `steps/observation/image`: BytesList (one JPEG/PNG per step)
    - `steps/observation/state`: FloatList (flattened [T * state_dim])
    - `steps/action`:            FloatList (flattened [T * action_dim])
    - `steps/language_instruction`: BytesList (one UTF-8 string per step)
- Key characteristics:
  1. NO built-in random-access byte-offset index! Pure TFRecord is strictly sequential;
     random access requires scanning headers once to build an external offset index (which
     is why Google later introduced ArrayRecord).
  2. Decoding a single step from a TFRecord RLDS episode requires reading & deserializing
     the ENTIRE episode's Protobuf payload (all steps' JPEG bytes + floats) into memory!
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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


# ---------------------------------------------------------------------------
# Pure-Python CRC32C (Castagnoli, poly 0x82F63B78) + TFRecord Masking
# ---------------------------------------------------------------------------
def _make_crc32c_table() -> list[int]:
    poly = 0x82F63B78
    table: list[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ poly if (crc & 1) else (crc >> 1)
        table.append(crc)
    return table


_CRC32C_TABLE = _make_crc32c_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    table = _CRC32C_TABLE
    for b in data:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    c = crc32c(data)
    return (((c >> 15) | ((c << 17) & 0xFFFFFFFF)) + 0xA282EAD8) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Minimal Protobuf Wire Decoder / Encoder for tf.train.Example
# ---------------------------------------------------------------------------
# Protobuf schema of tf.train.Example:
#   message Example  { Features features = 1; }
#   message Features { map<string, Feature> feature = 1; }
#     (map entry: string key = 1; Feature value = 2;)
#   message Feature  {
#     oneof kind {
#       BytesList bytes_list = 1;   // repeated bytes value = 1;
#       FloatList float_list = 2;   // repeated float value = 1 [packed=true];
#       Int64List int64_list = 3;   // repeated int64 value = 1 [packed=true];
#     }
#   }
def _read_varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _encode_varint(val: int) -> bytes:
    if val < 0:
        val &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _iter_pb_fields(buf: memoryview) -> list[tuple[int, int, Any]]:
    """Decode top-level Protobuf wire fields: returns list of (field_num, wire_type, value)."""
    fields: list[tuple[int, int, Any]] = []
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:  # varint
            val, pos = _read_varint(buf, pos)
            fields.append((field_num, wire_type, val))
        elif wire_type == 1:  # 64-bit
            fields.append((field_num, wire_type, bytes(buf[pos : pos + 8])))
            pos += 8
        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(buf, pos)
            fields.append((field_num, wire_type, buf[pos : pos + length]))
            pos += length
        elif wire_type == 5:  # 32-bit
            fields.append((field_num, wire_type, bytes(buf[pos : pos + 4])))
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire_type={wire_type} at pos={pos}")
    return fields


def decode_tf_example(payload: bytes) -> dict[str, tuple[str, Any]]:
    """Decode a serialized `tf.train.Example` into `{key: (kind, values)}`.

    `kind` is one of `'bytes'`, `'float'`, `'int64'`.
    """
    mv = memoryview(payload)
    features_dict: dict[str, tuple[str, Any]] = {}
    for fnum, wtype, val in _iter_pb_fields(mv):
        if fnum != 1 or wtype != 2:
            continue
        # Features message: repeated map entry (field 1)
        for mfnum, mwtype, entry_mv in _iter_pb_fields(val):
            if mfnum != 1 or mwtype != 2:
                continue
            key = ""
            feat_mv = None
            for efnum, ewtype, eval_ in _iter_pb_fields(entry_mv):
                if efnum == 1 and ewtype == 2:
                    key = bytes(eval_).decode("utf-8")
                elif efnum == 2 and ewtype == 2:
                    feat_mv = eval_
            if feat_mv is None:
                continue
            # Feature oneof: 1=BytesList, 2=FloatList, 3=Int64List
            for kfnum, kwtype, kval in _iter_pb_fields(feat_mv):
                if kwtype != 2:
                    continue
                if kfnum == 1:
                    items = [bytes(v) for fn, wt, v in _iter_pb_fields(kval) if fn == 1 and wt == 2]
                    features_dict[key] = ("bytes", items)
                elif kfnum == 2:
                    floats: list[float] = []
                    for fn, wt, v in _iter_pb_fields(kval):
                        if fn == 1 and wt == 2:  # packed float32
                            floats.extend(np.frombuffer(v, dtype="<f4").tolist())
                        elif fn == 1 and wt == 5:  # unpacked float32
                            floats.append(struct.unpack("<f", v)[0])
                    features_dict[key] = ("float", np.asarray(floats, dtype=np.float32))
                elif kfnum == 3:
                    ints: list[int] = []
                    for fn, wt, v in _iter_pb_fields(kval):
                        if fn == 1 and wt == 2:  # packed varint
                            p = 0
                            while p < len(v):
                                iv, p = _read_varint(v, p)
                                ints.append(iv)
                        elif fn == 1 and wt == 0:
                            ints.append(v)
                    features_dict[key] = ("int64", np.asarray(ints, dtype=np.int64))
    return features_dict


def encode_tf_example(features: dict[str, tuple[str, Any]]) -> bytes:
    """Encode `{key: (kind, values)}` into a valid `tf.train.Example` protobuf payload."""
    entries = bytearray()
    for key, (kind, values) in features.items():
        if kind == "bytes":
            bl = bytearray()
            for item in values:
                b = item.encode("utf-8") if isinstance(item, str) else bytes(item)
                bl.extend(b"\x0a" + _encode_varint(len(b)) + b)
            feat = b"\x0a" + _encode_varint(len(bl)) + bytes(bl)
        elif kind == "float":
            arr = np.asarray(values, dtype="<f4").tobytes()
            fl = b"\x0a" + _encode_varint(len(arr)) + arr
            feat = b"\x12" + _encode_varint(len(fl)) + fl
        elif kind == "int64":
            packed = bytearray()
            for v in np.asarray(values, dtype=np.int64).tolist():
                packed.extend(_encode_varint(int(v)))
            il = b"\x0a" + _encode_varint(len(packed)) + bytes(packed)
            feat = b"\x1a" + _encode_varint(len(il)) + il
        else:
            raise ValueError(f"Unknown feature kind: {kind}")

        kb = key.encode("utf-8")
        entry = b"\x0a" + _encode_varint(len(kb)) + kb + b"\x12" + _encode_varint(len(feat)) + feat
        entries.extend(b"\x0a" + _encode_varint(len(entry)) + entry)

    return b"\x0a" + _encode_varint(len(entries)) + bytes(entries)


def write_tfrecord_item(f: Any, payload: bytes) -> None:
    """Write a single CRC32C-protected TFRecord frame."""
    len_bytes = struct.pack("<Q", len(payload))
    f.write(len_bytes)
    f.write(struct.pack("<I", masked_crc32c(len_bytes)))
    f.write(payload)
    f.write(struct.pack("<I", masked_crc32c(payload)))


class RLDSFormatReader(BaseFormatReader):
    """Pure-Python binary inspector & reader for RLDS TFRecord shards."""

    format_name = "rlds"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        if self.path.is_file():
            self._shards = [self.path]
            self._meta_dir = self.path.parent
        else:
            self._shards = sorted(
                p for p in self.path.rglob("*") if p.is_file() and ".tfrecord" in p.name
            )
            self._meta_dir = self.path
        if not self._shards:
            raise FileNotFoundError(f"No .tfrecord* files found under {self.path}")

        self._dataset_info: dict[str, Any] = {}
        info_path = self._meta_dir / "dataset_info.json"
        if info_path.exists():
            self._dataset_info = json.loads(info_path.read_text(encoding="utf-8"))

        # External byte-offset index built by scanning TFRecord framing headers:
        # list of (shard_path, payload_offset, payload_len, crc_ok)
        self._index: list[tuple[Path, int, int, bool]] = []
        self._episodes: list[EpisodeMeta] | None = None
        self._last_ep_cache: tuple[int, dict[str, tuple[str, Any]]] | None = None
        self._build_offset_index()

    def _build_offset_index(self) -> None:
        """Scan TFRecord headers sequentially to build an in-memory random-access index."""
        self._index.clear()
        for shard in self._shards:
            with shard.open("rb") as f:
                while True:
                    hdr = f.read(12)
                    if not hdr:
                        break
                    if len(hdr) < 12:
                        break
                    length = struct.unpack("<Q", hdr[:8])[0]
                    stored_len_crc = struct.unpack("<I", hdr[8:12])[0]
                    len_crc_ok = stored_len_crc == masked_crc32c(hdr[:8])
                    payload_offset = f.tell()
                    f.seek(length, io.SEEK_CUR)
                    footer = f.read(4)
                    if len(footer) < 4:
                        break
                    self._index.append((shard, payload_offset, length, len_crc_ok))

    def _read_episode_dict(self, episode_id: int, verify_payload_crc: bool = False) -> dict[str, tuple[str, Any]]:
        if self._last_ep_cache is not None and self._last_ep_cache[0] == episode_id and not verify_payload_crc:
            return self._last_ep_cache[1]

        shard, offset, length, _ = self._index[episode_id]
        with shard.open("rb") as f:
            f.seek(offset)
            payload = f.read(length)
            if verify_payload_crc:
                stored_crc = struct.unpack("<I", f.read(4))[0]
                if stored_crc != masked_crc32c(payload):
                    raise ValueError(f"TFRecord payload CRC32C mismatch at episode {episode_id}")
        decoded = decode_tf_example(payload)
        self._last_ep_cache = (episode_id, decoded)
        return decoded

    def _infer_step_count(self, ep_dict: dict[str, tuple[str, Any]]) -> int:
        for k in ("steps/is_first", "steps/is_last", "steps/reward", "steps/discount"):
            if k in ep_dict:
                return len(ep_dict[k][1])
        for k, (kind, vals) in ep_dict.items():
            if k.startswith("steps/observation/") and kind == "bytes":
                return len(vals)
        return 0

    def list_episodes(self) -> list[EpisodeMeta]:
        if self._episodes is not None:
            return self._episodes
        fps = float(self._dataset_info.get("fps", 5.0))
        eps: list[EpisodeMeta] = []
        for idx in range(len(self._index)):
            ep_dict = self._read_episode_dict(idx)
            n = self._infer_step_count(ep_dict)
            lang = ""
            if "steps/language_instruction" in ep_dict and ep_dict["steps/language_instruction"][1]:
                raw_lang = ep_dict["steps/language_instruction"][1][0]
                lang = raw_lang.decode("utf-8", errors="replace") if isinstance(raw_lang, bytes) else str(raw_lang)
            shard, offset, length, crc_ok = self._index[idx]
            eps.append(
                EpisodeMeta(
                    episode_id=idx,
                    num_steps=n,
                    fps=fps,
                    language_instruction=lang,
                    extra={
                        "shard": shard.name,
                        "byte_offset": offset,
                        "record_bytes": length,
                        "header_crc32c_ok": crc_ok,
                    },
                )
            )
        self._episodes = eps
        return eps

    def inspect(self) -> InspectReport:
        disk_bytes, file_count = dir_disk_usage(self.path)
        episodes = self.list_episodes()

        # Verify full CRC32C on first episode and inspect schema
        ep0 = self._read_episode_dict(0, verify_payload_crc=True)
        n0 = max(1, self._infer_step_count(ep0))

        schema_fields: list[FieldSchema] = []
        for key in sorted(ep0.keys()):
            kind, vals = ep0[key]
            if kind == "bytes":
                avg_b = int(np.mean([len(x) for x in vals])) if vals else 0
                is_jpeg = bool(vals and vals[0][:2] == b"\xff\xd8")
                is_png = bool(vals and vals[0][:4] == b"\x89PNG")
                enc = "JPEG" if is_jpeg else ("PNG" if is_png else "utf8/raw-bytes")
                schema_fields.append(
                    FieldSchema(
                        name=key,
                        dtype="bytes_list",
                        shape=f"len={len(vals)}",
                        encoding=enc,
                        details=f"avg_item={format_bytes(avg_b)}",
                    )
                )
            elif kind == "float":
                per_step = len(vals) // n0 if len(vals) % n0 == 0 else -1
                shape_str = f"({n0}, {per_step})" if per_step > 0 else f"({len(vals)},)"
                schema_fields.append(
                    FieldSchema(
                        name=key,
                        dtype="float32 (packed)",
                        shape=shape_str,
                        encoding="protobuf",
                        details=f"flat_len={len(vals)}",
                    )
                )
            else:
                schema_fields.append(
                    FieldSchema(
                        name=key,
                        dtype="int64 (varint)",
                        shape=f"({len(vals)},)",
                        encoding="protobuf",
                        details=f"len={len(vals)}",
                    )
                )

        rec_sizes = [x[2] for x in self._index]
        crc_ok_count = sum(1 for x in self._index if x[3])
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
                "tfrecord_shards": len(self._shards),
                "header_crc32c_verified": f"{crc_ok_count}/{len(self._index)} OK (payload CRC32C ep0 OK)",
                "record_size_min": format_bytes(min(rec_sizes)),
                "record_size_mean": format_bytes(float(np.mean(rec_sizes))),
                "record_size_max": format_bytes(max(rec_sizes)),
                "has_dataset_info_json": bool(self._dataset_info),
                "native_random_access_index": False,
            },
            notes=[
                "TFRecord 本质是无索引的顺序二进制日志（[uint64 len][masked_crc32c][protobuf][masked_crc32c]），随机寻址必须先扫描建立外部偏移索引（这也是 ArrayRecord 诞生的原因）。",
                "RLDS 将整条 Episode 嵌套打包为一个 Protobuf Example，逐帧图像单独 JPEG 编码；因此随机抽 1 帧往往需要读取并反序列化整条轨迹的 Protobuf 载荷。",
                "优势在于极其统一的 `Episode → Steps` 嵌套规范，Open X-Embodiment / OpenVLA 跨机器人混训生态最成熟。",
            ],
        )

    def read_step(self, episode_id: int, step_index: int) -> StepSample:
        ep = self.list_episodes()[episode_id]
        ep_dict = self._read_episode_dict(episode_id)
        n = ep.num_steps

        images: dict[str, np.ndarray] = {}
        for k, (kind, vals) in ep_dict.items():
            if k.startswith("steps/observation/") and kind == "bytes" and len(vals) == n:
                raw = vals[step_index]
                if raw[:2] == b"\xff\xd8" or raw[:4] == b"\x89PNG":
                    cam_name = k.removeprefix("steps/observation/")
                    with Image.open(io.BytesIO(raw)) as im:
                        images[cam_name] = np.asarray(im.convert("RGB"), dtype=np.uint8)

        state = None
        for sk in ("steps/observation/state", "steps/observation/proprio"):
            if sk in ep_dict and ep_dict[sk][0] == "float":
                arr = ep_dict[sk][1]
                dim = len(arr) // n
                state = arr[step_index * dim : (step_index + 1) * dim].copy()
                break

        action = None
        if "steps/action" in ep_dict and ep_dict["steps/action"][0] == "float":
            arr = ep_dict["steps/action"][1]
            dim = len(arr) // n
            action = arr[step_index * dim : (step_index + 1) * dim].copy()

        return StepSample(
            episode_id=episode_id,
            step_index=step_index,
            timestamp_s=step_index / max(ep.fps, 1e-6),
            images=images,
            state=state,
            action=action,
            language_instruction=ep.language_instruction,
        )

    def read_window(
        self,
        episode_id: int,
        step_index: int,
        obs_horizon: int = 2,
        action_horizon: int = 16,
    ) -> WindowSample:
        """Read window from a single decoded RLDS Episode Protobuf (1 episode decode per window)."""
        ep = self.list_episodes()[episode_id]
        ep_dict = self._read_episode_dict(episode_id)
        n = ep.num_steps

        obs_indices = [max(0, min(n - 1, step_index - (obs_horizon - 1 - i))) for i in range(obs_horizon)]
        act_indices = [max(0, min(n - 1, step_index + i)) for i in range(action_horizon)]
        timestamps = np.asarray(obs_indices, dtype=np.float64) / max(ep.fps, 1e-6)

        images: dict[str, np.ndarray] = {}
        for k, (kind, vals) in ep_dict.items():
            if k.startswith("steps/observation/") and kind == "bytes" and len(vals) == n:
                if vals[0][:2] == b"\xff\xd8" or vals[0][:4] == b"\x89PNG":
                    cam_name = k.removeprefix("steps/observation/")
                    frames = []
                    for idx in obs_indices:
                        with Image.open(io.BytesIO(vals[idx])) as im:
                            frames.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
                    images[cam_name] = np.stack(frames, axis=0)

        states = None
        for sk in ("steps/observation/state", "steps/observation/proprio"):
            if sk in ep_dict and ep_dict[sk][0] == "float":
                arr = ep_dict[sk][1].reshape(n, -1)
                states = arr[obs_indices].copy()
                break

        action_chunk = None
        if "steps/action" in ep_dict and ep_dict["steps/action"][0] == "float":
            arr = ep_dict["steps/action"][1].reshape(n, -1)
            action_chunk = arr[act_indices].copy()

        return WindowSample(
            episode_id=episode_id,
            step_index=step_index,
            timestamps_s=timestamps,
            images=images,
            states=states,
            action_chunk=action_chunk,
            language_instruction=ep.language_instruction,
        )
