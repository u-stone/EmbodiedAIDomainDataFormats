"""Ultra-deep systems anatomy & verification lab for RLDS / TFRecord (ZERO TensorFlow dependency).

Designed for genuine Open X-Embodiment TFDS shards (e.g. `berkeley_cable_routing` shard-00004):
  1. Full-shard binary framing & Castagnoli Masked CRC32C verification across ALL records.
  2. Protobuf `tf.train.Example` wire breakdown (image PNG/JPEG share vs embeddings vs state/action)
     + raw PNG/JPEG header parser to verify embedded resolution & color depth.
  3. TFDS `features.json` recursive schema tree parser showing how flat 1D Protobuf lists
     reshape back into nested (T, H, W, 3) / (T, D) tensors.
  4. Random-access degradation profiler: compares Cached-Episode vs No-Cache (true i.i.d.
     cross-episode random single-step sampling) to quantify the cost of decoding an entire
     2–5 MB episode Protobuf just to extract 1 frame.
"""

from __future__ import annotations

import io
import json
import random
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from embodied_formats.common import format_bytes
from embodied_formats.formats.rlds_format import decode_tf_example, masked_crc32c


def _inspect_image_header(raw: bytes) -> dict[str, Any]:
    """Parse raw PNG IHDR or JPEG SOF header without full pixel decode."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 26:
        w, h, bit_depth, color_type = struct.unpack(">IIBB", raw[16:26])
        return {"codec": "PNG", "width": w, "height": h, "bit_depth": bit_depth, "color_type": color_type, "bytes": len(raw)}
    if raw[:2] == b"\xff\xd8":
        with Image.open(io.BytesIO(raw)) as im:
            return {"codec": "JPEG", "width": im.width, "height": im.height, "mode": im.mode, "bytes": len(raw)}
    return {"codec": "unknown", "bytes": len(raw)}


def audit_tfrecord_shard(root: Path) -> dict[str, Any]:
    """Verify length + payload Masked CRC32C on EVERY record in shard 0 and check dataset_info closure."""
    shards = sorted(p for p in root.rglob("*") if p.is_file() and ".tfrecord" in p.name)
    if not shards:
        raise FileNotFoundError(f"No .tfrecord* shard found under {root}")
    s0 = shards[0]

    records_meta: list[dict[str, Any]] = []
    with s0.open("rb") as f:
        rec_i = 0
        while True:
            rec_start = f.tell()
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            length = struct.unpack("<Q", hdr[:8])[0]
            len_crc = struct.unpack("<I", hdr[8:12])[0]
            len_crc_ok = len_crc == masked_crc32c(hdr[:8])
            payload_off = f.tell()
            payload = f.read(length)
            pay_crc_raw = f.read(4)
            if len(pay_crc_raw) < 4:
                break
            pay_crc = struct.unpack("<I", pay_crc_raw)[0]
            pay_crc_ok = pay_crc == masked_crc32c(payload)
            records_meta.append(
                {
                    "record_index": rec_i,
                    "frame_offset": rec_start,
                    "payload_offset": payload_off,
                    "payload_bytes": length,
                    "payload_human": format_bytes(length),
                    "len_crc_ok": len_crc_ok,
                    "payload_crc_ok": pay_crc_ok,
                }
            )
            rec_i += 1

    dinfo_path = root / "dataset_info.json"
    shard_len_expected = None
    if dinfo_path.exists():
        dinfo = json.loads(dinfo_path.read_text(encoding="utf-8"))
        # Infer shard index from filename like `...-train.tfrecord-00004-of-00064`
        parts = s0.name.split("-of-")[0].split("-")
        if parts and parts[-1].isdigit():
            shard_idx = int(parts[-1])
            for sp in dinfo.get("splits", []):
                if sp.get("name") in s0.name and shard_idx < len(sp.get("shardLengths", [])):
                    shard_len_expected = int(sp["shardLengths"][shard_idx])

    sizes = [r["payload_bytes"] for r in records_meta]
    return {
        "shard_file": s0.name,
        "shard_disk_size": format_bytes(s0.stat().st_size),
        "total_episode_records": len(records_meta),
        "expected_from_dataset_info_shardLengths": shard_len_expected,
        "shard_count_match": (shard_len_expected == len(records_meta)) if shard_len_expected is not None else None,
        "all_len_crc32c_ok": all(r["len_crc_ok"] for r in records_meta),
        "all_payload_crc32c_ok": all(r["payload_crc_ok"] for r in records_meta),
        "record_size_min": format_bytes(min(sizes)),
        "record_size_median": format_bytes(float(np.median(sizes))),
        "record_size_max": format_bytes(max(sizes)),
        "first_4_records_framing": records_meta[:4],
    }


def analyze_episode_protobuf_payload(root: Path) -> dict[str, Any]:
    """Dissect Episode 0's raw Protobuf wire map: image bytes vs embeddings vs state/action."""
    s0 = sorted(p for p in root.rglob("*") if p.is_file() and ".tfrecord" in p.name)[0]
    with s0.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        f.read(4)
        payload = f.read(length)

    ep0 = decode_tf_example(payload)
    n_steps = len(ep0["steps/is_first"][1]) if "steps/is_first" in ep0 else 1

    categories = {"images_bytes": 0, "language_embedding_bytes": 0, "state_action_bytes": 0, "other_bytes": 0}
    image_headers: dict[str, Any] = {}
    reshaped_tensors: list[dict[str, Any]] = []

    for k in sorted(ep0.keys()):
        kind, vals = ep0[k]
        if kind == "bytes":
            b_total = sum(len(x) for x in vals)
            if vals and (vals[0][:2] == b"\xff\xd8" or vals[0][:4] == b"\x89PNG"):
                categories["images_bytes"] += b_total
                image_headers[k] = _inspect_image_header(vals[0])
            else:
                categories["other_bytes"] += b_total
        elif kind == "float":
            b_total = len(vals) * 4
            dim = len(vals) // n_steps if len(vals) % n_steps == 0 else -1
            if "embedding" in k:
                categories["language_embedding_bytes"] += b_total
            elif "action" in k or "state" in k:
                categories["state_action_bytes"] += b_total
            else:
                categories["other_bytes"] += b_total
            reshaped_tensors.append(
                {
                    "key": k,
                    "flat_pb_len": len(vals),
                    "reshaped_logical": f"({n_steps}, {dim})" if dim > 0 else f"({len(vals)},)",
                    "bytes": format_bytes(b_total),
                }
            )
        else:
            categories["other_bytes"] += len(vals) * 8

    return {
        "episode_0_steps": n_steps,
        "episode_0_payload_bytes": format_bytes(length),
        "byte_share_pct": {
            k: round(100.0 * v / max(length, 1), 2) for k, v in categories.items()
        },
        "embedded_image_headers": image_headers,
        "flat_to_tensor_reshapes": reshaped_tensors,
    }


def parse_tfds_features_tree(root: Path) -> dict[str, Any]:
    """Flatten `features.json` into readable logical Tensor specs."""
    fpath = root / "features.json"
    if not fpath.exists():
        return {"present": False}
    raw = json.loads(fpath.read_text(encoding="utf-8"))
    flat: list[dict[str, str]] = []

    def _walk(node: dict[str, Any], prefix: str) -> None:
        cls = node.get("pythonClassName", "")
        if "featuresDict" in node:
            for k, v in node["featuresDict"].get("features", {}).items():
                _walk(v, f"{prefix}/{k}" if prefix else k)
        elif "sequence" in node:
            _walk(node["sequence"].get("feature", {}), prefix)
        elif "tensor" in node or "image" in node:
            t = node.get("tensor") or node.get("image")
            raw_dims = t.get("shape", {}).get("dimensions", [])
            dims = [d.get("size", "?") if isinstance(d, dict) else str(d) for d in raw_dims]
            flat.append(
                {
                    "path": prefix,
                    "class": cls.split(".")[-1],
                    "dtype": t.get("dtype", ""),
                    "shape": str(tuple(dims)),
                    "encoding": t.get("encodingFormat") or t.get("encoding", "none"),
                }
            )
        else:
            flat.append({"path": prefix, "class": cls.split(".")[-1], "dtype": "scalar/text", "shape": "()", "encoding": ""})

    _walk(raw, "")
    return {"present": True, "logical_fields": flat}


def profile_random_read_degradation(root: Path, num_queries: int = 30, seed: int = 42) -> dict[str, Any]:
    """Quantify why TFRecord RLDS suffers on true i.i.d. random single-step access.

    Compares:
      - No-Cache (true cross-episode random read): seek shard -> read entire 2–5 MB episode
        -> deserialize entire Protobuf -> decode 1 PNG frame.
      - Episode-Cached (within-episode sequential / shuffle-buffer read): decode 1 PNG frame
        from already-deserialized episode Protobuf.
    """
    s0 = sorted(p for p in root.rglob("*") if p.is_file() and ".tfrecord" in p.name)[0]
    offsets: list[tuple[int, int]] = []
    with s0.open("rb") as f:
        while True:
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            ln = struct.unpack("<Q", hdr[:8])[0]
            offsets.append((f.tell(), ln))
            f.seek(ln + 4, io.SEEK_CUR)

    rng = random.Random(seed)
    n = min(num_queries, len(offsets) * 5)

    # Pick primary image key from ep0
    with s0.open("rb") as f:
        f.seek(offsets[0][0])
        ep0 = decode_tf_example(f.read(offsets[0][1]))
    img_key = next(k for k, (kd, vs) in ep0.items() if kd == "bytes" and vs and (vs[0][:4] == b"\x89PNG" or vs[0][:2] == b"\xff\xd8"))

    # 1. No-Cache: full disk read + full Protobuf decode per single frame query
    nocache_ms: list[float] = []
    bytes_read_total = 0
    bytes_useful_total = 0
    with s0.open("rb") as f:
        t0 = time.perf_counter()
        for _ in range(n):
            ep_i = rng.randrange(len(offsets))
            off, ln = offsets[ep_i]
            q0 = time.perf_counter()
            f.seek(off)
            raw_ep = f.read(ln)
            dec = decode_tf_example(raw_ep)
            imgs = dec[img_key][1]
            step_i = rng.randrange(len(imgs))
            with Image.open(io.BytesIO(imgs[step_i])) as im:
                _ = np.asarray(im.convert("RGB"), dtype=np.uint8)
            nocache_ms.append((time.perf_counter() - q0) * 1000.0)
            bytes_read_total += ln
            bytes_useful_total += len(imgs[step_i])
        nocache_s = max(time.perf_counter() - t0, 1e-9)

    # 2. Cached-Episode: only PNG decode (simulating TFRecord sequential stream + in-RAM shuffle buffer)
    cached_ms: list[float] = []
    imgs0 = ep0[img_key][1]
    t1 = time.perf_counter()
    for _ in range(n):
        step_i = rng.randrange(len(imgs0))
        q0 = time.perf_counter()
        with Image.open(io.BytesIO(imgs0[step_i])) as im:
            _ = np.asarray(im.convert("RGB"), dtype=np.uint8)
        cached_ms.append((time.perf_counter() - q0) * 1000.0)
    cached_s = max(time.perf_counter() - t1, 1e-9)

    nc_arr = np.asarray(nocache_ms, dtype=np.float64)
    c_arr = np.asarray(cached_ms, dtype=np.float64)
    return {
        "queries": n,
        "image_key_tested": img_key,
        "read_amplification_ratio": round(bytes_read_total / max(bytes_useful_total, 1), 1),
        "nocache_true_random": {
            "throughput_sps": round(n / nocache_s, 2),
            "p50_ms": round(float(np.percentile(nc_arr, 50)), 3),
            "p99_ms": round(float(np.percentile(nc_arr, 99)), 3),
        },
        "cached_episode_shuffle_buffer": {
            "throughput_sps": round(n / cached_s, 2),
            "p50_ms": round(float(np.percentile(c_arr, 50)), 3),
            "p99_ms": round(float(np.percentile(c_arr, 99)), 3),
        },
        "speedup_with_shuffle_buffer": f"{( (n / cached_s) / max(n / nocache_s, 1e-9) ):.1f}x",
    }


def run_full_rlds_deep_study(root: Path, num_queries: int = 30) -> dict[str, Any]:
    return {
        "dataset_path": str(root),
        "tfrecord_framing_and_crc32c_audit": audit_tfrecord_shard(root),
        "episode_0_protobuf_wire_breakdown": analyze_episode_protobuf_payload(root),
        "tfds_features_json_schema": parse_tfds_features_tree(root),
        "random_access_degradation_profile": profile_random_read_degradation(root, num_queries=num_queries),
    }
