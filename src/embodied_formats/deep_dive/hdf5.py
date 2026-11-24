"""Ultra-deep systems anatomy & verification lab for HDF5 / robomimic.

Designed for genuine robomimic / ALOHA HDF5 files (e.g. `v1.5/lift/ph/demo_v15.hdf5`):
  1. Raw HDF5 Superblock binary header parse (`\x89HDF\r\n\x1a\n`, version, offset/length size,
     base address, end-of-file address).
  2. Full B-Tree storage accounting: sum of logical bytes vs physical dataset storage bytes
     vs actual file size on disk -> exposes exact HDF5 B-Tree/heap metadata overhead!
  3. Embedded non-binary metadata extraction: `/data.env_args` JSON + `/data/demo_0.model_file`
     MuJoCo XML topology summary (bodies/joints/actuators/sensors).
  4. Contiguous zero-decompression slicing benchmark (sequential vs cross-demo random single-step
     vs action-chunk window) measuring microsecond-level P50/P99 latency.
"""

from __future__ import annotations

import json
import random
import struct
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from embodied_formats.common import format_bytes


def _find_hdf5_file(root: Path) -> Path:
    if root.is_file():
        return root
    cands = sorted(list(root.glob("*.hdf5")) + list(root.glob("*.h5")))
    if not cands:
        raise FileNotFoundError(f"No .hdf5/.h5 file found under {root}")
    return cands[0]


def parse_hdf5_superblock(h5_path: Path) -> dict[str, Any]:
    """Parse raw HDF5 binary Superblock (v0/v1 or v2/v3) from file offset 0."""
    fsize = h5_path.stat().st_size
    with h5_path.open("rb") as f:
        raw = f.read(64)

    sig_ok = raw[:8] == b"\x89HDF\r\n\x1a\n"
    ver = raw[8]
    info: dict[str, Any] = {
        "file": h5_path.name,
        "file_size_bytes": fsize,
        "file_size_human": format_bytes(fsize),
        "magic_signature_ok": sig_ok,
        "superblock_version": ver,
    }
    if ver in (0, 1):
        offset_sz = raw[13]
        length_sz = raw[14]
        group_leaf_k, group_internal_k = struct.unpack("<HH", raw[16:20])
        base_addr, free_addr, eof_addr, drv_addr = struct.unpack("<QQQQ", raw[24:56])
        info.update(
            {
                "size_of_offsets": offset_sz,
                "size_of_lengths": length_sz,
                "group_leaf_node_k": group_leaf_k,
                "group_internal_node_k": group_internal_k,
                "base_address": base_addr,
                "eof_address": eof_addr,
                "eof_matches_file_size": (eof_addr == fsize),
            }
        )
    else:
        offset_sz = raw[9]
        length_sz = raw[10]
        info.update({"size_of_offsets": offset_sz, "size_of_lengths": length_sz})
    return info


def audit_hdf5_storage_and_metadata(h5_path: Path) -> dict[str, Any]:
    """Walk every Group/Dataset to compute B-Tree metadata overhead and parse embedded XML/JSON."""
    fsize = h5_path.stat().st_size
    total_datasets = 0
    contiguous_cnt = 0
    chunked_cnt = 0
    total_logical = 0
    total_stored = 0
    demo_lengths: list[int] = []
    env_args_summary: dict[str, Any] = {}
    mujoco_xml_summary: dict[str, Any] = {}
    mask_splits: dict[str, int] = {}

    with h5py.File(h5_path, "r") as hf:
        def _walk(_: str, obj: Any) -> None:
            nonlocal total_datasets, contiguous_cnt, chunked_cnt, total_logical, total_stored
            if isinstance(obj, h5py.Dataset):
                total_datasets += 1
                total_logical += int(obj.size) * int(obj.dtype.itemsize)
                total_stored += int(obj.id.get_storage_size())
                if obj.chunks is None:
                    contiguous_cnt += 1
                else:
                    chunked_cnt += 1

        hf.visititems(_walk)

        if "mask" in hf and isinstance(hf["mask"], h5py.Group):
            for mk, mv in hf["mask"].items():
                if isinstance(mv, h5py.Dataset):
                    mask_splits[mk] = int(mv.shape[0])

        if "data" in hf and isinstance(hf["data"], h5py.Group):
            if "env_args" in hf["data"].attrs:
                ea = json.loads(hf["data"].attrs["env_args"])
                env_args_summary = {
                    "env_name": ea.get("env_name"),
                    "type": ea.get("type"),
                    "control_freq": ea.get("env_kwargs", {}).get("control_freq"),
                    "use_camera_obs": ea.get("env_kwargs", {}).get("use_camera_obs"),
                    "use_object_obs": ea.get("env_kwargs", {}).get("use_object_obs"),
                    "robots": ea.get("env_kwargs", {}).get("robots"),
                }

            demos = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
            for d in demos:
                grp = hf["data"][d]
                if "actions" in grp:
                    demo_lengths.append(int(grp["actions"].shape[0]))
            if demos and "model_file" in hf["data"][demos[0]].attrs:
                xml_str = str(hf["data"][demos[0]].attrs["model_file"])
                root_el = ET.fromstring(xml_str)
                mujoco_xml_summary = {
                    "embedded_in": f"/data/{demos[0]}.attrs['model_file']",
                    "xml_bytes": format_bytes(len(xml_str.encode("utf-8"))),
                    "model_name": root_el.attrib.get("model", ""),
                    "body_count": len(root_el.findall(".//body")),
                    "joint_count": len(root_el.findall(".//joint")),
                    "geom_count": len(root_el.findall(".//geom")),
                    "actuator_count": len(root_el.findall(".//actuator/*")),
                }

    btree_overhead = max(0, fsize - total_stored)
    return {
        "num_demos": len(demo_lengths),
        "total_steps": sum(demo_lengths),
        "demo_len_min_median_max": (
            [min(demo_lengths), int(np.median(demo_lengths)), max(demo_lengths)] if demo_lengths else []
        ),
        "total_datasets_in_file": total_datasets,
        "contiguous_datasets": contiguous_cnt,
        "chunked_datasets": chunked_cnt,
        "total_logical_payload": format_bytes(total_logical),
        "total_stored_payload": format_bytes(total_stored),
        "hdf5_btree_and_attrs_overhead": f"{format_bytes(btree_overhead)} ({100.0 * btree_overhead / max(fsize, 1):.2f}%)",
        "embedded_train_val_masks": mask_splits,
        "embedded_env_args": env_args_summary,
        "embedded_mujoco_xml": mujoco_xml_summary,
    }


def profile_hdf5_slicing(h5_path: Path, num_queries: int = 200, action_horizon: int = 16, seed: int = 42) -> dict[str, Any]:
    """Measure microsecond-level latency of contiguous HDF5 slicing across demos."""
    rng = random.Random(seed)
    with h5py.File(h5_path, "r") as hf:
        data_grp = hf["data"]
        demos = sorted(data_grp.keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
        coords = [(d, t) for d in demos for t in range(data_grp[d]["actions"].shape[0])]
        n = min(num_queries, len(coords))

        seq_coords = coords[:n]
        rand_coords = [rng.choice(coords) for _ in range(n)]

        # 1. Sequential step read (state + action)
        seq_us: list[float] = []
        t0 = time.perf_counter()
        for d, t in seq_coords:
            q0 = time.perf_counter()
            _ = data_grp[d]["states"][t]
            _ = data_grp[d]["actions"][t]
            seq_us.append((time.perf_counter() - q0) * 1e6)
        seq_s = max(time.perf_counter() - t0, 1e-9)

        # 2. Cross-demo random single-step read (B-Tree group lookup + contiguous slice)
        rand_us: list[float] = []
        t1 = time.perf_counter()
        for d, t in rand_coords:
            q0 = time.perf_counter()
            _ = data_grp[d]["states"][t]
            _ = data_grp[d]["actions"][t]
            rand_us.append((time.perf_counter() - q0) * 1e6)
        rand_s = max(time.perf_counter() - t1, 1e-9)

        # 3. Cross-demo random action-chunk window slice
        win_us: list[float] = []
        t2 = time.perf_counter()
        for d, t in rand_coords:
            q0 = time.perf_counter()
            ds_a = data_grp[d]["actions"]
            _ = data_grp[d]["states"][max(0, t - 1) : t + 1]
            _ = ds_a[t : min(ds_a.shape[0], t + action_horizon)]
            win_us.append((time.perf_counter() - q0) * 1e6)
        win_s = max(time.perf_counter() - t2, 1e-9)

        # 4. Verify physical byte offset of first demo's contiguous `actions` dataset
        act0_offset = data_grp[demos[0]]["actions"].id.get_offset()

    return {
        "queries": n,
        "demo_0_actions_physical_file_offset": act0_offset,
        "sequential": {
            "throughput_sps": round(n / seq_s, 1),
            "p50_us": round(float(np.percentile(seq_us, 50)), 1),
            "p99_us": round(float(np.percentile(seq_us, 99)), 1),
        },
        "random_cross_demo_step": {
            "throughput_sps": round(n / rand_s, 1),
            "p50_us": round(float(np.percentile(rand_us, 50)), 1),
            "p99_us": round(float(np.percentile(rand_us, 99)), 1),
        },
        "random_cross_demo_window_act16": {
            "throughput_sps": round(n / win_s, 1),
            "p50_us": round(float(np.percentile(win_us, 50)), 1),
            "p99_us": round(float(np.percentile(win_us, 99)), 1),
        },
    }


def run_full_hdf5_deep_study(root: Path, num_queries: int = 200) -> dict[str, Any]:
    h5_path = _find_hdf5_file(root)
    return {
        "dataset_path": str(h5_path),
        "superblock_binary": parse_hdf5_superblock(h5_path),
        "storage_btree_and_embedded_metadata": audit_hdf5_storage_and_metadata(h5_path),
        "contiguous_slicing_profile": profile_hdf5_slicing(h5_path, num_queries=num_queries),
    }
