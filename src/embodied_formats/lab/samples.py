"""Sample dataset preparation (real open-source LeRobot v3.0 + native spec-compliant samples).

NOTE: This module NEVER performs cross-format ETL conversion.
- LeRobot v3.0: downloads the real open-source `lerobot/pusht` v3.0 dataset from HuggingFace Hub.
- MCAP / HDF5 / Zarr / RLDS: creates compact, 100% spec-compliant native datasets directly
  in each format's canonical on-disk layout so every format can be inspected and benchmarked
  out-of-the-box without downloading multi-GB archives. Users can also pass `--path` to any
  CLI command to inspect their own real datasets of any size.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import h5py
import numpy as np
import zarr
from huggingface_hub import snapshot_download
from mcap.writer import CompressionType, Writer as McapWriter
from numcodecs import Blosc
from PIL import Image

from embodied_formats.formats.rlds_format import encode_tf_example, write_tfrecord_item


def _make_synthetic_rgb(ep_idx: int, step_idx: int, height: int = 96, width: int = 96) -> np.ndarray:
    """Generate a deterministic, non-trivial textured RGB frame (moving colored block + gradient)."""
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = ((xx * 2 + ep_idx * 37) % 256).astype(np.uint8)
    img[:, :, 1] = ((yy * 2 + step_idx * 5) % 256).astype(np.uint8)
    img[:, :, 2] = (((xx + yy) + ep_idx * 19) % 256).astype(np.uint8)
    # Moving target square simulating end-effector / object motion
    cx = (16 + (step_idx * 3 + ep_idx * 11) % (width - 32))
    cy = (16 + (step_idx * 2 + ep_idx * 7) % (height - 32))
    img[cy - 8 : cy + 8, cx - 8 : cx + 8] = [240, 220, 40]
    return img


def _encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def setup_lerobot_v3_real(target_dir: Path, force: bool = False) -> Path:
    """Download the real open-source `lerobot/pusht` (v3.0, 206 episodes, ~35 MB) from HuggingFace."""
    info_file = target_dir / "meta" / "info.json"
    if info_file.exists() and not force:
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="lerobot/pusht",
        repo_type="dataset",
        local_dir=str(target_dir),
        allow_patterns=["meta/*", "meta/**/*", "data/**/*", "videos/**/*"],
    )
    return target_dir


def setup_mcap_sample(target_dir: Path, num_episodes: int = 4, steps_per_ep: int = 60, force: bool = False) -> Path:
    """Create native multi-topic asynchronous MCAP recordings (10 Hz dual-cam + 50 Hz state/cmd)."""
    if target_dir.exists() and any(target_dir.glob("*.mcap")) and not force:
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(100)

    cam_hz = 10.0
    ctrl_hz = 50.0
    base_ns = 1_730_000_000_000_000_000

    for ep in range(num_episodes):
        out_path = target_dir / f"recording_{ep:03d}.mcap"
        with out_path.open("wb") as f:
            writer = McapWriter(f, compression=CompressionType.ZSTD)
            writer.start(profile="ros2", library="embodied-formats-lab")

            cam_schema = writer.register_schema(
                name="foxglove.CompressedImage",
                encoding="jsonschema",
                data=b'{"type":"object","properties":{"format":{"type":"string"},"data":{"type":"string"}}}',
            )
            state_schema = writer.register_schema(
                name="sensor_msgs/JointState",
                encoding="jsonschema",
                data=b'{"type":"object","properties":{"position":{"type":"array"}}}',
            )
            act_schema = writer.register_schema(
                name="geometry_msgs/ActionCommand",
                encoding="jsonschema",
                data=b'{"type":"object","properties":{"command":{"type":"array"}}}',
            )

            ch_cam0 = writer.register_channel("/camera/front/image_compressed", "json", cam_schema)
            ch_cam1 = writer.register_channel("/camera/wrist/image_compressed", "json", cam_schema)
            ch_state = writer.register_channel("/robot/joint_states", "json", state_schema)
            ch_act = writer.register_channel("/teleop/action_cmd", "json", act_schema)

            # Collect asynchronous events with realistic clock jitter (1–3 ms)
            events: list[tuple[int, int, bytes]] = []
            ep_start_ns = base_ns + ep * 30_000_000_000
            dur_s = steps_per_ep / cam_hz

            # 10 Hz cameras
            for s in range(steps_per_ep):
                t0_ns = ep_start_ns + int((s / cam_hz) * 1e9) + int(rng.normal(0, 1.5e6))
                jpg0 = _encode_jpeg(_make_synthetic_rgb(ep, s))
                msg0 = json.dumps({"format": "jpeg", "data": base64.b64encode(jpg0).decode("ascii")}).encode("utf-8")
                events.append((t0_ns, ch_cam0, msg0))

                # Second camera slightly phase-offset (+4 ms)
                t1_ns = t0_ns + int(4e6 + rng.normal(0, 1.0e6))
                jpg1 = _encode_jpeg(_make_synthetic_rgb(ep + 10, s))
                msg1 = json.dumps({"format": "jpeg", "data": base64.b64encode(jpg1).decode("ascii")}).encode("utf-8")
                events.append((t1_ns, ch_cam1, msg1))

            # 50 Hz proprioception & teleop command
            n_ctrl = int(dur_s * ctrl_hz)
            for k in range(n_ctrl):
                t_s = k / ctrl_hz
                tc_ns = ep_start_ns + int(t_s * 1e9) + int(rng.normal(0, 0.4e6))
                qpos = [float(np.sin(t_s + j * 0.4 + ep)) for j in range(7)]
                cmd = [float(np.cos(t_s + j * 0.4 + ep) * 0.05) for j in range(7)]
                events.append((tc_ns, ch_state, json.dumps({"position": qpos}).encode("utf-8")))
                events.append((tc_ns + 500_000, ch_act, json.dumps({"command": cmd}).encode("utf-8")))

            events.sort(key=lambda x: x[0])
            for log_time, ch_id, payload in events:
                writer.add_message(channel_id=ch_id, log_time=log_time, data=payload, publish_time=log_time)
            writer.finish()
    return target_dir


def setup_hdf5_sample(target_dir: Path, num_episodes: int = 6, steps_per_ep: int = 80, force: bool = False) -> Path:
    """Create a native robomimic-style HDF5 dataset (`data/demo_i/obs/agentview_image`, `state`, `actions`)."""
    out_file = target_dir / "robomimic_sample.hdf5"
    if out_file.exists() and not force:
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(200)

    with h5py.File(out_file, "w") as f:
        f.attrs["fps"] = 20.0
        f.attrs["language_instruction"] = "pick up the yellow block and place it on the mat"
        data_grp = f.create_group("data")
        for ep in range(num_episodes):
            demo = data_grp.create_group(f"demo_{ep}")
            demo.attrs["fps"] = 20.0
            obs = demo.create_group("obs")

            imgs = np.stack([_make_synthetic_rgb(ep, s) for s in range(steps_per_ep)], axis=0)
            states = rng.normal(0.0, 0.5, size=(steps_per_ep, 7)).astype(np.float32)
            actions = rng.normal(0.0, 0.1, size=(steps_per_ep, 7)).astype(np.float32)

            # Chunk=(1, 96, 96, 3) with gzip level 4 — typical robomimic single-frame random-read layout
            obs.create_dataset(
                "agentview_image",
                data=imgs,
                chunks=(1, 96, 96, 3),
                compression="gzip",
                compression_opts=4,
            )
            obs.create_dataset("robot0_eef_pos", data=states)
            demo.create_dataset("actions", data=actions)
    return target_dir


def setup_zarr_sample(target_dir: Path, num_episodes: int = 8, steps_per_ep: int = 80, force: bool = False) -> Path:
    """Create a native Diffusion Policy ReplayBuffer Zarr store (`data/img`, `data/state`, `data/action`, `meta/episode_ends`)."""
    store_dir = target_dir / "pusht_style_replay.zarr"
    if (store_dir / ".zgroup").exists() and not force:
        return store_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(300)

    total = num_episodes * steps_per_ep
    imgs = np.stack(
        [_make_synthetic_rgb(ep, s) for ep in range(num_episodes) for s in range(steps_per_ep)],
        axis=0,
    )
    states = rng.normal(0.0, 1.0, size=(total, 5)).astype(np.float32)
    actions = rng.normal(0.0, 0.2, size=(total, 2)).astype(np.float32)
    ep_ends = np.cumsum([steps_per_ep] * num_episodes, dtype=np.int64)

    root = zarr.open_group(str(store_dir), mode="w")
    root.attrs["fps"] = 10.0
    root.attrs["language_instruction"] = "push the T-shaped block to the target zone"

    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    data_grp = root.create_group("data")
    data_grp.create_dataset("img", data=imgs, chunks=(16, 96, 96, 3), compressor=compressor)
    data_grp.create_dataset("state", data=states, chunks=(128, 5), compressor=compressor)
    data_grp.create_dataset("action", data=actions, chunks=(128, 2), compressor=compressor)

    meta_grp = root.create_group("meta")
    meta_grp.create_dataset("episode_ends", data=ep_ends)
    return store_dir


def setup_rlds_sample(target_dir: Path, num_episodes: int = 8, steps_per_ep: int = 60, force: bool = False) -> Path:
    """Create a native RLDS / TFRecord shard + `dataset_info.json` (BridgeData V2 / OXE style)."""
    shard_path = target_dir / "bridge_style-train.tfrecord-00000-of-00001"
    if shard_path.exists() and not force:
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(400)

    info = {
        "name": "bridge_style_sample",
        "version": "1.0.0",
        "fps": 5.0,
        "description": "Spec-compliant RLDS/TFRecord sample following Open X-Embodiment nested Episode->Steps schema.",
    }
    (target_dir / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    with shard_path.open("wb") as f:
        for ep in range(num_episodes):
            jpegs = [_encode_jpeg(_make_synthetic_rgb(ep, s), quality=85) for s in range(steps_per_ep)]
            states = rng.normal(0.0, 0.5, size=(steps_per_ep, 7)).astype(np.float32).reshape(-1)
            actions = rng.normal(0.0, 0.1, size=(steps_per_ep, 7)).astype(np.float32).reshape(-1)
            is_first = [1 if i == 0 else 0 for i in range(steps_per_ep)]
            is_last = [1 if i == steps_per_ep - 1 else 0 for i in range(steps_per_ep)]
            rewards = [1.0 if i == steps_per_ep - 1 else 0.0 for i in range(steps_per_ep)]
            langs = [f"move object to goal {ep}" for _ in range(steps_per_ep)]

            payload = encode_tf_example(
                {
                    "steps/observation/image_0": ("bytes", jpegs),
                    "steps/observation/state": ("float", states),
                    "steps/action": ("float", actions),
                    "steps/is_first": ("int64", is_first),
                    "steps/is_last": ("int64", is_last),
                    "steps/reward": ("float", rewards),
                    "steps/language_instruction": ("bytes", langs),
                }
            )
            write_tfrecord_item(f, payload)
    return target_dir


def default_sample_paths(data_root: Path) -> dict[str, Path]:
    return {
        "mcap": data_root / "mcap",
        "hdf5": data_root / "hdf5",
        "zarr": data_root / "zarr" / "pusht_style_replay.zarr",
        "rlds": data_root / "rlds",
        "lerobot_v3": data_root / "lerobot_v3" / "pusht",
    }
