"""Download genuine, production open-source robotics datasets for all 5 formats.

All 5 sources are official, publicly hosted open-source robotics datasets:
  1. LeRobot v3.0 : HuggingFace `lerobot/pusht` (~7.3 MB, 206 episodes, AV1 + Parquet)
  2. Zarr         : Columbia Diffusion Policy `pusht.zip` (~29.6 MB, exact same PushT data in Zarr!)
  3. HDF5         : Stanford robomimic `v1.5/lift/ph/demo_v15.hdf5` (~35 MB, Franka Panda Lift)
  4. RLDS         : Open X-Embodiment GCS `berkeley_cable_routing/0.1.0` (dataset_info.json + features.json + 1 shard ~31.8 MB)
  5. MCAP         : RobotisAI `evButtonPush-260608-0-MCAP` episode 0 (`0_0.mcap` ~154 MB + metadata.yaml + episode_info.json + robot.urdf)
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def _download_url(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "embodied-formats-lab/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    tmp.replace(dst)
    return dst


def real_sample_paths(real_root: Path) -> dict[str, Path]:
    return {
        "lerobot_v3": real_root / "lerobot_v3_pusht",
        "zarr": real_root / "zarr_pusht" / "pusht_cchi_v7_replay.zarr",
        "hdf5": real_root / "hdf5_robomimic",
        "rlds": real_root / "rlds_berkeley_cable_routing",
        "mcap": real_root / "mcap_robotis_button_push",
    }


def fetch_real_lerobot_v3(dst: Path, force: bool = False) -> Path:
    if (dst / "meta" / "info.json").exists() and not force:
        return dst
    # Reuse already downloaded data/lerobot_v3/pusht if present to avoid re-downloading
    existing = Path("data/lerobot_v3/pusht")
    if existing.exists() and (existing / "meta" / "info.json").exists() and not force:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(existing, dst, dirs_exist_ok=True)
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="lerobot/pusht",
        repo_type="dataset",
        local_dir=str(dst),
        allow_patterns=["meta/*", "meta/**/*", "data/**/*", "videos/**/*", "README.md"],
    )
    return dst


def fetch_real_zarr(dst_root: Path, force: bool = False) -> Path:
    zarr_dir = dst_root / "pusht_cchi_v7_replay.zarr"
    if (zarr_dir / ".zgroup").exists() and not force:
        return zarr_dir
    dst_root.mkdir(parents=True, exist_ok=True)
    zip_path = dst_root / "pusht.zip"
    if not zip_path.exists() or force:
        _download_url("https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_root)
    # Handle nested pusht/pusht_cchi_v7_replay.zarr if zip wraps in a folder
    nested = dst_root / "pusht" / "pusht_cchi_v7_replay.zarr"
    if nested.exists() and not zarr_dir.exists():
        shutil.move(str(nested), str(zarr_dir))
    zip_path.unlink(missing_ok=True)
    return zarr_dir


def fetch_real_hdf5(dst: Path, force: bool = False) -> Path:
    target = dst / "demo_v15.hdf5"
    if target.exists() and not force:
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id="robomimic/robomimic_datasets",
        repo_type="dataset",
        filename="v1.5/lift/ph/demo_v15.hdf5",
    )
    shutil.copy2(downloaded, target)
    return dst


def fetch_real_rlds(dst: Path, force: bool = False) -> Path:
    base = "https://storage.googleapis.com/gresearch/robotics/berkeley_cable_routing/0.1.0"
    files = [
        "dataset_info.json",
        "features.json",
        "berkeley_cable_routing-train.tfrecord-00004-of-00064",
    ]
    dst.mkdir(parents=True, exist_ok=True)
    for fname in files:
        out = dst / fname
        if not out.exists() or force:
            _download_url(f"{base}/{fname}", out)
    return dst


def fetch_real_mcap(dst: Path, force: bool = False) -> Path:
    if (dst / "0_0.mcap").exists() and not force:
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for fname in ("0/0_0.mcap", "0/metadata.yaml", "0/episode_info.json", "0/robot.urdf"):
        downloaded = hf_hub_download(
            repo_id="RobotisAI/evButtonPush-260608-0-MCAP",
            repo_type="dataset",
            filename=fname,
        )
        shutil.copy2(downloaded, dst / Path(fname).name)
    return dst
