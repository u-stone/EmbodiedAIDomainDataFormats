"""Low-level HDF5 format inspector and read-only reader (robomimic / ALOHA / ACT conventions).

Systems-level anatomy of HDF5:
- Single binary file organized as a POSIX-like hierarchy of Groups and multi-dimensional Datasets.
- Internally uses B-Trees and heaps to locate contiguous or chunked dataset blocks.
- Key characteristics:
  1. Extremely fast single-process slicing (especially contiguous or well-chunked datasets).
  2. No universal robotics schema: robomimic uses `data/demo_i/obs/...`, ALOHA uses
     `observations/images/...` per episode file.
  3. Global HDF5 library lock (not thread-safe without SWMR/locks) + raw uncompressed RGB
     arrays blow up disk usage (10x–40x larger than MP4-encoded LeRobot).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import h5py
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


def _decode_hdf5_image(arr: Any) -> np.ndarray:
    """Handle both raw uint8 HxWx3 arrays and JPEG-compressed 1D byte arrays."""
    if isinstance(arr, (bytes, bytearray)):
        with Image.open(io.BytesIO(arr)) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    np_arr = np.asarray(arr)
    if np_arr.ndim == 3:
        return np_arr.astype(np.uint8, copy=False)
    if np_arr.ndim == 1 and np_arr.dtype == np.uint8:
        with Image.open(io.BytesIO(np_arr.tobytes())) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    raise ValueError(f"Unsupported HDF5 image array shape={np_arr.shape}, dtype={np_arr.dtype}")


class HDF5FormatReader(BaseFormatReader):
    """Inspect and read HDF5 datasets (supports both robomimic single-file & ALOHA multi-file layouts)."""

    format_name = "hdf5"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        if self.path.is_file():
            self._files = [self.path]
        else:
            self._files = sorted(list(self.path.glob("*.hdf5")) + list(self.path.glob("*.h5")))
        if not self._files:
            raise FileNotFoundError(f"No .hdf5/.h5 files found under {self.path}")

        self._handles: dict[Path, h5py.File] = {}
        self._ep_map: list[tuple[Path, str]] = []  # (file_path, group_path)
        self._episodes: list[EpisodeMeta] | None = None
        self._discover_episodes()

    def _get_handle(self, fp: Path) -> h5py.File:
        if fp not in self._handles:
            self._handles[fp] = h5py.File(fp, "r")
        return self._handles[fp]

    def _discover_episodes(self) -> None:
        self._ep_map.clear()
        for fp in self._files:
            f = self._get_handle(fp)
            if "data" in f and isinstance(f["data"], h5py.Group):
                demos = sorted(
                    f["data"].keys(),
                    key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k,
                )
                for d in demos:
                    self._ep_map.append((fp, f"data/{d}"))
            else:
                # ALOHA style: one episode per .hdf5 file at root
                self._ep_map.append((fp, "/"))

    def _resolve_ep_datasets(self, grp: h5py.Group) -> tuple[dict[str, h5py.Dataset], h5py.Dataset | None, h5py.Dataset | None]:
        """Locate camera datasets, state dataset, and action dataset inside an episode group."""
        cameras: dict[str, h5py.Dataset] = {}
        state_ds: h5py.Dataset | None = None
        action_ds: h5py.Dataset | None = None

        # Action
        for cand in ("actions", "action"):
            if cand in grp and isinstance(grp[cand], h5py.Dataset):
                action_ds = grp[cand]
                break

        # Observations (robomimic `obs/` or ALOHA `observations/`)
        obs_grp = grp.get("obs") or grp.get("observations")
        if isinstance(obs_grp, h5py.Group):
            if "images" in obs_grp and isinstance(obs_grp["images"], h5py.Group):
                for cam_name, ds in obs_grp["images"].items():
                    if isinstance(ds, h5py.Dataset):
                        cameras[cam_name] = ds
            for k, v in obs_grp.items():
                if not isinstance(v, h5py.Dataset):
                    continue
                kl = k.lower()
                if "image" in kl or "rgb" in kl or (v.ndim == 4 and v.shape[-1] == 3):
                    cameras[k] = v
                elif state_ds is None and any(tok in kl for tok in ("state", "qpos", "eef", "joint", "pos")):
                    state_ds = v
        return cameras, state_ds, action_ds

    def list_episodes(self) -> list[EpisodeMeta]:
        if self._episodes is not None:
            return self._episodes
        eps: list[EpisodeMeta] = []
        for idx, (fp, gpath) in enumerate(self._ep_map):
            f = self._get_handle(fp)
            grp = f if gpath == "/" else f[gpath]
            cams, state_ds, act_ds = self._resolve_ep_datasets(grp)
            ref_ds = act_ds or state_ds or next(iter(cams.values()), None)
            num_steps = int(ref_ds.shape[0]) if ref_ds is not None else 0
            fps = float(grp.attrs.get("fps", f.attrs.get("fps", 20.0)))
            lang = str(grp.attrs.get("language_instruction", f.attrs.get("language_instruction", "")))
            eps.append(
                EpisodeMeta(
                    episode_id=idx,
                    num_steps=num_steps,
                    fps=fps,
                    language_instruction=lang,
                    extra={"file": fp.name, "group": gpath},
                )
            )
        self._episodes = eps
        return eps

    def inspect(self) -> InspectReport:
        disk_bytes, file_count = dir_disk_usage(self.path)
        episodes = self.list_episodes()

        schema_fields: list[FieldSchema] = []
        total_logical_bytes = 0
        total_storage_bytes = 0
        chunked_count = 0
        contiguous_count = 0
        compressions: set[str] = set()

        # Walk the first file (and first episode group if multi-demo) for schema details
        first_fp, first_gpath = self._ep_map[0]
        f0 = self._get_handle(first_fp)
        root_grp = f0 if first_gpath == "/" else f0[first_gpath]

        def _visit_schema(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                logical = int(obj.size) * int(obj.dtype.itemsize)
                storage = int(obj.id.get_storage_size())
                ratio = (logical / storage) if storage > 0 else 1.0
                comp = obj.compression or "none"
                if obj.compression_opts is not None:
                    comp = f"{comp}(lvl={obj.compression_opts})"
                schema_fields.append(
                    FieldSchema(
                        name=name,
                        dtype=str(obj.dtype),
                        shape=str(obj.shape),
                        encoding=comp,
                        details=(
                            f"chunks={obj.chunks or 'contiguous'}, "
                            f"logical={format_bytes(logical)}, stored={format_bytes(storage)} ({ratio:.2f}x)"
                        ),
                    )
                )

        root_grp.visititems(_visit_schema)

        # Aggregate storage vs logical bytes across ALL files
        for fp in self._files:
            f = self._get_handle(fp)

            def _accum(_: str, obj: Any) -> None:
                nonlocal total_logical_bytes, total_storage_bytes, chunked_count, contiguous_count
                if isinstance(obj, h5py.Dataset):
                    total_logical_bytes += int(obj.size) * int(obj.dtype.itemsize)
                    total_storage_bytes += int(obj.id.get_storage_size())
                    if obj.chunks is not None:
                        chunked_count += 1
                    else:
                        contiguous_count += 1
                    compressions.add(obj.compression or "none")

            f.visititems(_accum)

        overall_ratio = (total_logical_bytes / total_storage_bytes) if total_storage_bytes > 0 else 1.0
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
                "layout": "robomimic (data/demo_i)" if first_gpath.startswith("data/") else "ALOHA (per-file root)",
                "total_logical_bytes": format_bytes(total_logical_bytes),
                "total_stored_dataset_bytes": format_bytes(total_storage_bytes),
                "overall_compression_ratio": f"{overall_ratio:.2f}x",
                "chunked_datasets": chunked_count,
                "contiguous_datasets": contiguous_count,
                "compressions": sorted(compressions),
                "hdf5_lib_version": h5py.version.hdf5_version,
            },
            notes=[
                "HDF5 内部以 B-Tree + Heap 管理树状 Group/Dataset；若采用 contiguous 或按单帧 chunk=(1,H,W,3) 布局，单机随机切片延迟极低。",
                "缺乏行业统一的键名 Schema（robomimic 用 data/demo_i/obs，ALOHA 用 observations/images），跨项目复用需要写 Adapter。",
                "不支持视频帧间压缩（只能逐帧存原始像素或 gzip/JPEG 字节流），高清多视角下体积比 LeRobot MP4 大一个数量级；且单文件多进程写/读受 HDF5 全局锁制约。",
            ],
        )

    def read_step(self, episode_id: int, step_index: int) -> StepSample:
        fp, gpath = self._ep_map[episode_id]
        f = self._get_handle(fp)
        grp = f if gpath == "/" else f[gpath]
        cams, state_ds, act_ds = self._resolve_ep_datasets(grp)
        ep = self.list_episodes()[episode_id]

        images = {cam: _decode_hdf5_image(ds[step_index]) for cam, ds in cams.items()}
        state = np.asarray(state_ds[step_index], dtype=np.float32) if state_ds is not None else None
        action = np.asarray(act_ds[step_index], dtype=np.float32) if act_ds is not None else None

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
        """Optimized HDF5 range-slice window read (avoids redundant per-step HDF5 B-Tree lookups)."""
        fp, gpath = self._ep_map[episode_id]
        f = self._get_handle(fp)
        grp = f if gpath == "/" else f[gpath]
        cams, state_ds, act_ds = self._resolve_ep_datasets(grp)
        ep = self.list_episodes()[episode_id]
        n = ep.num_steps

        obs_start = max(0, step_index - obs_horizon + 1)
        obs_end = min(n, step_index + 1)
        pad_left = obs_horizon - (obs_end - obs_start)

        act_start = step_index
        act_end = min(n, step_index + action_horizon)
        pad_right = action_horizon - (act_end - act_start)

        obs_idx = [0] * pad_left + list(range(obs_start, obs_end))
        timestamps = np.asarray(obs_idx, dtype=np.float64) / max(ep.fps, 1e-6)

        images: dict[str, np.ndarray] = {}
        for cam, ds in cams.items():
            raw_slice = ds[obs_start:obs_end]
            decoded = np.stack([_decode_hdf5_image(raw_slice[i]) for i in range(len(raw_slice))], axis=0)
            if pad_left > 0:
                pad = np.repeat(decoded[:1], pad_left, axis=0)
                decoded = np.concatenate([pad, decoded], axis=0)
            images[cam] = decoded

        states = None
        if state_ds is not None:
            st = np.asarray(state_ds[obs_start:obs_end], dtype=np.float32)
            if pad_left > 0:
                st = np.concatenate([np.repeat(st[:1], pad_left, axis=0), st], axis=0)
            states = st

        action_chunk = None
        if act_ds is not None:
            ac = np.asarray(act_ds[act_start:act_end], dtype=np.float32)
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

    def close(self) -> None:
        for h in self._handles.values():
            h.close()
        self._handles.clear()
