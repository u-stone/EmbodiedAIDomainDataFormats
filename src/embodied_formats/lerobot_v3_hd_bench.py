"""Dual-Camera HD LeRobot v3.0 Systems Lab: CPU vs NVDEC (CUVID) + Multi-Worker DataLoader Benchmark.

Designed for real multi-camera 480x640+ LeRobot v3.0 datasets (e.g. `lerobot/svla_so100_pickplace`).
Measures:
  1. Time breakdown of a single dual-cam VLA training sample:
     Parquet lookup vs Camera-1 seek+decode vs Camera-2 seek+decode vs YUV->RGB conversion.
  2. CPU software decode vs NVIDIA NVDEC hardware decode (`h264_cuvid` / `av1_cuvid` / `hevc_cuvid`)
     across both sequential streaming and random-seek workloads.
  3. Multi-process worker scaling (1 / 2 / 4 / 8 workers, simulating PyTorch DataLoader num_workers)
     reporting samples/sec, P50/P99 latency, and effective RGB tensor bandwidth (MB/s).
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

from embodied_formats.formats.lerobot_v3_format import LeRobotV3FormatReader
from embodied_formats.lerobot_v3_deep import parse_mp4_sample_table

_SW_TO_CUVID = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "av1": "av1_cuvid",
    "libdav1d": "av1_cuvid",
    "vp9": "vp9_cuvid",
    "vp8": "vp8_cuvid",
}


def _pick_cuvid_decoder(mp4_path: Path) -> tuple[str, str | None]:
    """Detect stream codec and matching NVIDIA CUVID hardware decoder name if available."""
    with av.open(str(mp4_path)) as c:
        sw_name = c.streams.video[0].codec_context.name
    cuvid = _SW_TO_CUVID.get(sw_name)
    if cuvid and cuvid in av.codecs_available:
        return sw_name, cuvid
    return sw_name, None


def benchmark_single_video_decoder(
    mp4_path: Path,
    fps: float,
    codec_override: str | None = None,
    num_queries: int = 80,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark sequential vs random-seek decode on one MP4 using CPU or CUVID decoder."""
    stbl = parse_mp4_sample_table(mp4_path)
    total_frames = stbl.total_samples
    n = min(num_queries, total_frames)
    rng = random.Random(seed)
    rand_frames = [rng.randint(0, total_frames - 1) for _ in range(n)]

    open_kwargs: dict[str, Any] = {}
    if codec_override:
        open_kwargs["codec_name"] = codec_override

    try:
        container = av.open(str(mp4_path))
        stream = container.streams.video[0]
        if codec_override:
            # Re-open with explicit hardware decoder via container options or CodecContext
            container.close()
            container = av.open(str(mp4_path), options={})
            stream = container.streams.video[0]
            # PyAV supports specifying decoder by passing `codec_name` to av.Codec -> av.open doesn't take codec_name directly,
            # instead we decode packets with av.CodecContext.create(codec_override, 'r')!
            return _benchmark_via_explicit_codec(mp4_path, fps, codec_override, rand_frames, n)
    except Exception as exc:
        return {"decoder": codec_override or "cpu", "error": str(exc)}

    tb = float(stream.time_base)
    half_frame = 0.5 / max(fps, 1.0)
    w, h = stream.width, stream.height
    decoder_used = stream.codec_context.name

    # 1. Sequential decode
    t0 = time.perf_counter()
    container.seek(0, backward=True, stream=stream)
    cnt = 0
    for frame in container.decode(stream):
        _ = frame.to_ndarray(format="rgb24")
        cnt += 1
        if cnt >= n:
            break
    seq_s = max(time.perf_counter() - t0, 1e-9)

    # 2. Random seek decode
    lat_ms: list[float] = []
    frames_per_seek: list[int] = []
    t1 = time.perf_counter()
    for fidx in rand_frames:
        req_ts = fidx / fps
        q0 = time.perf_counter()
        container.seek(int(req_ts / tb), backward=True, stream=stream)
        dec = 0
        for frame in container.decode(stream):
            dec += 1
            f_ts = float(frame.pts * tb) if frame.pts is not None else 0.0
            if f_ts + half_frame >= req_ts:
                _ = frame.to_ndarray(format="rgb24")
                break
        lat_ms.append((time.perf_counter() - q0) * 1000.0)
        frames_per_seek.append(dec)
    rand_s = max(time.perf_counter() - t1, 1e-9)
    container.close()

    arr = np.asarray(lat_ms, dtype=np.float64)
    rgb_mb_per_frame = (w * h * 3) / (1024 * 1024)
    return {
        "decoder": decoder_used,
        "resolution": f"{w}x{h}",
        "seq_fps": round(n / seq_s, 1),
        "seq_rgb_mb_s": round((n / seq_s) * rgb_mb_per_frame, 1),
        "rand_seek_fps": round(n / rand_s, 1),
        "rand_seek_p50_ms": round(float(np.percentile(arr, 50)), 3),
        "rand_seek_p99_ms": round(float(np.percentile(arr, 99)), 3),
        "avg_frames_per_seek": round(float(np.mean(frames_per_seek)), 2),
    }


def _benchmark_via_explicit_codec(
    mp4_path: Path,
    fps: float,
    codec_name: str,
    rand_frames: list[int],
    n: int,
) -> dict[str, Any]:
    """Decode demuxed MP4 packets through an explicit NVIDIA CUVID hardware CodecContext."""
    container = av.open(str(mp4_path))
    stream = container.streams.video[0]
    tb = float(stream.time_base)
    half_frame = 0.5 / max(fps, 1.0)
    w, h = stream.width, stream.height

    def _make_ctx() -> av.CodecContext:
        ctx = av.Codec(codec_name, "r").create()
        ctx.extradata = stream.codec_context.extradata
        ctx.open()
        return ctx

    try:
        # 1. Sequential HW decode
        ctx = _make_ctx()
        t0 = time.perf_counter()
        container.seek(0, backward=True, stream=stream)
        cnt = 0
        for pkt in container.demux(stream):
            if pkt.dts is None:
                continue
            for frame in ctx.decode(pkt):
                _ = frame.to_ndarray(format="rgb24")
                cnt += 1
                if cnt >= n:
                    break
            if cnt >= n:
                break
        seq_s = max(time.perf_counter() - t0, 1e-9)
        del ctx

        # 2. Random seek HW decode (reuse single HW decoder context, flush on each seek)
        ctx = _make_ctx()
        lat_ms: list[float] = []
        frames_per_seek: list[int] = []
        t1 = time.perf_counter()
        for fidx in rand_frames:
            req_ts = fidx / fps
            q0 = time.perf_counter()
            container.seek(int(req_ts / tb), backward=True, stream=stream)
            ctx.flush_buffers()
            dec = 0
            found = False
            for pkt in container.demux(stream):
                if pkt.dts is None:
                    continue
                for frame in ctx.decode(pkt):
                    dec += 1
                    f_ts = float(frame.pts * tb) if frame.pts is not None else 0.0
                    if f_ts + half_frame >= req_ts:
                        _ = frame.to_ndarray(format="rgb24")
                        found = True
                        break
                if found:
                    break
            lat_ms.append((time.perf_counter() - q0) * 1000.0)
            frames_per_seek.append(dec)
        rand_s = max(time.perf_counter() - t1, 1e-9)
        del ctx
    except Exception as exc:
        container.close()
        return {"decoder": codec_name, "error": str(exc)}

    container.close()
    arr = np.asarray(lat_ms, dtype=np.float64)
    rgb_mb_per_frame = (w * h * 3) / (1024 * 1024)
    return {
        "decoder": codec_name,
        "resolution": f"{w}x{h}",
        "seq_fps": round(n / seq_s, 1),
        "seq_rgb_mb_s": round((n / seq_s) * rgb_mb_per_frame, 1),
        "rand_seek_fps": round(n / rand_s, 1),
        "rand_seek_p50_ms": round(float(np.percentile(arr, 50)), 3),
        "rand_seek_p99_ms": round(float(np.percentile(arr, 99)), 3),
        "avg_frames_per_seek": round(float(np.mean(frames_per_seek)), 2),
    }


def profile_step_latency_breakdown(root: Path, num_samples: int = 50, seed: int = 42) -> dict[str, Any]:
    """Break down exact wall-clock cost of 1 random multi-camera LeRobot v3.0 sample."""
    with LeRobotV3FormatReader(root) as reader:
        eps = reader.list_episodes()
        rng = random.Random(seed)
        coords = [(ep.episode_id, t) for ep in eps for t in range(ep.num_steps)]
        chosen = [rng.choice(coords) for _ in range(min(num_samples, len(coords)))]

        # Warm up caches
        reader.read_step(*chosen[0])

        t_parquet: list[float] = []
        per_cam_ms: dict[str, list[float]] = {vk: [] for vk in reader._video_keys}
        t_total: list[float] = []

        for ep_id, step_idx in chosen:
            r = reader._ep_rows[ep_id]
            c_idx, f_idx = int(r["data/chunk_index"]), int(r["data/file_index"])
            local_row = (int(r["dataset_from_index"]) + step_idx) - reader._shard_start_row[(c_idx, f_idx)]

            t0 = time.perf_counter()
            tbl = reader._get_parquet_table(reader._data_shard_path(c_idx, f_idx))
            _ = np.asarray(tbl["observation.state"][local_row].as_py(), dtype=np.float32)
            _ = np.asarray(tbl["action"][local_row].as_py(), dtype=np.float32)
            t1 = time.perf_counter()
            t_parquet.append((t1 - t0) * 1000.0)

            for vk in reader._video_keys:
                vc = int(r[f"videos/{vk}/chunk_index"])
                vf = int(r[f"videos/{vk}/file_index"])
                v_from = float(r[f"videos/{vk}/from_timestamp"])
                mp4_p = reader._video_shard_path(vk, vc, vf)
                tc0 = time.perf_counter()
                _ = reader._decode_video_frames(mp4_p, [v_from + step_idx / reader._fps])[0]
                per_cam_ms[vk].append((time.perf_counter() - tc0) * 1000.0)
            t_total.append((time.perf_counter() - t0) * 1000.0)

    mean_total = float(np.mean(t_total))
    mean_pq = float(np.mean(t_parquet))
    cam_summary = {
        vk: {
            "mean_ms": round(float(np.mean(vals)), 3),
            "p99_ms": round(float(np.percentile(vals, 99)), 3),
            "share_pct": round(100.0 * float(np.mean(vals)) / max(mean_total, 1e-9), 2),
        }
        for vk, vals in per_cam_ms.items()
    }
    return {
        "samples": len(chosen),
        "total_mean_ms": round(mean_total, 3),
        "total_p99_ms": round(float(np.percentile(t_total, 99)), 3),
        "parquet_mean_ms": round(mean_pq, 4),
        "parquet_share_pct": round(100.0 * mean_pq / max(mean_total, 1e-9), 2),
        "cameras": cam_summary,
    }


def _worker_read_batch(args: tuple[str, list[tuple[int, int]], int, int]) -> tuple[int, int, float]:
    """Worker process simulating one PyTorch DataLoader worker reading random windows."""
    root_str, coords, obs_h, act_h = args
    total_rgb_bytes = 0
    with LeRobotV3FormatReader(Path(root_str)) as reader:
        # Warm up 1 read (metadata + container open) outside steady-state timer
        if coords:
            reader.read_window(coords[0][0], coords[0][1], obs_horizon=obs_h, action_horizon=act_h)
        t0 = time.perf_counter()
        for ep_id, step_idx in coords:
            win = reader.read_window(ep_id, step_idx, obs_horizon=obs_h, action_horizon=act_h)
            for img in win.images.values():
                total_rgb_bytes += img.nbytes
        worker_elapsed = max(time.perf_counter() - t0, 1e-9)
    return len(coords), total_rgb_bytes, worker_elapsed


def benchmark_multi_worker_scaling(
    root: Path,
    worker_counts: tuple[int, ...] = (1, 2, 4, 8),
    total_samples: int = 320,
    obs_horizon: int = 1,
    action_horizon: int = 16,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Measure steady-state multi-process DataLoader throughput scaling (excluding Windows spawn overhead)."""
    with LeRobotV3FormatReader(root) as reader:
        eps = reader.list_episodes()
    all_coords = [(ep.episode_id, t) for ep in eps for t in range(ep.num_steps)]
    rng = random.Random(seed)
    chosen = [rng.choice(all_coords) for _ in range(min(total_samples, len(all_coords)))]

    results: list[dict[str, Any]] = []
    for nw in worker_counts:
        shards = [chosen[i::nw] for i in range(nw)]
        if nw == 1:
            done, rgb_bytes, steady_s = _worker_read_batch((str(root), shards[0], obs_horizon, action_horizon))
            wall_s = steady_s
        else:
            with ProcessPoolExecutor(max_workers=nw) as pool:
                # Pre-warm pool processes before timing steady-state batch
                list(pool.map(int, range(nw)))
                t0 = time.perf_counter()
                outs = list(pool.map(_worker_read_batch, [(str(root), s, obs_horizon, action_horizon) for s in shards]))
                wall_s = max(time.perf_counter() - t0, 1e-9)
            done = sum(x[0] for x in outs)
            rgb_bytes = sum(x[1] for x in outs)
            steady_s = max(x[2] for x in outs)
        results.append(
            {
                "num_workers": nw,
                "samples": done,
                "steady_elapsed_s": round(steady_s, 3),
                "wall_with_open_s": round(wall_s, 3),
                "steady_throughput_samples_s": round(done / steady_s, 2),
                "effective_rgb_bandwidth_mb_s": round((rgb_bytes / (1024 * 1024)) / steady_s, 2),
            }
        )
    return results


def run_hd_dual_cam_lab(root: Path, num_queries: int = 60) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    video_tpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    decoder_comparison: dict[str, Any] = {}
    for vk in video_keys:
        mp4_0 = root / video_tpl.format(video_key=vk, chunk_index=0, file_index=0)
        if not mp4_0.exists():
            continue
        sw_codec, cuvid_codec = _pick_cuvid_decoder(mp4_0)
        cpu_res = benchmark_single_video_decoder(mp4_0, fps=fps, codec_override=None, num_queries=num_queries)
        hw_res = (
            benchmark_single_video_decoder(mp4_0, fps=fps, codec_override=cuvid_codec, num_queries=num_queries)
            if cuvid_codec
            else {"decoder": "unavailable"}
        )
        decoder_comparison[vk] = {
            "detected_sw_codec": sw_codec,
            "detected_cuvid_codec": cuvid_codec,
            "cpu_soft_decode": cpu_res,
            "nvidia_cuvid_hw_decode": hw_res,
        }

    return {
        "dataset": str(root),
        "fps": fps,
        "cameras": video_keys,
        "cpu_vs_nvdec": decoder_comparison,
        "step_latency_breakdown": profile_step_latency_breakdown(root, num_samples=min(num_queries, 50)),
        "multi_worker_scaling_obs1_act16": benchmark_multi_worker_scaling(
            root, worker_counts=(1, 2, 4, 8), total_samples=max(num_queries * 4, 240), obs_horizon=1, action_horizon=16
        ),
    }
