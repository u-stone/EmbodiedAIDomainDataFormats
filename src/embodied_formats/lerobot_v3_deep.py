"""Ultra-deep systems anatomy & verification lab specifically for LeRobot v3.0.

Covers 5 low-level systems subsystems without black-box `lerobot` wrappers:
  1. Control-plane & mathematical invariant audit (global row contiguity, CFR video time-slot
     drift, Parquet vs MP4 frame count closure, per-episode -> global Chan/Welford stats check).
  2. Parquet physical RowGroup & Page offset anatomy (file_offset, dictionary_page_offset,
     data_page_offset, encodings, and episode_index Min/Max statistics for predicate pushdown).
  3. Raw ISO BMFF (MP4) `stbl` binary table parser (`stts`, `stss`, `stsc`, `stsz`, `stco`/`co64`)
     performing exact timestamp -> keyframe -> physical `mdat` byte-offset resolution.
  4. Real PyAV seek & decode profiler (packets demuxed per random seek vs sequential decode).
  5. `delta_timestamps` query tracer with boundary clamping and `*_is_pad` mask generation.
"""

from __future__ import annotations

import bisect
import json
import random
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

from embodied_formats.common import format_bytes


# ---------------------------------------------------------------------------
# 1. Raw ISO BMFF (MP4) `stbl` Binary Parser — Exact Byte-Offset Resolution
# ---------------------------------------------------------------------------
@dataclass
class MP4SampleTable:
    file_path: str
    file_size: int
    moov_offset: int
    mdat_offset: int
    is_faststart: bool  # True if moov precedes mdat
    timescale: int
    total_samples: int
    keyframes_1based: list[int]  # from `stss` (1-based sample numbers)
    sample_sizes: list[int]  # from `stsz`
    chunk_offsets: list[int]  # from `stco` or `co64`
    stsc_entries: list[tuple[int, int, int]]  # (first_chunk_1based, samples_per_chunk, sdi)
    stts_entries: list[tuple[int, int]]  # (sample_count, sample_delta)

    def _build_sample_offsets(self) -> list[int]:
        """Expand (stsc, stco/co64, stsz) into exact byte offset in `mdat` for every sample."""
        offsets: list[int] = []
        num_chunks = len(self.chunk_offsets)
        sample_idx = 0
        for i, (first_chunk, spc, _) in enumerate(self.stsc_entries):
            next_first = self.stsc_entries[i + 1][0] if i + 1 < len(self.stsc_entries) else num_chunks + 1
            for chunk_1based in range(first_chunk, min(next_first, num_chunks + 1)):
                cur_off = self.chunk_offsets[chunk_1based - 1]
                for _ in range(spc):
                    if sample_idx >= self.total_samples:
                        return offsets
                    offsets.append(cur_off)
                    cur_off += self.sample_sizes[sample_idx]
                    sample_idx += 1
        return offsets

    def resolve_sample(self, sample_idx_0based: int) -> dict[str, Any]:
        """Resolve a 0-based frame index to its exact keyframe, decode distance, and byte offsets."""
        if not (0 <= sample_idx_0based < self.total_samples):
            raise IndexError(f"sample_idx {sample_idx_0based} out of range [0, {self.total_samples})")
        offsets = self._build_sample_offsets()
        s1 = sample_idx_0based + 1

        if self.keyframes_1based:
            pos = bisect.bisect_right(self.keyframes_1based, s1) - 1
            kf_1based = self.keyframes_1based[max(0, pos)]
        else:
            # No stss box means every sample is a sync sample (all-intra)
            kf_1based = s1
        kf_0based = kf_1based - 1

        decode_frames_needed = (sample_idx_0based - kf_0based) + 1
        bytes_from_kf_to_target = sum(self.sample_sizes[kf_0based : sample_idx_0based + 1])

        return {
            "target_sample_0based": sample_idx_0based,
            "target_mdat_byte_offset": offsets[sample_idx_0based],
            "target_frame_bytes": self.sample_sizes[sample_idx_0based],
            "preceding_keyframe_0based": kf_0based,
            "keyframe_mdat_byte_offset": offsets[kf_0based],
            "keyframe_bytes": self.sample_sizes[kf_0based],
            "frames_to_decode_from_keyframe": decode_frames_needed,
            "compressed_bytes_to_read": bytes_from_kf_to_target,
        }


def parse_mp4_sample_table(mp4_path: Path) -> MP4SampleTable:
    """Parse raw ISO BMFF boxes to extract mdhd/stts/stss/stsc/stsz/stco(co64)."""
    file_size = mp4_path.stat().st_size
    moov_offset = -1
    mdat_offset = -1
    timescale = 1000
    keyframes_1based: list[int] = []
    sample_sizes: list[int] = []
    chunk_offsets: list[int] = []
    stsc_entries: list[tuple[int, int, int]] = []
    stts_entries: list[tuple[int, int]] = []

    CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}

    with mp4_path.open("rb") as f:
        def _walk(start: int, end: int) -> None:
            nonlocal moov_offset, mdat_offset, timescale, keyframes_1based
            nonlocal sample_sizes, chunk_offsets, stsc_entries, stts_entries
            pos = start
            while pos + 8 <= end:
                f.seek(pos)
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                size, btype = struct.unpack(">I4s", hdr)
                hlen = 8
                if size == 1:
                    size = struct.unpack(">Q", f.read(8))[0]
                    hlen = 16
                elif size == 0:
                    size = end - pos
                if size < hlen or pos + size > end:
                    break

                if btype == b"moov":
                    moov_offset = pos
                elif btype == b"mdat" and mdat_offset < 0:
                    mdat_offset = pos
                elif btype == b"mdhd":
                    f.seek(pos + hlen)
                    ver = f.read(1)[0]
                    f.read(3)  # flags
                    if ver == 1:
                        f.read(16)  # creation + mod (8+8)
                        timescale = struct.unpack(">I", f.read(4))[0]
                    else:
                        f.read(8)  # creation + mod (4+4)
                        timescale = struct.unpack(">I", f.read(4))[0]
                elif btype == b"stts":
                    f.seek(pos + hlen + 4)
                    cnt = struct.unpack(">I", f.read(4))[0]
                    raw = f.read(cnt * 8)
                    stts_entries = [struct.unpack(">II", raw[i * 8 : (i + 1) * 8]) for i in range(cnt)]
                elif btype == b"stss":
                    f.seek(pos + hlen + 4)
                    cnt = struct.unpack(">I", f.read(4))[0]
                    raw = f.read(cnt * 4)
                    keyframes_1based = list(struct.unpack(f">{cnt}I", raw))
                elif btype == b"stsc":
                    f.seek(pos + hlen + 4)
                    cnt = struct.unpack(">I", f.read(4))[0]
                    raw = f.read(cnt * 12)
                    stsc_entries = [struct.unpack(">III", raw[i * 12 : (i + 1) * 12]) for i in range(cnt)]
                elif btype == b"stsz":
                    f.seek(pos + hlen + 4)
                    const_sz, cnt = struct.unpack(">II", f.read(8))
                    if const_sz != 0:
                        sample_sizes = [const_sz] * cnt
                    else:
                        raw = f.read(cnt * 4)
                        sample_sizes = list(struct.unpack(f">{cnt}I", raw))
                elif btype == b"stco":
                    f.seek(pos + hlen + 4)
                    cnt = struct.unpack(">I", f.read(4))[0]
                    raw = f.read(cnt * 4)
                    chunk_offsets = list(struct.unpack(f">{cnt}I", raw))
                elif btype == b"co64":
                    f.seek(pos + hlen + 4)
                    cnt = struct.unpack(">I", f.read(4))[0]
                    raw = f.read(cnt * 8)
                    chunk_offsets = list(struct.unpack(f">{cnt}Q", raw))

                if btype in CONTAINERS:
                    _walk(pos + hlen, pos + size)
                pos += size

        _walk(0, file_size)

    return MP4SampleTable(
        file_path=str(mp4_path),
        file_size=file_size,
        moov_offset=moov_offset,
        mdat_offset=mdat_offset,
        is_faststart=(0 <= moov_offset < mdat_offset),
        timescale=timescale,
        total_samples=len(sample_sizes),
        keyframes_1based=keyframes_1based,
        sample_sizes=sample_sizes,
        chunk_offsets=chunk_offsets,
        stsc_entries=stsc_entries,
        stts_entries=stts_entries,
    )


# ---------------------------------------------------------------------------
# 2. Control-Plane & Mathematical Invariants Audit
# ---------------------------------------------------------------------------
def audit_v3_integrity(root: Path) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    stats_path = root / "meta" / "stats.json"
    global_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    fps = float(info["fps"])
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    ep_rows: list[dict[str, Any]] = []
    for ef in ep_files:
        ep_rows.extend(pq.read_table(ef).to_pylist())
    ep_rows.sort(key=lambda r: int(r["episode_index"]))

    # Invariant 1: Global row contiguity
    row_gaps: list[str] = []
    expected_from = 0
    shard_expected_rows: dict[tuple[int, int], int] = {}
    for r in ep_rows:
        g_from = int(r["dataset_from_index"])
        g_to = int(r["dataset_to_index"])
        ep_len = int(r["length"])
        if g_from != expected_from:
            row_gaps.append(f"ep {r['episode_index']}: expected from={expected_from}, got {g_from}")
        if g_to - g_from != ep_len:
            row_gaps.append(f"ep {r['episode_index']}: to-from={g_to - g_from} != length={ep_len}")
        expected_from = g_to
        skey = (int(r["data/chunk_index"]), int(r["data/file_index"]))
        shard_expected_rows[skey] = shard_expected_rows.get(skey, 0) + ep_len

    # Invariant 2: Physical Parquet shard row counts match metadata
    data_tpl = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    parquet_shard_checks: list[dict[str, Any]] = []
    for (ci, fi), exp_rows in sorted(shard_expected_rows.items()):
        p = root / data_tpl.format(chunk_index=ci, file_index=fi)
        actual_rows = pq.ParquetFile(p).metadata.num_rows if p.exists() else -1
        parquet_shard_checks.append(
            {
                "shard": f"chunk-{ci:03d}/file-{fi:03d}.parquet",
                "expected_rows_from_meta": exp_rows,
                "actual_parquet_rows": actual_rows,
                "match": (exp_rows == actual_rows),
            }
        )

    # Invariant 3: Video time-slot contiguity, CFR drift, and MP4 physical frame count closure
    video_tpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    video_checks: dict[str, Any] = {}
    for vk in video_keys:
        per_vshard_expected: dict[tuple[int, int], int] = {}
        per_vshard_last_to: dict[tuple[int, int], float] = {}
        slot_gaps: list[str] = []
        drifts_ms: list[float] = []
        for r in ep_rows:
            vc = int(r[f"videos/{vk}/chunk_index"])
            vf = int(r[f"videos/{vk}/file_index"])
            v_from = float(r[f"videos/{vk}/from_timestamp"])
            v_to = float(r[f"videos/{vk}/to_timestamp"])
            ep_len = int(r["length"])

            vkey = (vc, vf)
            if vkey in per_vshard_last_to:
                gap_ms = abs(v_from - per_vshard_last_to[vkey]) * 1000.0
                if gap_ms > 5.0:
                    slot_gaps.append(f"ep {r['episode_index']}: gap={gap_ms:.2f}ms")
            per_vshard_last_to[vkey] = v_to
            per_vshard_expected[vkey] = per_vshard_expected.get(vkey, 0) + ep_len

            expected_dur = ep_len / fps
            actual_dur = v_to - v_from
            drifts_ms.append(abs(actual_dur - expected_dur) * 1000.0)

        vshard_reports = []
        for (vc, vf), exp_frames in sorted(per_vshard_expected.items()):
            mp4_p = root / video_tpl.format(video_key=vk, chunk_index=vc, file_index=vf)
            stbl = parse_mp4_sample_table(mp4_p)
            vshard_reports.append(
                {
                    "shard": f"chunk-{vc:03d}/file-{vf:03d}.mp4",
                    "file_size": format_bytes(stbl.file_size),
                    "expected_frames_from_meta": exp_frames,
                    "actual_mp4_stsz_frames": stbl.total_samples,
                    "frame_count_match": (exp_frames == stbl.total_samples),
                    "keyframes_in_stss": len(stbl.keyframes_1based),
                    "avg_gop_from_stss": round(stbl.total_samples / max(len(stbl.keyframes_1based), 1), 3),
                    "moov_before_mdat_faststart": stbl.is_faststart,
                    "moov_offset": stbl.moov_offset,
                    "mdat_offset": stbl.mdat_offset,
                }
            )
        video_checks[vk] = {
            "max_cfr_duration_drift_ms": round(max(drifts_ms) if drifts_ms else 0.0, 4),
            "slot_discontinuities_gt_5ms": len(slot_gaps),
            "shards": vshard_reports,
        }

    # Invariant 4: Chan/Welford parallel variance check (reconstruct global action mean/std from per-episode stats)
    welford_check: dict[str, Any] = {}
    if "action" in global_stats and "stats/action/mean" in ep_rows[0]:
        total_n = 0
        agg_mean = None
        agg_m2 = None
        for r in ep_rows:
            cnt = int(r["stats/action/count"][0])
            m = np.asarray(r["stats/action/mean"], dtype=np.float64)
            s = np.asarray(r["stats/action/std"], dtype=np.float64)
            m2 = (s ** 2) * cnt
            if agg_mean is None:
                total_n = cnt
                agg_mean = m.copy()
                agg_m2 = m2.copy()
            else:
                delta = m - agg_mean
                new_n = total_n + cnt
                agg_mean = agg_mean + delta * (cnt / new_n)
                agg_m2 = agg_m2 + m2 + (delta ** 2) * (total_n * cnt / new_n)
                total_n = new_n
        rec_std = np.sqrt(agg_m2 / max(total_n, 1))
        ref_mean = np.asarray(global_stats["action"]["mean"], dtype=np.float64)
        ref_std = np.asarray(global_stats["action"]["std"], dtype=np.float64)
        welford_check = {
            "feature": "action",
            "reconstructed_count": total_n,
            "max_abs_mean_diff": float(np.max(np.abs(agg_mean - ref_mean))),
            "max_abs_std_diff": float(np.max(np.abs(rec_std - ref_std))),
            "has_q01_q99_in_global_stats": ("q01" in global_stats["action"] and "q99" in global_stats["action"]),
        }

    return {
        "codebase_version": info.get("codebase_version"),
        "fps": fps,
        "total_episodes": len(ep_rows),
        "total_frames_info_json": info.get("total_frames"),
        "total_frames_from_episodes_parquet": expected_from,
        "global_row_contiguity_ok": (len(row_gaps) == 0 and expected_from == info.get("total_frames")),
        "row_gaps": row_gaps[:5],
        "parquet_shard_checks": parquet_shard_checks,
        "video_checks": video_checks,
        "chan_welford_stats_reconstruction": welford_check,
    }


# ---------------------------------------------------------------------------
# 3. Parquet RowGroup & Predicate Pushdown Deep Dive
# ---------------------------------------------------------------------------
def analyze_parquet_internals(root: Path) -> dict[str, Any]:
    data_pqs = sorted((root / "data").rglob("*.parquet"))
    p0 = data_pqs[0]
    pf = pq.ParquetFile(p0)
    md = pf.metadata

    rg_summaries = []
    for rg_i in range(min(md.num_row_groups, 6)):
        rg = md.row_group(rg_i)
        ep_col_stats = None
        col_offsets = []
        for c_i in range(rg.num_columns):
            cc = rg.column(c_i)
            if cc.path_in_schema == "episode_index" and cc.is_stats_set:
                ep_col_stats = {"min": cc.statistics.min, "max": cc.statistics.max}
            if c_i < 4:
                col_offsets.append(
                    {
                        "col": cc.path_in_schema,
                        "dict_page_off": cc.dictionary_page_offset,
                        "data_page_off": cc.data_page_offset,
                        "compressed": format_bytes(cc.total_compressed_size),
                        "uncompressed": format_bytes(cc.total_uncompressed_size),
                    }
                )
        rg_summaries.append(
            {
                "row_group": rg_i,
                "num_rows": rg.num_rows,
                "total_byte_size": format_bytes(rg.total_byte_size),
                "episode_index_minmax": ep_col_stats,
                "first_4_columns": col_offsets,
            }
        )

    return {
        "shard0": str(p0.relative_to(root)),
        "num_data_shards": len(data_pqs),
        "total_rows_shard0": md.num_rows,
        "num_row_groups": md.num_row_groups,
        "created_by": md.created_by,
        "first_6_row_groups": rg_summaries,
    }


# ---------------------------------------------------------------------------
# 4. PyAV Real Packet-Level Seek Profiler
# ---------------------------------------------------------------------------
def profile_mp4_seek(mp4_path: Path, fps: float, num_queries: int = 50, seed: int = 42) -> dict[str, Any]:
    stbl = parse_mp4_sample_table(mp4_path)
    rng = random.Random(seed)
    total_frames = stbl.total_samples
    n = min(num_queries, total_frames)
    rand_frames = [rng.randint(0, total_frames - 1) for _ in range(n)]

    # 1. Exact binary address trace on 3 representative frames
    sample_traces = [stbl.resolve_sample(idx) for idx in [0, min(165, total_frames - 1), total_frames - 1]]

    # 2. Measure actual PyAV decode latency & frames decoded per random seek vs sequential decode
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        tb = float(stream.time_base)
        half_frame = 0.5 / max(fps, 1.0)

        # Sequential decode of first `n` frames (1 seek to 0)
        t0 = time.perf_counter()
        container.seek(0, backward=True, stream=stream)
        seq_count = 0
        for frame in container.decode(stream):
            _ = frame.to_ndarray(format="rgb24")
            seq_count += 1
            if seq_count >= n:
                break
        seq_s = max(time.perf_counter() - t0, 1e-9)

        # Random seeks
        seek_lat_ms: list[float] = []
        frames_decoded_per_seek: list[int] = []
        t1 = time.perf_counter()
        for fidx in rand_frames:
            req_ts = fidx / fps
            q0 = time.perf_counter()
            container.seek(int(req_ts / tb), backward=True, stream=stream)
            decoded_cnt = 0
            for frame in container.decode(stream):
                decoded_cnt += 1
                f_ts = float(frame.pts * tb) if frame.pts is not None else 0.0
                if f_ts + half_frame >= req_ts:
                    _ = frame.to_ndarray(format="rgb24")
                    break
            seek_lat_ms.append((time.perf_counter() - q0) * 1000.0)
            frames_decoded_per_seek.append(decoded_cnt)
        rand_s = max(time.perf_counter() - t1, 1e-9)

    lat_arr = np.asarray(seek_lat_ms, dtype=np.float64)
    return {
        "video_file": mp4_path.name,
        "resolution": f"{stream.width}x{stream.height}",
        "codec": stream.codec_context.name,
        "is_faststart_moov_first": stbl.is_faststart,
        "total_frames": total_frames,
        "keyframes_in_stss": len(stbl.keyframes_1based),
        "sample_byte_offset_traces": sample_traces,
        "sequential_decode_fps": round(n / seq_s, 1),
        "random_seek_decode_fps": round(n / rand_s, 1),
        "random_seek_p50_ms": round(float(np.percentile(lat_arr, 50)), 3),
        "random_seek_p99_ms": round(float(np.percentile(lat_arr, 99)), 3),
        "avg_frames_decoded_per_random_seek": round(float(np.mean(frames_decoded_per_seek)), 2),
        "max_frames_decoded_per_random_seek": int(max(frames_decoded_per_seek)),
    }


# ---------------------------------------------------------------------------
# 5. `delta_timestamps` Query & `*_is_pad` Mask Simulator
# ---------------------------------------------------------------------------
def simulate_delta_timestamps(root: Path, episode_id: int = 1, action_horizon: int = 8) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]

    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    ep_rows: list[dict[str, Any]] = []
    for ef in ep_files:
        ep_rows.extend(pq.read_table(ef).to_pylist())
    ep_rows.sort(key=lambda r: int(r["episode_index"]))

    ep = ep_rows[min(episode_id, len(ep_rows) - 1)]
    g_from = int(ep["dataset_from_index"])
    g_to = int(ep["dataset_to_index"])
    ep_len = g_to - g_from

    # Query near the end of the episode to demonstrate boundary clamping + action_is_pad mask
    anchor_step = max(0, ep_len - 3)
    obs_deltas = [-1.0 / fps, 0.0]
    act_deltas = [i / fps for i in range(action_horizon)]

    obs_raw_steps = [anchor_step + round(d * fps) for d in obs_deltas]
    obs_clamped = [max(0, min(ep_len - 1, s)) for s in obs_raw_steps]
    obs_is_pad = [s < 0 or s >= ep_len for s in obs_raw_steps]

    act_raw_steps = [anchor_step + round(d * fps) for d in act_deltas]
    act_clamped = [max(0, min(ep_len - 1, s)) for s in act_raw_steps]
    act_is_pad = [s < 0 or s >= ep_len for s in act_raw_steps]

    cam0 = video_keys[0] if video_keys else None
    video_ts = (
        [round(float(ep[f"videos/{cam0}/from_timestamp"]) + s / fps, 4) for s in obs_clamped]
        if cam0
        else []
    )

    return {
        "episode_id": int(ep["episode_index"]),
        "episode_global_row_range": [g_from, g_to],
        "episode_length": ep_len,
        "anchor_step_near_tail": anchor_step,
        "obs_query": {
            "delta_timestamps_s": [round(d, 4) for d in obs_deltas],
            "raw_steps": obs_raw_steps,
            "clamped_steps": obs_clamped,
            "global_parquet_rows": [g_from + s for s in obs_clamped],
            "obs_is_pad": obs_is_pad,
            "video_target_timestamps_s": video_ts,
        },
        "action_chunk_query": {
            "delta_timestamps_s": [round(d, 4) for d in act_deltas],
            "raw_steps": act_raw_steps,
            "clamped_steps": act_clamped,
            "global_parquet_rows": [g_from + s for s in act_clamped],
            "action_is_pad": act_is_pad,
        },
    }


def run_full_lerobot_v3_deep_study(root: Path, num_seek_queries: int = 50) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    video_tpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")

    seek_profiles: dict[str, Any] = {}
    for vk in video_keys:
        mp4_0 = root / video_tpl.format(video_key=vk, chunk_index=0, file_index=0)
        if mp4_0.exists():
            seek_profiles[vk] = profile_mp4_seek(mp4_0, fps=fps, num_queries=num_seek_queries)

    return {
        "dataset_path": str(root),
        "integrity_audit": audit_v3_integrity(root),
        "parquet_internals": analyze_parquet_internals(root),
        "mp4_seek_profiles": seek_profiles,
        "delta_timestamps_simulation": simulate_delta_timestamps(root, episode_id=1, action_horizon=8),
    }
