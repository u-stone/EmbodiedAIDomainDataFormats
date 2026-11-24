"""Access-pattern I/O & decoding benchmark engine for Embodied AI dataset formats.

Simulates the 3 canonical data-loading patterns in robotics / VLA training:
  1. sequential    : contiguous scan (pretraining, statistics computation, offline QA)
  2. random_step   : i.i.d. random (episode, step) sampling across the dataset
  3. random_window : random (episode, step) + obs_horizon=2 + action_horizon=16 chunking
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from embodied_formats.common import BaseFormatReader


@dataclass
class PatternBenchmark:
    pattern: str
    num_samples: int
    total_time_s: float
    throughput_sps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class FormatBenchmarkResult:
    format_name: str
    disk_bytes: int
    total_steps: int
    bytes_per_step: float
    patterns: list[PatternBenchmark]


def _run_timed(fn, items: list[tuple[int, int]], pattern_name: str) -> PatternBenchmark:
    latencies_ms: list[float] = []
    t_start = time.perf_counter()
    for ep_id, step_idx in items:
        t0 = time.perf_counter()
        fn(ep_id, step_idx)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    total_s = max(time.perf_counter() - t_start, 1e-9)
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return PatternBenchmark(
        pattern=pattern_name,
        num_samples=len(items),
        total_time_s=round(total_s, 4),
        throughput_sps=round(len(items) / total_s, 2),
        p50_ms=round(float(np.percentile(arr, 50)), 3),
        p95_ms=round(float(np.percentile(arr, 95)), 3),
        p99_ms=round(float(np.percentile(arr, 99)), 3),
    )


def benchmark_reader(
    reader: BaseFormatReader,
    num_samples: int = 80,
    obs_horizon: int = 2,
    action_horizon: int = 16,
    seed: int = 42,
) -> FormatBenchmarkResult:
    """Run all 3 canonical access patterns on a reader and collect latency percentiles."""
    report = reader.inspect()
    episodes = [ep for ep in reader.list_episodes() if ep.num_steps > 0]
    if not episodes:
        raise ValueError(f"Dataset {reader.path} has no non-empty episodes")

    # Build flat pool of (ep_id, step_idx)
    all_coords: list[tuple[int, int]] = []
    for ep in episodes:
        for t in range(ep.num_steps):
            all_coords.append((ep.episode_id, t))

    n = min(num_samples, len(all_coords))
    rng = random.Random(seed)

    seq_coords = all_coords[:n]
    rand_coords = [rng.choice(all_coords) for _ in range(n)]
    win_coords = [rng.choice(all_coords) for _ in range(n)]

    # Warm up 1 read to exclude one-time metadata/container init from steady-state I/O
    reader.read_step(*seq_coords[0])

    p_seq = _run_timed(reader.read_step, seq_coords, "sequential")
    p_rand = _run_timed(reader.read_step, rand_coords, "random_step")
    p_win = _run_timed(
        lambda ep, t: reader.read_window(ep, t, obs_horizon=obs_horizon, action_horizon=action_horizon),
        win_coords,
        f"random_window(obs={obs_horizon},act={action_horizon})",
    )

    bytes_per_step = report.disk_bytes / max(report.total_steps, 1)
    return FormatBenchmarkResult(
        format_name=reader.format_name,
        disk_bytes=report.disk_bytes,
        total_steps=report.total_steps,
        bytes_per_step=round(bytes_per_step, 1),
        patterns=[p_seq, p_rand, p_win],
    )
