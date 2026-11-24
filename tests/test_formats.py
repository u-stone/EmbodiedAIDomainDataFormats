"""End-to-end verification tests for all 5 Embodied AI dataset format readers, benchmarks, and deep-dives."""

from __future__ import annotations

from pathlib import Path

import pytest

from embodied_formats.deep_dive import (
    run_full_hdf5_deep_study,
    run_full_lerobot_v3_deep_study,
    run_full_mcap_deep_study,
    run_full_rlds_deep_study,
    run_full_zarr_deep_study,
)
from embodied_formats.formats import (
    HDF5FormatReader,
    LeRobotV3FormatReader,
    MCAPFormatReader,
    RLDSFormatReader,
    ZarrFormatReader,
)
from embodied_formats.lab.benchmark import benchmark_reader
from embodied_formats.lab.real_samples import real_sample_paths
from embodied_formats.lab.samples import (
    setup_hdf5_sample,
    setup_mcap_sample,
    setup_rlds_sample,
    setup_zarr_sample,
)


def _assert_reader_contract(reader) -> None:
    rep = reader.inspect()
    assert rep.num_episodes > 0
    assert rep.total_steps > 0
    assert len(rep.schema) > 0

    eps = reader.list_episodes()
    assert len(eps) == rep.num_episodes

    step = reader.read_step(0, min(5, eps[0].num_steps - 1))
    assert len(step.images) >= 1
    first_img = next(iter(step.images.values()))
    assert first_img.ndim == 3 and first_img.shape[2] == 3
    assert step.action is not None

    win = reader.read_window(0, min(5, eps[0].num_steps - 1), obs_horizon=2, action_horizon=8)
    assert len(win.timestamps_s) == 2
    assert win.action_chunk is not None and win.action_chunk.shape[0] == 8
    first_win_img = next(iter(win.images.values()))
    assert first_win_img.shape[0] == 2

    bench = benchmark_reader(reader, num_samples=10, obs_horizon=2, action_horizon=8)
    assert len(bench.patterns) == 3
    assert all(p.throughput_sps > 0 for p in bench.patterns)


def test_mcap_reader(tmp_path: Path) -> None:
    p = setup_mcap_sample(tmp_path / "mcap", num_episodes=2, steps_per_ep=20)
    with MCAPFormatReader(p) as r:
        _assert_reader_contract(r)


def test_hdf5_reader(tmp_path: Path) -> None:
    p = setup_hdf5_sample(tmp_path / "hdf5", num_episodes=2, steps_per_ep=20)
    with HDF5FormatReader(p) as r:
        _assert_reader_contract(r)


def test_zarr_reader(tmp_path: Path) -> None:
    p = setup_zarr_sample(tmp_path / "zarr", num_episodes=2, steps_per_ep=20)
    with ZarrFormatReader(p) as r:
        _assert_reader_contract(r)


def test_rlds_reader(tmp_path: Path) -> None:
    p = setup_rlds_sample(tmp_path / "rlds", num_episodes=2, steps_per_ep=20)
    with RLDSFormatReader(p) as r:
        _assert_reader_contract(r)


def test_lerobot_v3_real_if_present() -> None:
    real_path = Path("data/real/lerobot_v3_pusht")
    if not (real_path / "meta" / "info.json").exists():
        pytest.skip("Real lerobot/pusht sample not downloaded yet")
    with LeRobotV3FormatReader(real_path) as r:
        _assert_reader_contract(r)

    deep = run_full_lerobot_v3_deep_study(real_path, num_seek_queries=10)
    assert deep["integrity_audit"]["global_row_contiguity_ok"] is True
    assert deep["parquet_internals"]["num_row_groups"] >= 1
    assert len(deep["mp4_seek_profiles"]) >= 1


def test_real_deep_dives_if_present() -> None:
    paths = real_sample_paths(Path("data/real"))
    if paths["rlds"].exists():
        r = run_full_rlds_deep_study(paths["rlds"], num_queries=6)
        assert r["tfrecord_framing_and_crc32c_audit"]["all_len_crc32c_ok"] is True
        assert r["tfrecord_framing_and_crc32c_audit"]["all_payload_crc32c_ok"] is True
    if paths["hdf5"].exists():
        h = run_full_hdf5_deep_study(paths["hdf5"], num_queries=20)
        assert h["superblock_binary"]["magic_signature_ok"] is True
        assert h["storage_btree_and_embedded_metadata"]["num_demos"] > 0
    if paths["zarr"].exists():
        z = run_full_zarr_deep_study(paths["zarr"], num_queries=15)
        assert z["blosc_all_chunks_scan"]["total_chunk_files"] > 0
    if paths["mcap"].exists():
        m = run_full_mcap_deep_study(paths["mcap"])
        assert m["chunks_and_ros2_cdr_decode"]["chunk_count"] > 0
        assert len(m["chunks_and_ros2_cdr_decode"]["decoded_ros2_cdr_samples"]) >= 2
