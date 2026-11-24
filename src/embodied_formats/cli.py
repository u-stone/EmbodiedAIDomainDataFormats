"""Rich interactive CLI (`edlab`) for inspecting, sampling, and benchmarking Embodied AI dataset formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from embodied_formats.common import BaseFormatReader, format_bytes
from embodied_formats.deep_dive import (
    run_full_hdf5_deep_study,
    run_full_lerobot_v3_deep_study,
    run_full_mcap_deep_study,
    run_full_rlds_deep_study,
    run_full_zarr_deep_study,
    run_hd_dual_cam_lab,
)
from embodied_formats.formats import FORMAT_REGISTRY
from embodied_formats.lab.anatomy import deep_analyze, inspect_hf_hub_cache
from embodied_formats.lab.benchmark import benchmark_reader
from embodied_formats.lab.real_samples import (
    fetch_real_hdf5,
    fetch_real_lerobot_v3,
    fetch_real_lerobot_v3_so100,
    fetch_real_mcap,
    fetch_real_rlds,
    fetch_real_zarr,
    real_sample_paths,
)
from embodied_formats.lab.samples import (
    default_sample_paths,
    setup_hdf5_sample,
    setup_lerobot_v3_real,
    setup_mcap_sample,
    setup_rlds_sample,
    setup_zarr_sample,
)

console = Console()
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_REAL_ROOT = Path("data/real")


def _open_reader(fmt: str, path: Path | None, data_root: Path) -> BaseFormatReader:
    if fmt not in FORMAT_REGISTRY:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: {list(FORMAT_REGISTRY.keys())}")
    target = path if path is not None else default_sample_paths(data_root)[fmt]
    return FORMAT_REGISTRY[fmt](target)


def cmd_setup_samples(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    paths = default_sample_paths(data_root)
    console.print(f"[bold cyan]Preparing sample datasets under[/bold cyan] {data_root.resolve()}")

    if not args.skip_download:
        console.print("  [1/5] Downloading real open-source [bold]LeRobot v3.0[/bold] (`lerobot/pusht`) from HuggingFace...")
        setup_lerobot_v3_real(paths["lerobot_v3"], force=args.force)
        console.print(f"        -> Ready: {paths['lerobot_v3']}")
    else:
        console.print("  [1/5] Skipped HuggingFace download (--skip-download).")

    console.print("  [2/5] Preparing native [bold]MCAP[/bold] multi-topic async sample...")
    setup_mcap_sample(paths["mcap"], force=args.force)
    console.print(f"        -> Ready: {paths['mcap']}")

    console.print("  [3/5] Preparing native [bold]HDF5[/bold] (robomimic chunked layout) sample...")
    setup_hdf5_sample(paths["hdf5"], force=args.force)
    console.print(f"        -> Ready: {paths['hdf5']}")

    console.print("  [4/5] Preparing native [bold]Zarr[/bold] (Diffusion Policy ReplayBuffer Blosc-LZ4) sample...")
    setup_zarr_sample(data_root / "zarr", force=args.force)
    console.print(f"        -> Ready: {paths['zarr']}")

    console.print("  [5/5] Preparing native [bold]RLDS / TFRecord[/bold] (CRC32C + Protobuf Example) sample...")
    setup_rlds_sample(paths["rlds"], force=args.force)
    console.print(f"        -> Ready: {paths['rlds']}")

    console.print("[bold green]All 5 format samples are ready![/bold green]")


def cmd_inspect(args: argparse.Namespace) -> None:
    with _open_reader(args.format, Path(args.path) if args.path else None, Path(args.data_root)) as reader:
        rep = reader.inspect()

    summary_text = (
        f"[bold]Format:[/bold] {rep.format_name}   "
        f"[bold]Path:[/bold] {rep.path}\n"
        f"[bold]Disk:[/bold] {format_bytes(rep.disk_bytes)} ({rep.file_count} files)   "
        f"[bold]Episodes:[/bold] {rep.num_episodes}   "
        f"[bold]Total Steps:[/bold] {rep.total_steps}   "
        f"[bold]Avg FPS:[/bold] {rep.avg_fps}   "
        f"[bold]Bytes/Step:[/bold] {format_bytes(rep.disk_bytes / max(rep.total_steps, 1))}"
    )
    console.print(Panel(summary_text, title=f"Dataset Anatomy: {rep.format_name}", border_style="cyan"))

    tbl = Table(title="Schema / Channels / Datasets", show_lines=False)
    tbl.add_column("Name", style="bold green")
    tbl.add_column("Dtype", style="yellow")
    tbl.add_column("Shape / Count")
    tbl.add_column("Encoding / Codec", style="magenta")
    tbl.add_column("Low-Level Details")
    for f in rep.schema:
        tbl.add_row(f.name, f.dtype, f.shape, f.encoding, f.details)
    console.print(tbl)

    console.print(Panel(json.dumps(rep.storage_details, indent=2, ensure_ascii=False), title="Storage Internals", border_style="blue"))

    notes_str = "\n".join(f"• {n}" for n in rep.notes)
    console.print(Panel(notes_str, title="Systems Engineering Notes", border_style="green"))


def cmd_sample(args: argparse.Namespace) -> None:
    with _open_reader(args.format, Path(args.path) if args.path else None, Path(args.data_root)) as reader:
        eps = reader.list_episodes()
        ep = eps[args.ep]
        step = reader.read_step(args.ep, args.step)

        console.print(
            Panel(
                f"Episode {ep.episode_id} (len={ep.num_steps}, fps={ep.fps}) | extra={ep.extra}\n"
                f"Step {step.step_index} @ t={step.timestamp_s:.4f}s | trace={step.extra}\n"
                f"Images: { {k: (v.shape, str(v.dtype)) for k, v in step.images.items()} }\n"
                f"State : {None if step.state is None else (step.state.shape, str(step.state.dtype), step.state[:4].round(3).tolist())}\n"
                f"Action: {None if step.action is None else (step.action.shape, str(step.action.dtype), step.action[:4].round(3).tolist())}\n"
                f"Lang  : {step.language_instruction!r}",
                title=f"Single Step Sample ({args.format})",
                border_style="cyan",
            )
        )

        if args.window:
            win = reader.read_window(args.ep, args.step, obs_horizon=args.obs_horizon, action_horizon=args.action_horizon)
            console.print(
                Panel(
                    f"obs_timestamps_s : {win.timestamps_s.round(4).tolist()}\n"
                    f"obs_images       : { {k: v.shape for k, v in win.images.items()} }\n"
                    f"obs_states       : {None if win.states is None else win.states.shape}\n"
                    f"action_chunk     : {None if win.action_chunk is None else win.action_chunk.shape}",
                    title=f"Temporal Window Sample (obs={args.obs_horizon}, act={args.action_horizon})",
                    border_style="magenta",
                )
            )


def cmd_bench(args: argparse.Namespace) -> None:
    with _open_reader(args.format, Path(args.path) if args.path else None, Path(args.data_root)) as reader:
        res = benchmark_reader(reader, num_samples=args.n)

    tbl = Table(title=f"Access Pattern Benchmark: {res.format_name} (disk={format_bytes(res.disk_bytes)}, {format_bytes(res.bytes_per_step)}/step)")
    tbl.add_column("Pattern", style="bold cyan")
    tbl.add_column("Samples", justify="right")
    tbl.add_column("Throughput (samples/s)", justify="right", style="bold green")
    tbl.add_column("P50 (ms)", justify="right")
    tbl.add_column("P95 (ms)", justify="right")
    tbl.add_column("P99 (ms)", justify="right", style="yellow")

    for p in res.patterns:
        tbl.add_row(
            p.pattern,
            str(p.num_samples),
            f"{p.throughput_sps:.1f}",
            f"{p.p50_ms:.2f}",
            f"{p.p95_ms:.2f}",
            f"{p.p99_ms:.2f}",
        )
    console.print(tbl)


def cmd_compare(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    paths = default_sample_paths(data_root)

    tbl = Table(title=f"5-Format Side-by-Side Comparison (N={args.n} samples/pattern)")
    tbl.add_column("Format", style="bold cyan")
    tbl.add_column("Disk", justify="right")
    tbl.add_column("Bytes/Step", justify="right")
    tbl.add_column("Files", justify="right")
    tbl.add_column("Seq (sps)", justify="right", style="green")
    tbl.add_column("Rand Step (sps)", justify="right", style="yellow")
    tbl.add_column("Rand Win (sps)", justify="right", style="magenta")
    tbl.add_column("Rand P99 (ms)", justify="right")

    for fmt in ("mcap", "hdf5", "zarr", "rlds", "lerobot_v3"):
        p = paths[fmt]
        if not p.exists():
            tbl.add_row(fmt, "[red]missing[/red]", "-", "-", "-", "-", "-", "-")
            continue
        with FORMAT_REGISTRY[fmt](p) as reader:
            rep = reader.inspect()
            res = benchmark_reader(reader, num_samples=args.n)
            p_seq, p_rand, p_win = res.patterns
            tbl.add_row(
                fmt,
                format_bytes(rep.disk_bytes),
                format_bytes(res.bytes_per_step),
                str(rep.file_count),
                f"{p_seq.throughput_sps:.1f}",
                f"{p_rand.throughput_sps:.1f}",
                f"{p_win.throughput_sps:.1f}",
                f"{p_rand.p99_ms:.2f}",
            )
    console.print(tbl)


def cmd_fetch_real(args: argparse.Namespace) -> None:
    real_root = Path(args.real_root)
    paths = real_sample_paths(real_root)
    console.print(f"[bold cyan]Downloading genuine open-source datasets into[/bold cyan] {real_root.resolve()}")

    tasks = [
        ("lerobot_v3", "LeRobot v3.0 (HuggingFace `lerobot/pusht`, ~7.3 MB)", lambda: fetch_real_lerobot_v3(paths["lerobot_v3"], force=args.force)),
        ("zarr", "Zarr (Columbia Diffusion Policy `pusht.zip`, ~29.6 MB)", lambda: fetch_real_zarr(real_root / "zarr_pusht", force=args.force)),
        ("hdf5", "HDF5 (Stanford robomimic `v1.5/lift/ph/demo_v15.hdf5`, ~35 MB)", lambda: fetch_real_hdf5(paths["hdf5"], force=args.force)),
        ("rlds", "RLDS/TFRecord (OXE GCS `berkeley_cable_routing` shard-00004, ~31.8 MB)", lambda: fetch_real_rlds(paths["rlds"], force=args.force)),
        ("mcap", "MCAP (RobotisAI `evButtonPush` episode 0, ~154 MB + YAML/JSON/URDF)", lambda: fetch_real_mcap(paths["mcap"], force=args.force)),
    ]
    selected = set(args.formats) if args.formats else {t[0] for t in tasks}
    for key, label, fn in tasks:
        if key not in selected:
            continue
        console.print(f"  -> Fetching [bold]{label}[/bold] ...")
        out = fn()
        console.print(f"     Ready: [green]{out}[/green]")
    console.print("[bold green]All requested real open-source datasets are ready![/bold green]")


def cmd_anatomy(args: argparse.Namespace) -> None:
    real_root = Path(args.real_root)
    paths = real_sample_paths(real_root)
    target = Path(args.path) if args.path else paths[args.format]
    rep = deep_analyze(args.format, target)

    fs = rep.filesystem
    console.print(
        Panel(
            f"[bold]Format:[/bold] {rep.format_name}   [bold]Path:[/bold] {rep.path}\n"
            f"[bold]Total:[/bold] {format_bytes(fs.total_bytes)} | {fs.total_files} files | {fs.total_dirs} dirs\n"
            f"[bold]File Size Stats:[/bold] min={fs.size_stats['min']}, median={fs.size_stats['median']}, "
            f"p95={fs.size_stats['p95']}, max={fs.size_stats['max']}, <64KB={fs.size_stats['small_files_ratio']}\n"
            f"[bold]Links & Inodes:[/bold] {json.dumps(fs.link_and_cache_details, ensure_ascii=False)}",
            title=f"Layer 1: Filesystem Anatomy — {rep.format_name}",
            border_style="cyan",
        )
    )
    console.print(Panel("\n".join(fs.tree_lines[:40]), title="Directory Tree", border_style="dim"))

    ext_tbl = Table(title="Extension & Role Breakdown")
    ext_tbl.add_column("Extension", style="bold green")
    ext_tbl.add_column("Role", style="yellow")
    ext_tbl.add_column("Files", justify="right")
    ext_tbl.add_column("Size", justify="right")
    ext_tbl.add_column("Share %", justify="right", style="cyan")
    for row in fs.by_extension:
        ext_tbl.add_row(row["ext"], row["role"], str(row["count"]), row["bytes_human"], f"{row['pct']:.2f}%")
    console.print(ext_tbl)

    console.print(
        Panel(
            json.dumps(rep.binary_anatomy, indent=2, ensure_ascii=False),
            title=f"Layer 2: Binary Wire & Container Anatomy — {rep.format_name}",
            border_style="magenta",
        )
    )
    console.print(
        Panel(
            json.dumps(rep.metadata_relations, indent=2, ensure_ascii=False),
            title=f"Layer 3: Non-Binary Metadata & Resolution Chain — {rep.format_name}",
            border_style="green",
        )
    )


def cmd_hf_cache(args: argparse.Namespace) -> None:
    res = inspect_hf_hub_cache(repo_pattern=args.filter)
    console.print(
        Panel(
            json.dumps(res, indent=2, ensure_ascii=False),
            title="HuggingFace Hub Content-Addressed Cache Anatomy (~/.cache/huggingface/hub)",
            border_style="cyan",
        )
    )


def cmd_lerobot_deep(args: argparse.Namespace) -> None:
    target = Path(args.path) if args.path else real_sample_paths(Path(args.real_root))["lerobot_v3"]
    res = run_full_lerobot_v3_deep_study(target, num_seek_queries=args.n)
    console.print(
        Panel(
            json.dumps(res["integrity_audit"], indent=2, ensure_ascii=False),
            title="1. Control-Plane & Mathematical Invariants Audit (Row / Video CFR / Chan-Welford)",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            json.dumps(res["parquet_internals"], indent=2, ensure_ascii=False),
            title="2. Parquet RowGroup, Page Offsets & Predicate Pushdown Statistics",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            json.dumps(res["mp4_seek_profiles"], indent=2, ensure_ascii=False),
            title="3. Raw MP4 `stbl` Byte-Offset Resolution & PyAV Seek Decode Profiler",
            border_style="magenta",
        )
    )
    console.print(
        Panel(
            json.dumps(res["delta_timestamps_simulation"], indent=2, ensure_ascii=False),
            title="4. `delta_timestamps` Query & `*_is_pad` Boundary Mask Trace",
            border_style="green",
        )
    )


def cmd_lerobot_hd(args: argparse.Namespace) -> None:
    target = Path(args.path) if args.path else real_sample_paths(Path(args.real_root))["lerobot_v3_so100"]
    if not (target / "meta" / "info.json").exists():
        console.print(f"[yellow]Dataset not found at {target}, downloading lerobot/svla_so100_pickplace (~448 MB)...[/yellow]")
        fetch_real_lerobot_v3_so100(target)
    res = run_hd_dual_cam_lab(target, num_queries=args.n)
    console.print(
        Panel(
            json.dumps(res["cpu_vs_nvdec"], indent=2, ensure_ascii=False),
            title="1. CPU Soft Decode vs NVIDIA RTX NVDEC (CUVID) Hardware Decode",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            json.dumps(res["step_latency_breakdown"], indent=2, ensure_ascii=False),
            title="2. Dual-Camera Single-Step Latency Breakdown (Parquet vs Top Cam vs Wrist Cam)",
            border_style="magenta",
        )
    )
    console.print(
        Panel(
            json.dumps(res["multi_worker_scaling_obs1_act16"], indent=2, ensure_ascii=False),
            title="3. Multi-Process DataLoader Scaling (obs=1, action_chunk=16)",
            border_style="green",
        )
    )


def cmd_deep(args: argparse.Namespace) -> None:
    fmt = args.format
    target = Path(args.path) if args.path else real_sample_paths(Path(args.real_root))[fmt]
    if fmt == "lerobot_v3":
        res = run_full_lerobot_v3_deep_study(target, num_seek_queries=args.n)
    elif fmt == "rlds":
        res = run_full_rlds_deep_study(target, num_queries=args.n)
    elif fmt == "hdf5":
        res = run_full_hdf5_deep_study(target, num_queries=args.n)
    elif fmt == "zarr":
        res = run_full_zarr_deep_study(target, num_queries=args.n)
    elif fmt == "mcap":
        res = run_full_mcap_deep_study(target)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    for section_key, section_val in res.items():
        if section_key == "dataset_path":
            continue
        console.print(
            Panel(
                json.dumps(section_val, indent=2, ensure_ascii=False),
                title=f"Deep Dive [{fmt}] — {section_key}",
                border_style="cyan",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="edlab",
        description="Embodied AI Dataset Formats Anatomy & Benchmark Lab (MCAP / HDF5 / Zarr / RLDS / LeRobot v3.0)",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory for sample datasets")
    parser.add_argument("--real-root", default=str(DEFAULT_REAL_ROOT), help="Root directory for real open-source datasets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_deep = sub.add_parser("deep", help="Format-specific low-level deep-dive study (lerobot_v3 / rlds / hdf5 / zarr / mcap)")
    p_deep.add_argument("format", choices=list(FORMAT_REGISTRY.keys()))
    p_deep.add_argument("--path", default=None, help="Optional custom dataset path")
    p_deep.add_argument("-n", type=int, default=60, help="Number of queries for profiling")
    p_deep.set_defaults(func=cmd_deep)

    p_hf = sub.add_parser("hf-cache", help="Inspect ~/.cache/huggingface/hub Git-like refs/blobs/snapshots & symlinks")
    p_hf.add_argument("--filter", default="datasets--", help="Substring filter for repo cache dirs")
    p_hf.set_defaults(func=cmd_hf_cache)

    p_lr = sub.add_parser("lerobot-deep", help="Ultra-deep 5-subsystem study of a LeRobot v3.0 dataset")
    p_lr.add_argument("--path", default=None, help="Path to LeRobot v3.0 dataset (default: real lerobot/pusht)")
    p_lr.add_argument("-n", type=int, default=50, help="Number of random video seek queries to profile")
    p_lr.set_defaults(func=cmd_lerobot_deep)

    p_hd = sub.add_parser("lerobot-hd", help="Dual-camera HD LeRobot v3.0 lab: CPU vs NVDEC + latency breakdown + multi-worker scaling")
    p_hd.add_argument("--path", default=None, help="Path to HD LeRobot v3.0 dataset (default: real lerobot/svla_so100_pickplace)")
    p_hd.add_argument("-n", type=int, default=60, help="Number of queries per benchmark")
    p_hd.set_defaults(func=cmd_lerobot_hd)

    p_fetch = sub.add_parser("fetch-real", help="Download genuine open-source datasets for all 5 formats")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.add_argument("--formats", nargs="*", choices=list(FORMAT_REGISTRY.keys()), default=None)
    p_fetch.set_defaults(func=cmd_fetch_real)

    p_anat = sub.add_parser("anatomy", help="Deep 3-layer anatomy (Filesystem / Binary Wire / Metadata Graph)")
    p_anat.add_argument("format", choices=list(FORMAT_REGISTRY.keys()))
    p_anat.add_argument("--path", default=None)
    p_anat.set_defaults(func=cmd_anatomy)

    p_setup = sub.add_parser("setup-samples", help="Download real LeRobot v3.0 sample + prepare native samples for other formats")
    p_setup.add_argument("--force", action="store_true", help="Overwrite existing samples")
    p_setup.add_argument("--skip-download", action="store_true", help="Skip HuggingFace download")
    p_setup.set_defaults(func=cmd_setup_samples)

    p_insp = sub.add_parser("inspect", help="Inspect low-level structure, schema, compression, and indexing")
    p_insp.add_argument("format", choices=list(FORMAT_REGISTRY.keys()))
    p_insp.add_argument("--path", default=None, help="Optional custom dataset path")
    p_insp.set_defaults(func=cmd_inspect)

    p_samp = sub.add_parser("sample", help="Read and trace a single step (and optional temporal window)")
    p_samp.add_argument("format", choices=list(FORMAT_REGISTRY.keys()))
    p_samp.add_argument("--path", default=None)
    p_samp.add_argument("--ep", type=int, default=0)
    p_samp.add_argument("--step", type=int, default=5)
    p_samp.add_argument("--window", action="store_true", help="Also read obs_horizon + action_horizon window")
    p_samp.add_argument("--obs-horizon", type=int, default=2)
    p_samp.add_argument("--action-horizon", type=int, default=16)
    p_samp.set_defaults(func=cmd_sample)

    p_bench = sub.add_parser("bench", help="Benchmark sequential / random_step / random_window access patterns")
    p_bench.add_argument("format", choices=list(FORMAT_REGISTRY.keys()))
    p_bench.add_argument("--path", default=None)
    p_bench.add_argument("-n", type=int, default=80, help="Number of samples per access pattern")
    p_bench.set_defaults(func=cmd_bench)

    p_cmp = sub.add_parser("compare", help="Compare all 5 formats side-by-side")
    p_cmp.add_argument("-n", type=int, default=60, help="Number of samples per access pattern")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
