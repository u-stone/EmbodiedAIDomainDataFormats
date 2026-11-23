"""Deep 3-layer anatomy engine for real Embodied AI datasets:
  Layer 1: Filesystem Layout (directory tree, extension breakdown, file-size / small-file stats)
  Layer 2: Binary Wire & Container Anatomy (magic bytes, headers/footers, MP4 boxes, Parquet footers,
           Blosc chunk headers, HDF5 superblock, TFRecord CRC32C framing, MCAP footer & chunk index)
  Layer 3: Non-Binary Metadata & Control-Plane Relationships (JSON/YAML/.zarray/features.json/URDF
           and exact resolution chain from metadata to binary offsets / timestamps)
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import h5py
import numpy as np
import pyarrow.parquet as pq
import zarr
from mcap.reader import make_reader

from embodied_formats.common import format_bytes
from embodied_formats.formats.rlds_format import decode_tf_example, masked_crc32c


TEXT_EXTS = {".json", ".jsonl", ".yaml", ".yml", ".md", ".txt", ".urdf", ".xml", ".zgroup", ".zarray", ".zattrs"}
BINARY_INDEX_NAMES = {"tasks.parquet"}


def _hexdump_line(data: bytes, max_bytes: int = 32) -> str:
    chunk = data[:max_bytes]
    hex_part = " ".join(f"{b:02X}" for b in chunk)
    ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
    return f"{hex_part}  |{ascii_part}|"


@dataclass
class FilesystemAnatomy:
    root: str
    total_files: int
    total_dirs: int
    total_bytes: int
    small_files_lt_64k: int
    size_stats: dict[str, str]
    by_extension: list[dict[str, Any]]
    tree_lines: list[str]
    link_and_cache_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeepAnatomyReport:
    format_name: str
    path: str
    filesystem: FilesystemAnatomy
    binary_anatomy: dict[str, Any] = field(default_factory=dict)
    metadata_relations: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer 1: Filesystem Layout
# ---------------------------------------------------------------------------
def analyze_filesystem(root: Path, max_children_per_dir: int = 10) -> FilesystemAnatomy:
    files: list[Path] = []
    dirs: list[Path] = []
    if root.is_file():
        files = [root]
    else:
        for p in sorted(root.rglob("*")):
            if p.is_dir():
                dirs.append(p)
            elif p.is_file():
                files.append(p)

    sizes = [f.stat().st_size for f in files]
    total_bytes = sum(sizes)
    small_count = sum(1 for s in sizes if s < 64 * 1024)

    arr = np.asarray(sizes, dtype=np.int64) if sizes else np.asarray([0])
    size_stats = {
        "min": format_bytes(int(np.min(arr))),
        "median": format_bytes(float(np.median(arr))),
        "p95": format_bytes(float(np.percentile(arr, 95))),
        "max": format_bytes(int(np.max(arr))),
        "small_files_ratio": f"{small_count}/{len(files)} ({(100.0 * small_count / max(len(files), 1)):.1f}%)",
    }

    ext_groups: dict[str, list[int]] = defaultdict(list)
    for f, sz in zip(files, sizes):
        if f.name in (".zgroup", ".zarray", ".zattrs"):
            ext = f.name
        elif ".tfrecord" in f.name:
            ext = ".tfrecord*"
        else:
            ext = f.suffix.lower() or "<no-ext (chunk)>"
        ext_groups[ext].append(sz)

    by_ext: list[dict[str, Any]] = []
    for ext, sz_list in sorted(ext_groups.items(), key=lambda kv: sum(kv[1]), reverse=True):
        ext_bytes = sum(sz_list)
        if ext in TEXT_EXTS:
            role = "text-metadata"
        elif ext == ".parquet":
            role = "binary-columnar / index"
        else:
            role = "binary-data"
        by_ext.append(
            {
                "ext": ext,
                "role": role,
                "count": len(sz_list),
                "bytes": ext_bytes,
                "bytes_human": format_bytes(ext_bytes),
                "pct": round(100.0 * ext_bytes / max(total_bytes, 1), 2),
            }
        )

    # Render compact directory tree
    tree_lines: list[str] = []
    if root.is_file():
        tree_lines.append(f"{root.name} ({format_bytes(root.stat().st_size)})")
    else:
        tree_lines.append(f"{root.name}/")

        def _walk_tree(cur: Path, prefix: str) -> None:
            entries = sorted(cur.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            hidden_count = 0
            if len(entries) > max_children_per_dir:
                shown = entries[: max_children_per_dir - 2] + [None] + entries[-1:]
                hidden_count = len(entries) - (max_children_per_dir - 1)
            else:
                shown = entries
            for idx, entry in enumerate(shown):
                is_last = idx == len(shown) - 1
                branch = "└── " if is_last else "├── "
                if entry is None:
                    tree_lines.append(f"{prefix}{branch}... ({hidden_count} more entries omitted)")
                    continue
                if entry.is_dir():
                    tree_lines.append(f"{prefix}{branch}{entry.name}/")
                    _walk_tree(entry, prefix + ("    " if is_last else "│   "))
                else:
                    tree_lines.append(f"{prefix}{branch}{entry.name} ({format_bytes(entry.stat().st_size)})")

        _walk_tree(root, "")

    # Inspect POSIX/NTFS symlinks, hardlinks (st_nlink / st_ino), and hidden .cache/huggingface metadata
    symlinks: list[dict[str, str]] = []
    hardlinks_count = 0
    unique_inodes: set[int] = set()
    hf_download_meta_samples: list[dict[str, Any]] = []
    for f in files:
        try:
            st = f.lstat()
            unique_inodes.add(st.st_ino)
            if f.is_symlink():
                symlinks.append({"link": str(f.relative_to(root)), "target": str(f.readlink())})
            elif st.st_nlink > 1:
                hardlinks_count += 1
        except OSError:
            pass
        if ".cache" in f.parts and f.suffix == ".metadata" and len(hf_download_meta_samples) < 3:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            hf_download_meta_samples.append(
                {
                    "file": str(f.relative_to(root)),
                    "commit_hash": lines[0] if len(lines) > 0 else "",
                    "etag_or_sha256": lines[1] if len(lines) > 1 else "",
                    "timestamp": lines[2] if len(lines) > 2 else "",
                }
            )

    link_details: dict[str, Any] = {
        "symlink_count": len(symlinks),
        "symlink_samples": symlinks[:5],
        "hardlinked_files_nlink_gt_1": hardlinks_count,
        "unique_inodes_vs_files": f"{len(unique_inodes)} unique inodes / {len(files)} file entries",
    }
    if hf_download_meta_samples:
        link_details["hf_local_dir_cache_metadata_samples"] = hf_download_meta_samples

    return FilesystemAnatomy(
        root=str(root),
        total_files=len(files),
        total_dirs=len(dirs),
        total_bytes=total_bytes,
        small_files_lt_64k=small_count,
        size_stats=size_stats,
        by_extension=by_ext,
        tree_lines=tree_lines,
        link_and_cache_details=link_details,
    )


def inspect_hf_hub_cache(repo_pattern: str = "datasets--") -> dict[str, Any]:
    """Inspect real `~/.cache/huggingface/hub/` Git-like content-addressed layout (refs/blobs/snapshots)."""
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_hub.exists():
        return {"cache_dir": str(hf_hub), "exists": False}

    repos_out: list[dict[str, Any]] = []
    for repo_dir in sorted(hf_hub.iterdir()):
        if not repo_dir.is_dir() or repo_pattern not in repo_dir.name:
            continue
        refs: dict[str, str] = {}
        refs_dir = repo_dir / "refs"
        if refs_dir.exists():
            for rf in refs_dir.rglob("*"):
                if rf.is_file():
                    refs[str(rf.relative_to(refs_dir))] = rf.read_text(encoding="utf-8", errors="replace").strip()

        blobs_dir = repo_dir / "blobs"
        blobs = [p for p in blobs_dir.iterdir() if p.is_file()] if blobs_dir.exists() else []
        blob_inodes = {p.stat().st_ino: p.name for p in blobs}

        snap_dir = repo_dir / "snapshots"
        snap_entries: list[dict[str, Any]] = []
        if snap_dir.exists():
            for sf in sorted(snap_dir.rglob("*")):
                if sf.is_file() or sf.is_symlink():
                    lst = sf.lstat()
                    is_sym = sf.is_symlink()
                    target = str(sf.readlink()) if is_sym else ""
                    linked_blob = blob_inodes.get(lst.st_ino, "")
                    snap_entries.append(
                        {
                            "path": str(sf.relative_to(snap_dir)),
                            "size": format_bytes(sf.stat().st_size) if sf.exists() else "broken",
                            "is_symlink": is_sym,
                            "symlink_target": target,
                            "st_nlink": lst.st_nlink,
                            "shares_inode_with_blob": linked_blob[:16] + "..." if linked_blob else "no (copied)",
                        }
                    )

        repos_out.append(
            {
                "repo_cache_dir": repo_dir.name,
                "refs": refs,
                "blob_count": len(blobs),
                "blob_total_size": format_bytes(sum(b.stat().st_size for b in blobs)),
                "sample_blob_names": [b.name for b in sorted(blobs)[:4]],
                "snapshot_file_count": len(snap_entries),
                "snapshot_samples": snap_entries[:6],
            }
        )

    return {
        "cache_dir": str(hf_hub),
        "exists": True,
        "dataset_repos_found": len(repos_out),
        "repos": repos_out,
    }


# ---------------------------------------------------------------------------
# Layer 2 & 3: Format-Specific Binary & Metadata Anatomy
# ---------------------------------------------------------------------------
def _parse_mp4_boxes(f: Any, start: int, end: int, depth: int = 0) -> list[dict[str, Any]]:
    """Parse ISO BMFF (MP4) Box/Atom hierarchy directly from raw bytes."""
    CONTAINER_BOXES = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}
    boxes: list[dict[str, Any]] = []
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        size, btype = struct.unpack(">I4s", hdr)
        header_len = 8
        if size == 1:
            ext = f.read(8)
            size = struct.unpack(">Q", ext)[0]
            header_len = 16
        elif size == 0:
            size = end - pos
        if size < header_len or pos + size > end:
            break
        box_info: dict[str, Any] = {
            "type": btype.decode("ascii", errors="replace"),
            "offset": pos,
            "size": size,
            "size_human": format_bytes(size),
        }
        # Extract keyframe count from `stss` (Sync Sample Box) and sample count from `stsz`
        if btype == b"stss" and size >= 16:
            f.seek(pos + 12)
            entry_count = struct.unpack(">I", f.read(4))[0]
            box_info["keyframe_entries_in_stss"] = entry_count
        elif btype == b"stsz" and size >= 20:
            f.seek(pos + 12)
            sample_size, sample_count = struct.unpack(">II", f.read(8))
            box_info["total_frames_in_stsz"] = sample_count
            box_info["const_sample_size"] = sample_size
        elif btype in CONTAINER_BOXES and depth < 5:
            box_info["children"] = _parse_mp4_boxes(f, pos + header_len, pos + size, depth + 1)
        boxes.append(box_info)
        pos += size
    return boxes


def _analyze_lerobot_v3(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    binary: dict[str, Any] = {}
    meta: dict[str, Any] = {}

    # 1. Parquet binary anatomy
    data_pqs = sorted((root / "data").rglob("*.parquet"))
    if data_pqs:
        p0 = data_pqs[0]
        fsize = p0.stat().st_size
        with p0.open("rb") as f:
            head = f.read(16)
            f.seek(max(0, fsize - 8))
            tail = f.read(8)
        footer_len = struct.unpack("<I", tail[:4])[0]
        pf = pq.ParquetFile(p0)
        rg0 = pf.metadata.row_group(0)
        cols_detail = [
            {
                "col": rg0.column(i).path_in_schema,
                "physical_type": rg0.column(i).physical_type,
                "compression": rg0.column(i).compression,
                "encodings": list(rg0.column(i).encodings),
                "uncompressed": format_bytes(rg0.column(i).total_uncompressed_size),
                "compressed": format_bytes(rg0.column(i).total_compressed_size),
            }
            for i in range(min(rg0.num_columns, 8))
        ]
        binary["parquet_shard0"] = {
            "file": str(p0.relative_to(root)),
            "file_size": format_bytes(fsize),
            "header_magic": head[:4].decode("ascii", errors="replace"),
            "footer_magic": tail[4:8].decode("ascii", errors="replace"),
            "thrift_footer_bytes": footer_len,
            "num_rows": pf.metadata.num_rows,
            "num_row_groups": pf.metadata.num_row_groups,
            "row_group_0_columns": cols_detail,
        }

    # 2. MP4 ISO BMFF box tree + PyAV GOP
    mp4s = sorted((root / "videos").rglob("*.mp4"))
    if mp4s:
        m0 = mp4s[0]
        msize = m0.stat().st_size
        with m0.open("rb") as f:
            head32 = f.read(32)
            boxes = _parse_mp4_boxes(f, 0, msize)
        with av.open(str(m0)) as c:
            s = c.streams.video[0]
            kfs = []
            n_pkts = 0
            for pkt in c.demux(s):
                if pkt.dts is None:
                    continue
                if pkt.is_keyframe:
                    kfs.append(n_pkts)
                n_pkts += 1
                if n_pkts >= 2000:
                    break
            gops = np.diff(kfs) if len(kfs) >= 2 else np.array([n_pkts])
            av_info = {
                "codec": s.codec_context.name,
                "pix_fmt": s.codec_context.pix_fmt,
                "resolution": f"{s.width}x{s.height}",
                "fps": str(s.average_rate),
                "inspected_packets": n_pkts,
                "keyframes": len(kfs),
                "min_gop": int(np.min(gops)) if len(gops) else 0,
                "mean_gop": round(float(np.mean(gops)), 2) if len(gops) else 0.0,
                "max_gop": int(np.max(gops)) if len(gops) else 0,
            }
        binary["mp4_shard0"] = {
            "file": str(m0.relative_to(root)),
            "file_size": format_bytes(msize),
            "header_hex": _hexdump_line(head32, 24),
            "iso_bmff_boxes": boxes,
            "pyav_packet_gop": av_info,
        }

    # 3. Non-binary metadata & control-plane graph
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    ep_pqs = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    ep_sample = pq.read_table(ep_pqs[0]).slice(0, 2).to_pylist() if ep_pqs else []
    meta = {
        "control_plane_files": {
            "meta/info.json": {
                "codebase_version": info.get("codebase_version"),
                "fps": info.get("fps"),
                "total_episodes": info.get("total_episodes"),
                "total_frames": info.get("total_frames"),
                "data_path_template": info.get("data_path"),
                "video_path_template": info.get("video_path"),
                "feature_keys": list(info.get("features", {}).keys()),
            },
            "meta/stats.json": {
                "present": (root / "meta" / "stats.json").exists(),
                "size": format_bytes((root / "meta" / "stats.json").stat().st_size)
                if (root / "meta" / "stats.json").exists()
                else "0 B",
            },
            "meta/tasks.parquet": {
                "present": (root / "meta" / "tasks.parquet").exists(),
                "rows": pq.read_table(root / "meta" / "tasks.parquet").to_pylist()
                if (root / "meta" / "tasks.parquet").exists()
                else [],
            },
            "meta/episodes/*.parquet": {
                "shard_count": len(ep_pqs),
                "columns": pq.ParquetFile(ep_pqs[0]).schema.names if ep_pqs else [],
                "first_2_episodes_index_rows": [
                    {
                        k: r.get(k)
                        for k in r
                        if any(tok in k for tok in ("episode_index", "dataset_", "chunk_index", "file_index", "timestamp", "length"))
                    }
                    for r in ep_sample
                ],
            },
        },
        "resolution_chain": [
            "1. meta/info.json 声明 data_path/video_path 分片路径模板与全局 features Schema",
            "2. meta/episodes/chunk-XXX/file-XXX.parquet 记录每条 Episode 的 (data/chunk_index, data/file_index, dataset_from_index..to_index) 与 (videos/*/from_timestamp..to_timestamp)",
            "3. 读表格数据：定位 data/chunk-c/file-f.parquet，将全局行号减去分片起始行号得到局部行号",
            "4. 读视觉数据：定位 videos/<cam>/chunk-c/file-f.mp4，查 MP4 moov->stbl->stss 关键帧表 Seek 到目标时间槽解码",
            "5. 读任务文本与归一化参数：分别查 meta/tasks.parquet (task_index) 与 meta/stats.json",
        ],
    }
    return binary, meta


def _analyze_zarr(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    binary: dict[str, Any] = {}
    meta: dict[str, Any] = {}

    # Inspect raw 16-byte Blosc header on the first image chunk file
    img_dir = root / "data" / "img"
    chunk_file = None
    if img_dir.exists():
        cands = [p for p in sorted(img_dir.iterdir()) if p.is_file() and not p.name.startswith(".")]
        if cands:
            chunk_file = cands[0]
    if chunk_file is not None:
        raw = chunk_file.read_bytes()
        # Blosc 16-byte header: version, versionlz, flags, typesize, nbytes(u32), blocksize(u32), cbytes(u32)
        ver, verlz, flags, typesize, nbytes, blocksize, cbytes = struct.unpack("<BBBBIII", raw[:16])
        binary["blosc_chunk_sample"] = {
            "file": str(chunk_file.relative_to(root)),
            "file_size_bytes": len(raw),
            "header_hex_32b": _hexdump_line(raw, 32),
            "blosc_header": {
                "version": ver,
                "compressor_version_lz": verlz,
                "flags_hex": f"0x{flags:02X}",
                "byte_shuffle": bool(flags & 0x01),
                "memcpy_uncompressed": bool(flags & 0x02),
                "bit_shuffle": bool(flags & 0x04),
                "typesize": typesize,
                "uncompressed_nbytes": f"{nbytes} ({format_bytes(nbytes)})",
                "blocksize": blocksize,
                "compressed_cbytes": f"{cbytes} ({format_bytes(cbytes)})",
                "compression_ratio": f"{(nbytes / max(cbytes, 1)):.2f}x",
            },
        }

    # Collect all .zgroup / .zarray / .zattrs files
    zmeta_files: dict[str, Any] = {}
    for p in sorted(root.rglob(".z*")):
        if p.is_file():
            zmeta_files[str(p.relative_to(root))] = json.loads(p.read_text(encoding="utf-8"))

    zroot = zarr.open_group(str(root), mode="r")
    ep_ends = zroot["meta"]["episode_ends"][:10].tolist() if "meta" in zroot and "episode_ends" in zroot["meta"] else []
    meta = {
        "json_control_files": zmeta_files,
        "episode_ends_first_10": ep_ends,
        "resolution_chain": [
            "1. 根目录 .zgroup 声明 Zarr v2 Group 层级（data/ 与 meta/）",
            "2. 每个数组目录下的 .zarray (JSON) 声明 shape、chunks、dtype、order(C/F)、compressor(Blosc/LZ4) 与 dimension_separator('.')",
            "3. meta/episode_ends 给出每条 Episode 在扁平全局数组中的结束行号 [e0, e1, ...]",
            "4. 计算全局行号 g = ep_start + step，再整除 chunk_dim0 得到具体磁盘分块文件名（如 data/img/0.0.0.0）",
            "5. 每个 Chunk 文件前 16 字节是 Blosc Header，无锁并发直接读取并解压对应 Block",
        ],
    }
    return binary, meta


def _analyze_hdf5(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    h5_files = [root] if root.is_file() else sorted(list(root.glob("*.hdf5")) + list(root.glob("*.h5")))
    f0 = h5_files[0]
    with f0.open("rb") as bf:
        hdr = bf.read(48)

    sig_ok = hdr[:8] == b"\x89HDF\r\n\x1a\n"
    sb_ver = hdr[8]
    offset_size = hdr[13] if sb_ver in (0, 1) else hdr[9]
    length_size = hdr[14] if sb_ver in (0, 1) else hdr[10]

    datasets_summary: list[dict[str, Any]] = []
    embedded_attrs: dict[str, Any] = {}
    with h5py.File(f0, "r") as hf:
        for k, v in hf.attrs.items():
            embedded_attrs[f"/{k}"] = str(v)[:300]
        if "data" in hf:
            for k, v in hf["data"].attrs.items():
                embedded_attrs[f"/data.{k}"] = str(v)[:400]
            demo_keys = sorted(hf["data"].keys())
            if demo_keys:
                d0 = hf["data"][demo_keys[0]]
                for k, v in d0.attrs.items():
                    embedded_attrs[f"/data/{demo_keys[0]}.{k}"] = str(v)[:200]

                def _collect(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Dataset):
                        logical = int(obj.size * obj.dtype.itemsize)
                        stored = int(obj.id.get_storage_size())
                        datasets_summary.append(
                            {
                                "path": name,
                                "shape": str(obj.shape),
                                "dtype": str(obj.dtype),
                                "chunks": str(obj.chunks or "contiguous"),
                                "compression": obj.compression or "none",
                                "logical": format_bytes(logical),
                                "stored": format_bytes(stored),
                                "ratio": f"{(logical / max(stored, 1)):.2f}x",
                            }
                        )

                d0.visititems(_collect)

    binary = {
        "file": f0.name,
        "header_hex_32b": _hexdump_line(hdr, 32),
        "superblock": {
            "magic_signature_ok": sig_ok,
            "superblock_version": sb_ver,
            "offset_field_bytes": offset_size,
            "length_field_bytes": length_size,
        },
        "first_demo_datasets": datasets_summary,
    }
    meta = {
        "external_text_metadata_files": "NONE (HDF5 将所有元数据自包含在单文件内部的 Group/Dataset Attributes 中)",
        "embedded_hdf5_attributes": embedded_attrs,
        "resolution_chain": [
            "1. 文件头 8 字节 Magic (89 48 44 46 0D 0A 1A 0A) 后紧跟 Superblock，指向 Root Group 的 Symbol Table / B-Tree",
            "2. 通过内部 B-Tree 逐级解析路径 /data -> /data/demo_0 -> /data/demo_0/obs/...",
            "3. 环境配置、仿真器参数等非二进制 JSON 字符串（如 /data.env_args）以 HDF5 Attribute 形式内嵌在 Group 节点上",
            "4. Dataset 叶子节点决定物理存储是 Contiguous（直接算基址+步长偏移）还是 Chunked（再查一层 Chunk B-Tree）",
        ],
    }
    return binary, meta


def _analyze_rlds(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    shards = sorted(p for p in root.rglob("*") if p.is_file() and ".tfrecord" in p.name)
    s0 = shards[0]
    with s0.open("rb") as f:
        hdr12 = f.read(12)
        rec0_len = struct.unpack("<Q", hdr12[:8])[0]
        rec0_len_crc = struct.unpack("<I", hdr12[8:12])[0]
        len_crc_ok = rec0_len_crc == masked_crc32c(hdr12[:8])
        payload0 = f.read(rec0_len)
        rec0_pay_crc = struct.unpack("<I", f.read(4))[0]
        pay_crc_ok = rec0_pay_crc == masked_crc32c(payload0)

    # Scan all records in shard 0 for size stats
    rec_sizes: list[int] = []
    with s0.open("rb") as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                break
            ln = struct.unpack("<Q", h[:8])[0]
            rec_sizes.append(ln)
            f.seek(ln + 4, 1)

    decoded0 = decode_tf_example(payload0)
    img_bytes_total = 0
    float_bytes_total = 0
    feature_summary: list[dict[str, Any]] = []
    for k in sorted(decoded0.keys()):
        kind, vals = decoded0[k]
        if kind == "bytes":
            b_sum = sum(len(x) for x in vals)
            is_jpeg = bool(vals and vals[0][:2] == b"\xff\xd8")
            is_png = bool(vals and vals[0][:4] == b"\x89PNG")
            if is_jpeg or is_png:
                img_bytes_total += b_sum
            feature_summary.append(
                {
                    "key": k,
                    "pb_type": "BytesList",
                    "count": len(vals),
                    "total_bytes": format_bytes(b_sum),
                    "magic": "JPEG (FF D8)" if is_jpeg else ("PNG (89 50)" if is_png else "text/raw"),
                }
            )
        elif kind == "float":
            b_sum = len(vals) * 4
            float_bytes_total += b_sum
            feature_summary.append({"key": k, "pb_type": "FloatList", "count": len(vals), "total_bytes": format_bytes(b_sum)})
        else:
            feature_summary.append({"key": k, "pb_type": "Int64List", "count": len(vals)})

    binary = {
        "shard0_file": s0.name,
        "header_hex_32b": _hexdump_line(hdr12 + payload0[:20], 32),
        "record_0_framing": {
            "length_u64_le": f"{rec0_len} ({format_bytes(rec0_len)})",
            "length_masked_crc32c": f"0x{rec0_len_crc:08X} (verified={len_crc_ok})",
            "payload_masked_crc32c": f"0x{rec0_pay_crc:08X} (verified={pay_crc_ok})",
            "image_payload_share_pct": round(100.0 * img_bytes_total / max(rec0_len, 1), 2),
            "float_payload_share_pct": round(100.0 * float_bytes_total / max(rec0_len, 1), 2),
        },
        "shard0_records_stats": {
            "episode_records_in_shard": len(rec_sizes),
            "min_record": format_bytes(min(rec_sizes)),
            "median_record": format_bytes(float(np.median(rec_sizes))),
            "max_record": format_bytes(max(rec_sizes)),
        },
        "record_0_protobuf_features": feature_summary,
    }

    dinfo = json.loads((root / "dataset_info.json").read_text(encoding="utf-8")) if (root / "dataset_info.json").exists() else {}
    feats = json.loads((root / "features.json").read_text(encoding="utf-8")) if (root / "features.json").exists() else {}
    meta = {
        "dataset_info.json": {
            "name": dinfo.get("name"),
            "version": dinfo.get("version"),
            "fileFormat": dinfo.get("fileFormat"),
            "splits": [
                {
                    "name": sp.get("name"),
                    "num_shards": len(sp.get("shardLengths", [])),
                    "total_episodes": sum(int(x) for x in sp.get("shardLengths", [])),
                    "first_5_shard_lengths": sp.get("shardLengths", [])[:5],
                }
                for sp in dinfo.get("splits", [])
            ],
        },
        "features.json": {
            "present": bool(feats),
            "top_level_type": feats.get("pythonClassName"),
            "top_level_keys": list(feats.get("featuresDict", {}).get("features", {}).keys()),
        },
        "resolution_chain": [
            "1. dataset_info.json 声明 fileFormat('tfrecord')、split 列表及每个分片包含的 Episode 条数 (shardLengths)",
            "2. features.json 声明嵌套 Schema（如 steps/observation/image 为 uint8[H,W,3] JPEG，steps/action 为 float32[7]）",
            "3. .tfrecord 分片内部没有任何随机索引，按 [uint64 len][crc32c][tf.train.Example][crc32c] 顺序堆叠",
            "4. Protobuf Example 内只存扁平化的一维 FloatList/BytesList，必须结合 features.json 的 shape 才能 reshape 回 (T, D) 张量",
        ],
    }
    return binary, meta


def _analyze_mcap(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    mcap_files = [root] if root.is_file() else sorted(root.glob("*.mcap"))
    m0 = mcap_files[0]
    fsize = m0.stat().st_size
    with m0.open("rb") as bf:
        start_magic = bf.read(8)
        # MCAP Footer is 29 bytes (1 opcode 0x02 + 8 len + 8 summary_start + 8 summary_offset_start + 4 crc) before 8-byte End Magic
        bf.seek(max(0, fsize - 37))
        tail37 = bf.read(37)
    end_magic = tail37[-8:]
    footer_bytes = tail37[:-8]
    footer_opcode = footer_bytes[0]
    footer_len, summary_start, summary_offset_start, summary_crc = struct.unpack("<QQQI", footer_bytes[1:29])

    with m0.open("rb") as bf:
        reader = make_reader(bf)
        header = reader.get_header()
        summary = reader.get_summary()
        stats = summary.statistics if summary else None
        chunks_comp: dict[str, int] = defaultdict(int)
        uncomp_total = 0
        comp_total = 0
        if summary:
            for ci in summary.chunk_indexes:
                chunks_comp[ci.compression or "none"] += 1
                uncomp_total += ci.uncompressed_size
                comp_total += ci.compressed_size

        channels_info = []
        if summary:
            dur_s = max(1e-6, (stats.message_end_time - stats.message_start_time) / 1e9) if stats else 1.0
            for cid, ch in sorted(summary.channels.items()):
                sch = summary.schemas.get(ch.schema_id)
                cnt = stats.channel_message_counts.get(cid, 0) if stats else 0
                channels_info.append(
                    {
                        "id": cid,
                        "topic": ch.topic,
                        "encoding": ch.message_encoding,
                        "schema": sch.name if sch else "none",
                        "schema_encoding": sch.encoding if sch else "none",
                        "messages": cnt,
                        "approx_hz": round(cnt / dur_s, 2),
                    }
                )

    binary = {
        "file": m0.name,
        "file_size": format_bytes(fsize),
        "start_magic_hex": _hexdump_line(start_magic, 8),
        "end_magic_hex": _hexdump_line(end_magic, 8),
        "footer_record": {
            "opcode": f"0x{footer_opcode:02X} (Footer)",
            "record_len": footer_len,
            "summary_start_offset": f"{summary_start} ({format_bytes(summary_start)})",
            "summary_offset_start": summary_offset_start,
            "summary_crc32": f"0x{summary_crc:08X}",
            "summary_section_bytes": format_bytes(fsize - summary_start),
        },
        "header": {"profile": header.profile, "library": header.library},
        "chunks_compression": {
            "chunk_counts_by_codec": dict(chunks_comp),
            "uncompressed_total": format_bytes(uncomp_total),
            "compressed_total": format_bytes(comp_total),
            "compression_ratio": f"{(uncomp_total / max(comp_total, 1)):.2f}x",
        },
        "channels": channels_info,
    }

    sidecar_files: dict[str, Any] = {}
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in (".yaml", ".yml", ".json", ".urdf"):
                txt = p.read_text(encoding="utf-8", errors="replace")
                sidecar_files[p.name] = {
                    "size": format_bytes(p.stat().st_size),
                    "preview": txt[:400].strip(),
                }

    meta = {
        "sidecar_non_binary_files": sidecar_files,
        "resolution_chain": [
            "1. 外部可选 sidecar（ROS 2 metadata.yaml / episode_info.json / robot.urdf）提供录制元信息、话题清单与机器人运动学树",
            "2. 读 .mcap 文件先跳到末尾倒数 37 字节读取 Footer，拿到 summary_start 偏移量",
            "3. 跳转到 Summary Section 一次性加载全部 Schema（内嵌 ROS2 msg / Protobuf 定义）、Channel 与 ChunkIndex",
            "4. 通过 ChunkIndex 的 [start_time, end_time] 与 byte_offset 定位目标时间段所在的压缩 Chunk，解压后按纳秒时间戳回放/对齐",
        ],
    }
    return binary, meta


def deep_analyze(fmt: str, path: Path) -> DeepAnatomyReport:
    fs = analyze_filesystem(path)
    if fmt == "lerobot_v3":
        binary, meta = _analyze_lerobot_v3(path)
    elif fmt == "zarr":
        binary, meta = _analyze_zarr(path)
    elif fmt == "hdf5":
        binary, meta = _analyze_hdf5(path)
    elif fmt == "rlds":
        binary, meta = _analyze_rlds(path)
    elif fmt == "mcap":
        binary, meta = _analyze_mcap(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return DeepAnatomyReport(
        format_name=fmt,
        path=str(path),
        filesystem=fs,
        binary_anatomy=binary,
        metadata_relations=meta,
    )
