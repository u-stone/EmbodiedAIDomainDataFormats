# Embodied AI Dataset Formats Anatomy & Benchmark Lab

面向系统程序员的具身智能（Embodied AI / VLA）主流数据集格式**底层解剖、统一只读访问与 I/O 压测实验项目**（不做跨格式转换，聚焦“文件系统 / 二进制结构 / 非二进制元数据组织 / I/O 与解码性能”）。

覆盖工业界与开源社区 5 种最具代表性的格式：
1. **LeRobot v3.0**（当前 VLA 开源主流 — HuggingFace / OpenPI / SmolVLA / GR00T）
2. **RLDS / TFRecord**（传统大规模 VLA 生态 — Open X-Embodiment / OpenVLA，零 TensorFlow 纯二进制 CRC32C + Protobuf）
3. **HDF5 / robomimic**（单机经典训练层 — robomimic / ALOHA / ACT）
4. **Zarr**（单机高并发训练层 — Diffusion Policy ReplayBuffer）
5. **ROS 2 bag / MCAP**（采集与日志层 — ROS 2 / Foxglove / Rerun，零 ROS 2 纯 Python CDR 解码）

---

## 深度研究文档索引（5 种格式独立落档）

| 文档 | 内容摘要 |
| --- | --- |
| [`docs/REAL_DATASETS_ANATOMY.md`](docs/REAL_DATASETS_ANATOMY.md) | 5 份真实开源生产数据集的三层横向解剖：文件系统 & `~/.cache/huggingface/hub/` 类 Git 内容寻址机制、二进制 Magic/Header/Box/CRC32C、非二进制元数据寻址链 |
| [`docs/LEROBOT_V3_DEEP_DIVE.md`](docs/LEROBOT_V3_DEEP_DIVE.md) | **LeRobot v3.0**：5 大数学不变式与 Chan/Welford 统计重构、Parquet RowGroup & Predicate Pushdown、MP4 `stbl` 精确字节寻址、`delta_timestamps` & `*_is_pad`，以及高清双摄 `svla_so100_pickplace` 在 RTX 5080 上的 CPU (`libdav1d`) vs NVDEC (`av1_cuvid`) 与多 Worker 实测 |
| [`docs/RLDS_DEEP_DIVE.md`](docs/RLDS_DEEP_DIVE.md) | **RLDS / TFRecord**：全分片 Masked CRC32C 审计、Protobuf Wire 4 路无损 PNG 拆解、`features.json` 重塑机制、`89.1x` 读放大与 Shuffle Buffer 实测 |
| [`docs/HDF5_DEEP_DIVE.md`](docs/HDF5_DEEP_DIVE.md) | **HDF5 / robomimic**：Superblock v0、6,008 个 Dataset 导致的 `68.47%` B-Tree/MuJoCo XML 开销账本、微秒级连续切片实测 |
| [`docs/ZARR_DEEP_DIVE.md`](docs/ZARR_DEEP_DIVE.md) | **Zarr / Diffusion Policy**：全量 160 个 Blosc 16B 二进制头扫描（`95.63x` 压缩比）、`chunks[0]=161` 导致的 `161x` 读放大与无锁多进程扩展实测 |
| [`docs/MCAP_DEEP_DIVE.md`](docs/MCAP_DEEP_DIVE.md) | **ROS 2 bag / MCAP**：197 个 ChunkIndex、零 ROS 2 依赖纯 Python `CDR_LE` 相机/关节解码、真机三路相机掉帧抖动与跨通道对齐偏差、`metadata.yaml` + `robot.urdf` |
| [`docs/DEBUGGING_GUIDE.md`](docs/DEBUGGING_GUIDE.md) | **5 种格式 VS Code 单步调试与字节对照指南**：每个格式的 `F5` 入口、精确断点与 Watch 观测变量、Hex Editor 磁盘字节偏移对照要点 |

---

## 仓库目录结构（分层内聚）

```text
src/embodied_formats/
├── common.py                 # 统一只读契约 (BaseFormatReader / StepSample / WindowSample)
├── cli.py                    # 统一命令行入口 (edlab)
├── formats/                  # 只读统一 Reader 层（5 种格式各 1 文件）
│   ├── lerobot_v3_format.py
│   ├── rlds_format.py
│   ├── hdf5_format.py
│   ├── zarr_format.py
│   └── mcap_format.py
├── deep_dive/                # 源码级 + 字节级深度研究层（可直接 F5 单步调试）
│   ├── lerobot_v3.py         # LeRobot v3.0 五大子系统 + MP4 stbl 字节寻址
│   ├── lerobot_v3_hd.py      # 高清双摄 SO-100: CPU libdav1d vs RTX NVDEC + Multi-Worker
│   ├── rlds.py               # TFRecord 全分片 CRC32C + Protobuf + Shuffle Buffer 退化实测
│   ├── hdf5.py               # Superblock + B-Tree 账本 + 内嵌 MuJoCo XML + 微秒切片
│   ├── zarr.py               # 全量 Blosc Header + 161x 读放大 + 无锁多进程扩展
│   └── mcap.py               # ChunkIndex + 纯 Python ROS 2 CDR 解码 + 真机时钟抖动/对齐偏差
└── lab/                      # 横向通用工具（三层解剖 / 压测引擎 / 样本下载）
    ├── anatomy.py
    ├── benchmark.py
    ├── real_samples.py
    └── samples.py
```

---

## 快速开始

### 1. 环境初始化（Python 3.12 via `uv`）

```powershell
uv sync --python 3.12
uv run pytest -v
```

### 2. 下载真实开源数据集（国内网络建议配置镜像）

```powershell
# PowerShell 设置国内镜像并关闭 hf-xet 防大视频分片卡死（支持断点续传）
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"

# 拉取 5 份真实开源样本（~250 MB）
uv run edlab fetch-real

# 运行高清双摄真机 SO-100 实战（首次运行自动拉取 lerobot/svla_so100_pickplace, ~448 MB）
uv run edlab lerobot-hd -n 80
```

### 3. CLI 解剖与深度研究命令

```powershell
# 5 种格式各自深度研究（统一入口）
uv run edlab deep lerobot_v3 -n 100
uv run edlab deep rlds -n 25
uv run edlab deep hdf5 -n 200
uv run edlab deep zarr -n 60
uv run edlab deep mcap

# 高清双摄真机 SO-100 (CPU vs RTX NVDEC + 延迟拆解 + 1/2/4/8 Worker)
uv run edlab lerobot-hd -n 80

# 三层横向解剖 & HuggingFace Hub 缓存解剖
uv run edlab anatomy lerobot_v3
uv run edlab hf-cache
```

---

## VS Code 源码调试与字节对照指南

项目已内置 `.vscode/settings.json`、`.vscode/launch.json` 与 `.vscode/extensions.json`（推荐安装 `Python`、`debugpy`、`Pylance` 与 **`Hex Editor`**）。按 `F5` 可直接单步调试任意格式的深度研究模块：

| `F5` 调试项 | 对应源码与推荐断点 |
| --- | --- |
| **`1. Deep Dive: LeRobot v3.0`** | `deep_dive/lerobot_v3.py`：`parse_mp4_sample_table()`、`resolve_sample()`、`audit_v3_integrity()` |
| **`1b. Deep Dive: LeRobot v3.0 HD SO-100`** | `deep_dive/lerobot_v3_hd.py`：`benchmark_single_video_decoder()`、`profile_step_latency_breakdown()` |
| **`2. Deep Dive: RLDS / TFRecord`** | `deep_dive/rlds.py`：`audit_tfrecord_shard()`、`profile_random_read_degradation()`；`formats/rlds_format.py`：`decode_tf_example()` |
| **`3. Deep Dive: HDF5 / robomimic`** | `deep_dive/hdf5.py`：`parse_hdf5_superblock()`、`audit_hdf5_storage_and_metadata()`、`profile_hdf5_slicing()` |
| **`4. Deep Dive: Zarr / Diffusion Policy`** | `deep_dive/zarr.py`：`scan_blosc_chunks()`、`analyze_chunk_alignment_and_read_amp()` |
| **`5. Deep Dive: ROS 2 bag / MCAP`** | `deep_dive/mcap.py`：`CDRReader`、`decode_ros2_compressed_image_cdr()`、`decode_ros2_joint_state_cdr()`、`analyze_async_clocks_and_alignment()` |
| **`9. Current Python File (Step into Libs)`** | `justMyCode: false`，支持单步钻入 `pyarrow / av / h5py / zarr / mcap` 源码 |
