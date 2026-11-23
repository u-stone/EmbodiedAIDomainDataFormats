# Embodied AI Dataset Formats Anatomy & Benchmark Lab

面向系统程序员的具身智能（Embodied AI / VLA）主流数据集格式**底层解剖、统一只读访问与 I/O 压测实验项目**（不做跨格式转换，聚焦“文件系统 / 二进制结构 / 非二进制元数据组织 / I/O 与解码性能”）。

覆盖工业界与开源社区 5 种最具代表性的格式：
1. **MCAP**（采集与日志层 — ROS 2 / Foxglove / Rerun）
2. **HDF5**（单机经典训练层 — robomimic / ALOHA / ACT）
3. **Zarr**（单机高并发训练层 — Diffusion Policy ReplayBuffer）
4. **RLDS / TFRecord**（传统大规模 VLA 生态 — Open X-Embodiment / OpenVLA，零 TensorFlow 纯二进制 CRC32C + Protobuf 解析）
5. **LeRobot v3.0**（当前 VLA 开源主流 — HuggingFace / OpenPI / SmolVLA / GR00T，PyArrow + PyAV + 原生 ISO BMFF MP4 `stbl` 解析）

---

## 深度研究文档索引（已落档）

| 文档 | 内容摘要 |
| --- | --- |
| [`docs/REAL_DATASETS_ANATOMY.md`](docs/REAL_DATASETS_ANATOMY.md) | 5 份真实开源生产数据集（`lerobot/pusht`、Columbia Diffusion Policy `pusht.zarr`、Stanford robomimic `demo_v15.hdf5`、OXE GCS `berkeley_cable_routing`、RobotisAI ROS 2 `evButtonPush` MCAP）的三层实测解剖：文件系统 & `~/.cache/huggingface/hub/` 类 Git 内容寻址机制、二进制 Magic/Header/Box/CRC32C、非二进制元数据寻址链 |
| [`docs/LEROBOT_V3_DEEP_DIVE.md`](docs/LEROBOT_V3_DEEP_DIVE.md) | **LeRobot v3.0 深度专题**：v2.1→v3.0 架构演进、5 大数学不变式与 Chan/Welford 统计重构、Parquet RowGroup & Predicate Pushdown、MP4 `stbl` 精确字节寻址、`delta_timestamps` & `*_is_pad` 掩码，以及**高清双摄真机 `lerobot/svla_so100_pickplace` 在 RTX 5080 上的 CPU (`libdav1d`) vs NVDEC (`av1_cuvid`) 与多 Worker 吞吐实测** |

---

## 快速开始

### 1. 环境初始化（Python 3.12 via `uv`）

```powershell
uv sync --python 3.12
uv run pytest -v
```

### 2. 下载真实开源数据集（国内网络建议配置镜像）

> **下载避坑提示**：在国内网络下拉取大型 HuggingFace 视频/MCAP 分片时，建议同时开启镜像并关闭 `hf-xet` 以避免大文件传输卡死；若中途中断，重新执行会自动借助 `.cache/huggingface/download/*.metadata` 跳过已下载文件并断点续传。

```powershell
# PowerShell 设置国内镜像与防卡死参数
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"

# 拉取 5 份真实开源样本（~250 MB）
uv run edlab fetch-real

# 单独运行高清双摄真机 SO-100 实战（首次运行自动拉取 lerobot/svla_so100_pickplace, ~448 MB）
uv run edlab lerobot-hd -n 80
```

### 3. 常用 CLI 解剖与压测命令

```powershell
# 1) 三层深度解剖（文件系统 / 二进制结构 / 非二进制元数据组织）
uv run edlab anatomy lerobot_v3
uv run edlab anatomy zarr
uv run edlab anatomy hdf5
uv run edlab anatomy rlds
uv run edlab anatomy mcap

# 2) 解剖本机 ~/.cache/huggingface/hub/ 类 Git 内容寻址缓存（refs / blobs / snapshots / symlink）
uv run edlab hf-cache

# 3) LeRobot v3.0 五大子系统深度审计与 MP4 字节级 Seek 剖析
uv run edlab lerobot-deep -n 100
uv run edlab lerobot-deep --path data/real/lerobot_v3_so100_pickplace -n 60

# 4) 高清双摄真机实战（CPU libdav1d vs NVIDIA NVDEC + 延迟拆解 + 1/2/4/8 Worker 扩展性）
uv run edlab lerobot-hd -n 80

# 5) 轻量合成/混合样本 5 格式横向压测对比
uv run edlab setup-samples
uv run edlab compare -n 60
```

> 所有 `inspect` / `sample` / `bench` / `anatomy` / `lerobot-deep` / `lerobot-hd` 命令均支持 `--path <你的本地真实数据集路径>`。

---

## VS Code 源码调试与字节对照指南

项目已内置 `.vscode/settings.json`、`.vscode/launch.json` 与 `.vscode/extensions.json`（推荐安装 `Python`、`debugpy`、`Pylance` 与 **`Hex Editor`** 扩展，可直接右键 `data/real/` 下的 `.parquet` / `.mp4` / `.hdf5` / `.tfrecord` / `.mcap` 对照十六进制字节流）。

按 `F5` 可直接选择以下核心调试入口：
- **`1. Debug: LeRobot v3.0 Deep Study (lerobot-deep)`**
- **`1b. Debug: Dual-Cam HD SO-100 Lab (CPU vs RTX NVDEC + Workers)`**
- **`2. Debug: Deep 3-Layer Anatomy (Pick Format)`**（弹窗选择 5 种格式之一）
- **`7. Debug: Current Python File (Step into Libs)`**（`justMyCode: false`，支持单步钻入 `pyarrow / av / h5py / zarr / mcap`）

### 推荐打断点的关键代码位置

| 代码文件 | 关键函数 / 观察点 |
| --- | --- |
| `src/embodied_formats/lerobot_v3_deep.py` | `parse_mp4_sample_table()`（手解 MP4 `stts/stss/stsc/stsz/stco`）、`resolve_sample()`（Keyframe 二分与 `mdat` 字节偏移）、`audit_v3_integrity()`（Chan/Welford 统计重构） |
| `src/embodied_formats/lerobot_v3_hd_bench.py` | `benchmark_single_video_decoder()`（`libdav1d` vs `av1_cuvid`）、`profile_step_latency_breakdown()`（Parquet vs 双摄解码耗时拆解） |
| `src/embodied_formats/formats/lerobot_v3_format.py` | `read_step()` / `read_window()`（跨分片全局行号→局部行号映射与 PyAV Seek） |
| `src/embodied_formats/formats/rlds_format.py` | `masked_crc32c()`、`decode_tf_example()`（纯 Python TFRecord 帧与 Protobuf Wire 解码） |
| `src/embodied_formats/formats/mcap_format.py` | `read_step()`（多通道异步时间戳主相机对齐与偏差测量） |
| `src/embodied_formats/anatomy.py` | `analyze_filesystem()`、`inspect_hf_hub_cache()`（symlink/hardlink/inode 与 HF 缓存结构） |
