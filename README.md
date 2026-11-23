# Embodied AI Dataset Formats Anatomy & Benchmark Lab

面向系统程序员的具身智能（Embodied AI / VLA）主流数据集格式**底层解剖、统一只读访问与 I/O 压测实验项目**（不做跨格式转换，聚焦“读透、看透、测透”）。

覆盖工业界与开源社区 5 种最具代表性的格式：
1. **MCAP**（采集与日志层 — ROS 2 / Foxglove / Rerun）
2. **HDF5**（单机经典训练层 — robomimic / ALOHA / ACT）
3. **Zarr**（单机高并发训练层 — Diffusion Policy ReplayBuffer）
4. **RLDS / TFRecord**（传统大规模 VLA 生态 — Open X-Embodiment / OpenVLA，零 TensorFlow 纯二进制 CRC32C + Protobuf 解析）
5. **LeRobot v3.0**（当前 VLA 开源主流 — HuggingFace / OpenPI / SmolVLA / GR00T，直接用 PyArrow + PyAV 剖析分片全局行号映射与视频真实 GOP）

## 快速开始

```powershell
# 1. 安装依赖（Python 3.12 via uv）
uv sync --python 3.12

# 2. 准备样本：自动从 HuggingFace 拉取真实开源 LeRobot v3.0 (lerobot/pusht, ~35MB)，
#    并为 MCAP / HDF5 / Zarr / RLDS 准备好严格符合原生规范的本地样本
uv run edlab setup-samples

# 3. 逐个解剖底层存储结构、Schema、压缩与索引机制
uv run edlab inspect lerobot_v3
uv run edlab inspect mcap
uv run edlab inspect hdf5
uv run edlab inspect zarr
uv run edlab inspect rlds

# 4. 追踪单步读取 & 滑窗采样（Action Chunking）的底层索引路径
uv run edlab sample lerobot_v3 --ep 0 --step 10 --window
uv run edlab sample mcap --ep 0 --step 10 --window

# 5. 五种格式横向压测对比（顺序读 / 随机单步 / 随机滑窗）
uv run edlab compare -n 60

# 6. 拉取 5 份真实开源样本并做三层深度解剖（文件系统 / 二进制结构 / 非二进制元数据关系）
uv run edlab fetch-real
uv run edlab anatomy lerobot_v3
uv run edlab anatomy zarr
uv run edlab anatomy hdf5
uv run edlab anatomy rlds
uv run edlab anatomy mcap

# 7. 解剖本机 ~/.cache/huggingface/hub/ 类 Git 内容寻址缓存与符号链接/硬链接关系
uv run edlab hf-cache
```

> 任何命令均支持 `--path <你的真实数据集路径>` 直接解剖你本地已有的真机 MCAP、ALOHA HDF5、Diffusion Policy Zarr、Bridge RLDS 或 LeRobot v3.0 数据集。

## 推荐阅读代码顺序（吃透每种格式的核心机制）

| 顺序 | 代码文件 | 重点理解的底层机制 |
| --- | --- | --- |
| 1 | `src/embodied_formats/common.py` | 统一只读契约：`StepSample` vs `WindowSample`（`obs_horizon` + `action_horizon`） |
| 2 | `src/embodied_formats/formats/mcap_format.py` | 异步多通道纳秒时间戳、无 Episode 边界、ChunkIndex、主相机时钟对齐与抖动测量 |
| 3 | `src/embodied_formats/formats/hdf5_format.py` | B-Tree 树状层次、Contiguous vs Chunked 压缩比对比、单文件切片 |
| 4 | `src/embodied_formats/formats/zarr_format.py` | 分块目录树、Blosc(LZ4/BitShuffle)、`meta/episode_ends` 扁平零拷贝索引 |
| 5 | `src/embodied_formats/formats/rlds_format.py` | TFRecord 二进制帧格式、Masked CRC32C、Protobuf Wire 格式、`Episode→Steps` 整轨反序列化代价 |
| 6 | `src/embodied_formats/formats/lerobot_v3_format.py` | v3.0 共享分片、**全局行号→局部行号映射**、PyAV 拆包验证真实 `max_gop` 与随机 Seek 解码代价 |
| 7 | `src/embodied_formats/benchmark.py` | 3 种典型训练 I/O 模式（Sequential / Random Step / Random Window）的延迟分位统计 |
