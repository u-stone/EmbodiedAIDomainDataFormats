# 5 种主流具身智能数据集真实开源样本三层深度解剖报告

> 本文所有字节偏移、Magic Number、Box 树、CRC32C 校验值、压缩比与文件计数，均来自本项目 `uv run edlab anatomy <format>` 与 `uv run edlab hf-cache` 对 **5 份真实开源生产数据集** 的实测解析结果（零合成、零转换）。

## 0. 实测真实样本清单

| 格式 | 真实开源数据源 | 本机路径 | 磁盘体积 | 文件数 | `<64KB` 小文件占比 |
| --- | --- | --- | ---: | ---: | ---: |
| **LeRobot v3.0** | HuggingFace `lerobot/pusht` (206 eps, 25,650 帧) | `data/real/lerobot_v3_pusht` | 7.33 MB | 15 | 12/15 (80.0%) |
| **Zarr v2** | Columbia Diffusion Policy `pusht_cchi_v7_replay.zarr` (**同源 206 eps, 25,650 帧**) | `data/real/zarr_pusht/pusht_cchi_v7_replay.zarr` | 29.57 MB | **819** | **660/819 (80.6%)** |
| **HDF5** | Stanford robomimic `v1.5/lift/ph/demo_v15.hdf5` | `data/real/hdf5_robomimic/demo_v15.hdf5` | 35.00 MB | **1** | 0/1 (0.0%) |
| **RLDS / TFRecord** | Open X-Embodiment GCS `berkeley_cable_routing` (shard 4/64) | `data/real/rlds_berkeley_cable_routing` | 31.84 MB | 3 | 2/3 (66.7%) |
| **MCAP (ROS 2)** | RobotisAI `evButtonPush-260608-0-MCAP` (ep 0) | `data/real/mcap_robotis_button_push` | 154.25 MB | 4 | 3/4 (75.0%) |

---

## 1. 文件系统层专题：HuggingFace Hub 类 Git 内容寻址缓存与链接机制

当通过 `huggingface_hub` 拉取 LeRobot / HDF5 / MCAP 等数据集时，存在**两套并行的文件系统维护机制**：

### 1.1 全局缓存 `~/.cache/huggingface/hub/`（类 Git Content-Addressed Store）

```text
~/.cache/huggingface/hub/
└── datasets--<org>--<repo>/
    ├── refs/
    │   └── main                                  # 纯文本：40 位 Git Commit SHA
    ├── blobs/
    │   └── <sha256_or_git_oid>                   # 不可变内容寻址大对象（跨 Commit 去重）
    └── snapshots/
        └── <commit_sha>/
            └── <repo 内部真实路径>              # Linux/macOS: symlink -> ../../blobs/<hash>
                                                  # Windows (未开开发者模式): 降级为独立拷贝
```

- **本机实测验证**（`uv run edlab hf-cache`）：
  - `datasets--robomimic--robomimic_datasets/refs/main` = `74fa018461f479cd9fd15b924a16103012096203`
  - `datasets--RobotisAI--evButtonPush-260608-0-MCAP/refs/main` = `bff78f9fc0dcffcbdc3905c99c719d016e28e9a8`
  - **Windows NTFS 关键差异**：Linux/macOS（或开启了 Developer Mode 的 Windows）下，`snapshots/<sha>/...` 里全是符号链接指向 `blobs/<hash>`，多个 Commit 共享同一份未修改的大文件实现零空间冗余；当前 Windows 未开启 Developer Mode，`huggingface_hub` 自动降级为 copy 模式（实测 `is_symlink=false, st_nlink=1, blob_count=0`），直接把实体文件落盘在 `snapshots/<sha>/` 下。

### 1.2 `local_dir` 模式下的隐藏控制目录 `.cache/huggingface/`

对 `data/real/lerobot_v3_pusht` 做目录树解剖可以看到：

```text
lerobot_v3_pusht/
├── .cache/huggingface/
│   ├── trees/7628202a2180972f291ba1bc6723834921e72c19.json  (1.70 KB, 整个 Commit 目录树快照)
│   ├── download/
│   │   ├── data/chunk-000/file-000.parquet.metadata         (128 B)
│   │   ├── meta/info.json.metadata                          (104 B)
│   │   └── videos/observation.image/chunk-000/file-000.mp4.metadata (128 B)
│   └── CACHEDIR.TAG                                         (195 B, 标准备份排除标记)
├── data/chunk-000/file-000.parquet                          (658.59 KB)
├── meta/
│   ├── episodes/chunk-000/file-000.parquet                  (104.09 KB)
│   ├── info.json                                            (2.21 KB)
│   ├── stats.json                                           (4.27 KB)
│   └── tasks.parquet                                        (2.21 KB)
└── videos/observation.image/chunk-000/file-000.mp4          (6.57 MB)
```

观察 `.metadata` 文件内容（三行纯文本：`commit_hash` / `etag` / `timestamp`），能直接看出 **Git 普通文件 vs LFS/Xet 大文件** 的底层区别：
- `meta/info.json.metadata` 的第 2 行是 **40 位 Git Blob SHA-1**（`de3c71d108d220f5d500416542bb5b8a49574eaa`）
- `data/chunk-000/file-000.parquet.metadata` 的第 2 行是 **64 位 SHA-256**（`9abc0431...bbbf01`），说明大文件走的是 LFS / Xet 存储层。

---

## 2. 同源数据苹果对苹果对比：LeRobot v3.0 vs Diffusion Policy Zarr（PushT 25,650 帧）

这两份真实数据集来自**同一份 PushT 轨迹集合（206 条轨迹、25,650 帧、96×96）**，对比极其直观：

### 2.1 LeRobot v3.0 (`lerobot/pusht`)

- **文件系统**：仅 **15 个文件（核心数据仅 6 个文件），总计 7.33 MB**。MP4 占 `89.69%`（6.57 MB），Parquet 占 `10.19%`（764.88 KB），JSON 仅占 `0.11%`（8.18 KB）。
- **Parquet 二进制结构（`data/chunk-000/file-000.parquet`, 658.59 KB）**：
  - 首尾 4 字节 Magic 均为 `PAR1`；文件倒数 8–4 字节 `uint32_le` 指出 Thrift Footer 长度 = **27,537 字节**。
  - 25,650 行切分为 **26 个 RowGroup**（每组约 1,000 行），全部列采用 `SNAPPY` + `RLE_DICTIONARY` 编码。
- **MP4 (ISO BMFF) 二进制 Box 树（`videos/observation.image/chunk-000/file-000.mp4`, 6.57 MB）**：
  - 文件头 24 字节：`00 00 00 20 66 74 79 70 69 73 6F 6D ... 69 73 6F 6D 61 76 30 31`（`ftyp isom av01`，AV1 视频）。
  - 顶层 Box 布局：`ftyp` @ `0` (32 B) → `free` @ `32` (8 B) → **`mdat` @ `40` (6.42 MB)** → **`moov` @ `6,735,984` (151.35 KB)**。
  - 深入 `moov -> trak -> mdia -> minf -> stbl`（Sample Table，随机 Seek 的核心）：
    - `stsz` (100.21 KB)：记录全部 **25,650 帧**各自的压缩后字节长度。
    - `stss` (50.33 KB，Sync Sample Box)：记录全部 **12,880 个 Keyframe** 的帧号！`25650 / 12880 ≈ 1.99`。
    - PyAV 逐包验证：`codec=libdav1d (AV1)`、`pix_fmt=yuv420p`、`min_gop=1, mean_gop=1.99, max_gop=2` — 彻底印证了 LeRobot 为随机抽帧强制 `GOP=2` 的设计。
- **非二进制控制面寻址链**：
  `meta/info.json`（路径模板 + Schema） → `meta/episodes/chunk-000/file-000.parquet`（查 `episode_index` 对应的全局行区间 `dataset_from_index..to_index` + 视频时间槽 `from_timestamp..to_timestamp`，如 ep0=`[0, 161)` / `[0.0s, 16.1s]`，ep1=`[161, 279)` / `[16.1s, 27.9s]`） → 分别定位 Parquet 局部行 & MP4 `stss` 关键帧 Seek 解码 + `meta/tasks.parquet` 查任务文本 + `meta/stats.json` 做归一化。

### 2.2 Diffusion Policy Zarr (`pusht_cchi_v7_replay.zarr`)

- **文件系统**：同样 25,650 帧，磁盘膨胀为 **819 个文件、29.57 MB**（体积 **4x**，文件数 **54x**，`<64KB` 小文件高达 **660 个 / 80.6%**！）。
- **Blosc Chunk 二进制结构（`data/img/0.0.0.0`, 170.83 KB）**：
  - 前 16 字节原始 Hex：`02 01 94 04 00 B0 0F 01 00 00 04 00 4D AB 02 00`
  - 逐字段解码：`version=2`, `flags=0x94`（**`bit_shuffle=True`**）, `typesize=4`（`<f4` float32 存图像像素！）, `nbytes=17,805,312`（16.98 MB 未压缩 = `161 × 96 × 96 × 3 × 4`）, `blocksize=262,144`（256 KB 内部并行块）, `cbytes=174,925`（170.83 KB），单块压缩比 **101.79x**（2D 渲染图 + Zstd + BitShuffle）。
- **非二进制控制面寻址链**：
  - 根目录 `.zgroup` → `data/.zgroup` & `meta/.zgroup` → 6 个 `.zarray` JSON（声明 `shape=[25650, ...]`, `chunks=[161, ...]`, `compressor=blosc/zstd/level5/bitshuffle`, `order="C"`）。
  - 注意细节：`chunks[0]=161` 恰好等于 `meta/episode_ends[0]=161`（创建时按第 0 条轨迹长度定 chunk）。读 `(ep, step)` 时，先查 `meta/episode_ends` 得到全局行号 `g`，再做 `g // 161` 映射到独立磁盘文件 `data/img/<chunk_id>.0.0.0`，无锁并发读取解压。

---

## 3. HDF5（Stanford robomimic `demo_v15.hdf5`）

- **文件系统**：极致单文件封装（**1 个文件 / 35.00 MB**，外部零文本元数据文件）。
- **二进制结构**：
  - 文件头 32 字节：`89 48 44 46 0D 0A 1A 0A 00 00 00 00 00 08 08 00 ...`
  - 8 字节 Magic `\x89HDF\r\n\x1a\n`，**Superblock v0**，8-byte offset / 8-byte length。
  - 所有 Dataset（`actions (59,7) float64`, `states (59,32) float64`, `controller_info/*`, `user_info/*`）全部采用 **`contiguous` + `compression=none`**（1.00x 无压缩），定位后直接按基址+步长做零解压切片。
- **非二进制元数据组织**：
  - **全部自包含在文件内部的 HDF5 Group Attributes** 中：
    - `/data.env_args`：内嵌完整 JSON 字符串（环境名 `Lift`、控制频率 `20Hz`、控制器配置）。
    - `/data/demo_0.model_file`：内嵌完整的 **MuJoCo XML 物理仿真模型**（`<?xml version='1.0'?><mujoco model="base">...`）用于精确复现场景状态！

---

## 4. RLDS / TFRecord（Open X-Embodiment GCS `berkeley_cable_routing`）

- **文件系统**：**3 个文件 / 31.84 MB**（`dataset_info.json` 2.01 KB + `features.json` 10.92 KB + 单个分片 `berkeley_cable_routing-train.tfrecord-00004-of-00064` 31.83 MB）。
- **二进制结构**：
  - 文件头 12 字节：`89 E0 20 00 00 00 00 00 | 89 F3 DD 25`
    - `uint64_le length` = `2,154,633 B`（2.05 MB，整条 Episode 0）
    - `uint32_le length_masked_crc32c` = `0x25DDF389`（实测 Castagnoli CRC32C 校验通过）
    - 尾部 `payload_masked_crc32c` = `0x32FC3171`（实测校验通过）
  - 紧随其后是 Protobuf Wire 字节流 `0A ... 73 74 65 70 73 2F 6F 62 73`（即 `steps/obs...`）。
  - 单个 31.83 MB 分片里仅存放 **12 条 Episode**（1.57–5.62 MB/条），原因是 4 路相机（`image` / `top_image` / `wrist225_image` / `wrist45_image`）全部以**无损 PNG (`89 50`)** 逐帧打包在 `BytesList` 中，图像占整条记录体积的 **98.19%**！
- **非二进制元数据组织**：
  - `dataset_info.json` 声明 `fileFormat: "tfrecord"` 及 64 个 train 分片各自的 Episode 数（`shardLengths[4] == "12"`，与二进制扫描出的 12 条严丝合缝）。
  - `features.json` 声明嵌套 `FeaturesDict -> Sequence(steps)` 的逻辑 Tensor 形状与数据类型；因为 TFRecord 内的 Protobuf 只存扁平一维 `FloatList`（如 `steps/observation/natural_language_embedding` 存 9,216 个 float = `18 steps × 512 dim`），**没有 `features.json` 就无法准确 reshape**。

---

## 5. MCAP（RobotisAI 真机 ROS 2 `evButtonPush-260608-0-MCAP`）

- **文件系统**：**4 个文件 / 154.25 MB**（`0_0.mcap` 154.23 MB + `metadata.yaml` 1.95 KB + `episode_info.json` 265 B + `robot.urdf` 15.41 KB）。
- **二进制结构**：
  - 首尾对称 8 字节 Magic：`89 4D 43 41 50 30 0D 0A`（`\x89MCAP0\r\n`）。
  - 倒数 37 字节是 **Footer Record**（`opcode=0x02`, `len=20`）：`summary_start = 161,685,840`（154.20 MB 处），`summary_crc32 = 0x671C73C1`。
  - 整个 **Summary Section 仅 34.45 KB（0.02%）**，却一次性索引了全部 **197 个 Chunk**（本文件 Chunk 压缩设为 `none`，因为三路相机已经是 `sensor_msgs/msg/CompressedImage` + `cdr` 编码，避免对 JPEG 重复压缩浪费机载 CPU）。
  - **真机多传感器异构异步频率实测**（27.16 秒录制）：
    - 3 路相机（`cam_wrist` / `cam_top` / `cam_belly`）：`29.97 Hz` (814 帧) / `28.54 Hz` (775 帧) / `28.09 Hz` (763 帧) — **同一次录制三路相机帧数完全不同**。
    - 本体关节 `/joint_states` & 遥操主臂 `/leader/joint_states`：**`193.31 Hz`** (5,250 条) / **`193.93 Hz`** (5,267 条)。
    - 坐标变换 `/tf`：**`38.85 Hz`** (1,055 条)。
- **非二进制 Sidecar 元数据组织**：
  - `metadata.yaml`：ROS 2 `rosbag2_bagfile_information` v9（录制时长纳秒、起始纪元时间戳、话题与 QoS 元信息）。
  - `episode_info.json`：业务层任务描述（`"task_instruction": "push the elevator call button"`、机器人型号 `frontier_omy_f3m`、设备序列号）。
  - `robot.urdf`（15.41 KB XML）：完整机器人运动学连杆/关节树，配合 `/joint_states` 和 `/tf` 才能做正运动学（FK）和 3D 可视化回放。
