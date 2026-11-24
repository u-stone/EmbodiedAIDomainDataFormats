# RLDS / HDF5 / Zarr / MCAP 四大主流格式底层深度研究报告

> 本文所有实测数据均来自本仓库 `src/embodied_formats/deep_dive/{rlds,hdf5,zarr,mcap}.py` 对 **4 份真实开源生产数据集** 的字节级解析与压测结果（可运行 `uv run edlab deep <format>` 或在 VS Code 中按 `F5` 选择 `2–5` 号调试项单步复现）。

---

## 1. RLDS / TFRecord 深度剖析（Open X-Embodiment `berkeley_cable_routing`）

**实测对象**：`data/real/rlds_berkeley_cable_routing`（`dataset_info.json` + `features.json` + `berkeley_cable_routing-train.tfrecord-00004-of-00064`，31.83 MB）  
**源码入口**：[`src/embodied_formats/deep_dive/rlds.py`](../src/embodied_formats/deep_dive/rlds.py)（VS Code `F5` → `2. Deep Dive: RLDS / TFRecord`）

### 1.1 全分片二进制帧与 Castagnoli Masked CRC32C 审计
- TFRecord 每条记录物理帧为 `[uint64_le len][uint32_le masked_crc32c(len)][payload][uint32_le masked_crc32c(payload)]`，**内部没有任何随机访问偏移表**。
- 全分片共 **12 条 Episode 记录**（与 `dataset_info.json` 中 `splits[train].shardLengths[4] == 12` 严丝合缝），全部 12 条的 **长度 CRC32C 与载荷 CRC32C 均 100% 校验通过**。
- 单条 Episode 体积：`min = 1.57 MB` / `median = 2.41 MB` / `max = 5.62 MB`（如 Record 0 位于偏移 `0`，载荷 `2,154,633 B`；Record 1 紧接于 `2,154,649 = 12 + 2,154,633 + 4`）。

### 1.2 Protobuf Wire 载荷拆解与 `features.json` 重塑机制
- 对 Episode 0（18 步，`2.05 MB`）的 Protobuf 字段做字节占比统计：
  - **4 路相机图像**：占 **`98.19%`**！手解原始 PNG IHDR 头确认全部为 **128×128 8-bit RGB 无损 PNG**（`image` / `top_image` / `wrist225_image` / `wrist45_image`，单帧 `24–32 KB`）。
  - **预计算 512 维语言 Embedding**：占 **`1.71%`**（`36 KB`，9,216 个 float = `18 × 512`）。
  - **机器人状态与动作**：仅占 **`0.05%`**（`robot_state (18, 7)` + `world_vector (18, 3)` + `rotation_delta (18, 3)`）。
- **核心设计机制**：Protobuf `tf.train.Example` 仅存扁平化一维 `FloatList/BytesList`，**必须依赖外部 `features.json`** 声明的嵌套 `FeaturesDict → Sequence(steps)` 逻辑 Shape，才能把 `flat_pb_len=9216` 正确 reshape 回 `(18, 512)`。

### 1.3 随机读取退化实测（为何 OpenVLA / TFDS 必须用 Shuffle Buffer）

| 访问方式（`steps/observation/top_image`） | 读放大倍数 | 吞吐 (samples/s) | P50 延迟 | P99 延迟 |
| --- | ---: | ---: | ---: | ---: |
| **无缓存跨轨迹随机抽 1 帧**（读整条 2–5 MB Episode + 全量 Protobuf 反序列化） | **`89.1x`** | `348.7` | `2.23 ms` | **`10.24 ms`** |
| **内存 Episode / Shuffle Buffer**（顺序流式预读后在内存打乱） | `1.0x` | **`2,929.6`** | `0.34 ms` | **`0.36 ms`** |

> **结论**：因为 RLDS 将整条轨迹打包进单个 Protobuf 记录，磁盘级随机单帧读取会产生高达 **89x 的读放大**（延迟暴涨 **8.4 倍**）；工业界训练只能顺序流式扫描分片 + 在 RAM 中维护大型 `shuffle_buffer`（或迁移至带随机偏移索引的 `ArrayRecord` / `LeRobot v3.0`）。

---

## 2. HDF5 / robomimic 深度剖析（Stanford `v1.5/lift/ph/demo_v15.hdf5`）

**实测对象**：`data/real/hdf5_robomimic/demo_v15.hdf5`（35.00 MB 单文件，200 条 Panda Lift 演示，9,666 步）  
**源码入口**：[`src/embodied_formats/deep_dive/hdf5.py`](../src/embodied_formats/deep_dive/hdf5.py)（VS Code `F5` → `3. Deep Dive: HDF5 / robomimic`）

### 2.1 Superblock 二进制头与惊人的 B-Tree 元数据开销
- **Superblock v0**（偏移 `0`）：Magic `\x89HDF\r\n\x1a\n` 校验通过，`size_of_offsets = 8`，`size_of_lengths = 8`，`group_leaf_node_k = 4`，`group_internal_node_k = 16`，`eof_address = 36,699,586 == 文件实际字节数`。
- **存储账本（关键发现）**：
  - 全文件共有 **6,008 个独立 Dataset**（200 条 Demo × 每条 ~30 个数组），**100% 为 `contiguous` 无压缩**。
  - 所有 Dataset 的真实张量载荷总和仅 **`11.04 MB`**！
  - 而 **HDF5 内部 B-Tree 节点 + 200 份重复内嵌的 MuJoCo XML (`37.55 KB × 200 ≈ 7.5 MB`) + Heap/Attributes 占了 `23.96 MB`（占整个文件 `68.47%`！）**。

### 2.2 自包含非二进制元数据解剖
HDF5 外部零文本文件，所有控制面信息全部内嵌在内部 Group/Dataset Attributes 与 `/mask` 分组中：
- `/data.attrs["env_args"]`：完整 JSON（`env_name: "Lift"`, `robots: ["Panda"]`, `control_freq: 20`）。
- `/data/demo_0.attrs["model_file"]`：完整 **37.55 KB MuJoCo XML**（23 个 `<body>`、10 个 `<joint>`、86 个 `<geom>`、9 个 `<actuator>`），用于精准回放仿真物理状态。
- `/mask/*`：内嵌官方训练/验证集划分掩码（`train=180`, `valid=20`, `20_percent=40`, `50_percent=100`）。

### 2.3 Contiguous 零解压切片微秒级实测
因为所有 Dataset 均采用 `contiguous` 布局（如 `demo_0/actions` 直接从物理偏移 `20,288` 连续存放），切片只需 B-Tree 查基址 + 指针偏移：
- **顺序读取**：`10,628.7 sps`（`P50 = 85.1 µs`）
- **跨 Demo 随机单步**：`7,933.9 sps`（`P50 = 97.7 µs`）
- **跨 Demo 随机滑窗 (`action_horizon=16`)**：`9,584.6 sps`（`P50 = 96.1 µs`）

---

## 3. Zarr / Diffusion Policy 深度剖析（Columbia `pusht_cchi_v7_replay.zarr`）

**实测对象**：`data/real/zarr_pusht/pusht_cchi_v7_replay.zarr`（206 条轨迹、25,650 帧、819 个文件、29.57 MB）  
**源码入口**：[`src/embodied_formats/deep_dive/zarr.py`](../src/embodied_formats/deep_dive/zarr.py)（VS Code `F5` → `4. Deep Dive: Zarr / Diffusion Policy`）

### 3.1 全量 160 个 Blosc Chunk 二进制头扫描
- `data/img` 原始形状为 `(25650, 96, 96, 3)` 且使用 **`float32` (`typesize=4`)** 存储，未压缩体积高达 **`2.65 GB`**！
- 全量 160 个分块均采用 `Blosc(zstd, flags=0x94 [bitshuffle=True], blocksize=256 KB)`，压成 **`28.41 MB`**（总压缩比 **`95.63x`**，单块范围 `85.19x – 275.32x`）。

### 3.2 Chunk 边界错位与 `161x` 读放大陷阱
- 创建者直接用了第 0 条轨迹的长度 **`161`** 作为 `chunks[0]`（`chunks = [161, 96, 96, 3]`），导致：
  1. **151 / 206 条轨迹（`73.3%`）跨越了物理 Chunk 边界**；
  2. 单帧未压缩 `108 KB`，但单个 Chunk 未压缩高达 **`16.98 MB`**——每次随机抽 1 帧都必须把整块 161 帧全部解压，**读放大高达 `161x`**！
- **性能实测印证**：
  - 单帧随机读：`363.4 sps`（`P50 = 2.74 ms`）——在同等 96×96 PushT 数据上，比 LeRobot v3.0 的 AV1 视频随机 Seek（`1,613 sps` / `P50 = 0.57 ms`）慢了 **4.4 倍**！
  - **无锁多进程扩展**：`1 worker = 254.4 sps` → `4 workers = 382.4 sps`（无全局锁竞争，但受限于单块 16.98 MB 的 Blosc 解压内存带宽；若将 `chunks[0]` 调小至 `1` 或 `8`，随机读会大幅提速，代价是文件数暴增与压缩率下降）。

---

## 4. ROS 2 bag / MCAP 深度剖析（RobotisAI 真机 `evButtonPush-260608-0-MCAP`）

**实测对象**：`data/real/mcap_robotis_button_push`（`0_0.mcap` 154.23 MB + `metadata.yaml` + `episode_info.json` + `robot.urdf`）  
**源码入口**：[`src/embodied_formats/deep_dive/mcap.py`](../src/embodied_formats/deep_dive/mcap.py)（VS Code `F5` → `5. Deep Dive: ROS 2 bag / MCAP`）

### 4.1 197 个 ChunkIndex 与零 ROS 2 依赖的原始 CDR 二进制解码
- 全部 **197 个 Chunk** 未做二次压缩（`none`），中位数大小 `801.44 KB`、时间跨度中位数 `136.47 ms`。
- 通过手写 `CDRReader` 直接解析 ROS 2 原始 CDR 小端封装头（`00 01 00 00` = `CDR_LE`）：
  - **3 路相机（`sensor_msgs/msg/CompressedImage`）**：解出内嵌 JPEG `FF D8 FF E0`，全部为 **`640×480 RGB`**（单帧 52–77 KB）；同时暴露真实工程细节：`format = "rgb8; jpeg compressed bgr8"`（需严格核对通道顺序）。
  - **双路关节状态（`sensor_msgs/msg/JointState`）**：
    - `/leader/joint_states`：7 维主手关节（`joint1..6 + rh_r1_joint`）精确 `float64` 弧度；
    - `/joint_states`：16 维整机状态（4 轮转向/驱动 + 颈/头 + `omy_joint1..6`）。

### 4.2 真机多传感器异步时钟抖动与跨通道对齐偏差实测（27.16 秒录制）

| Topic | 消息数 | 有效频率 | 帧间隔 `dt` 均值 / 标准差 | 最大帧间隔（掉帧尖峰） | 对齐主相机 `cam_top` 偏差（均值 / P95 / 最大） |
| --- | ---: | ---: | ---: | ---: | ---: |
| `/camera/cam_top/...`（主时钟） | 775 | `28.51 Hz` | `35.07 ± 15.59 ms` | **`123.39 ms`** | 基准 (`0 ms`) |
| `/camera/cam_belly/...` | 763 | `28.08 Hz` | `35.62 ± 17.57 ms` | **`147.42 ms`** | `11.61 / 32.82 / 70.07 ms` |
| `/camera/cam_wrist/...` | 814 | `30.00 Hz` | `33.34 ± 8.77 ms` | `95.76 ms` | `9.82 / 17.47 / 42.03 ms` |
| `/joint_states` | 5,250 | **`193.37 Hz`** | `5.17 ± 1.76 ms` | `32.91 ms` | `1.33 / 2.91 / 13.40 ms` |
| `/leader/joint_states` | 5,267 | **`194.00 Hz`** | `5.16 ± 1.62 ms` | `18.68 ms` | `1.48 / 2.89 / 11.76 ms` |
| `/tf` | 1,055 | `38.84 Hz` | `25.74 ± 14.89 ms` | `53.57 ms` | `8.94 / 21.27 / 25.01 ms` |

> **工程启示**：真机三路 USB/GMSL 相机不仅帧率抖动大（最大掉帧达 `147 ms`），相互之间也存在 `10–70 ms` 的异步相位差；这直接解释了为什么工业界必须将 MCAP 仅作为**真机采集与回放日志**，在离线同步清洗并做 CFR 重采样后再进入 VLA 训练。

### 4.3 Sidecar 非二进制元数据联合关系
- `metadata.yaml`（ROS 2 `rosbag2` v9，13,924 条消息、27.159s）提供快速 Bag 级索引；
- `episode_info.json` 补充业务任务语义（`"task_instruction": "push the elevator call button"`、`robot_type: "frontier_omy_f3m"`）；
- `robot.urdf`（17 个 Link、16 个 Joint = `6 fixed + 10 revolute`）与 `/joint_states` + `/tf` 结合还原完整 3D 运动学树。
