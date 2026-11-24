# 5 种具身智能数据集格式 VS Code 单步调试与字节对照实战指南

> 本文面向有系统编程背景的开发者，提供在 VS Code 中通过 **断点单步跟踪 + Watch 变量观测 + Hex Editor 原始字节对照** 吃透 5 种主流具身智能数据集格式的实操路线。
>
> **前置建议**：安装 `.vscode/extensions.json` 推荐的 `Hex Editor (ms-vscode.hexeditor)` 扩展，右键 `data/real/` 下任意二进制文件选择 **`Open With... -> Hex Editor`**，即可一边单步调试 Python 解析代码，一边在右侧对照磁盘原始十六进制字节。

---

## 1. LeRobot v3.0（Parquet + ISO BMFF MP4 + 关系型元数据）

- **VS Code `F5` 入口**：
  - `1. Deep Dive: LeRobot v3.0 (pusht)`（快速看清全套机制）
  - `1b. Deep Dive: LeRobot v3.0 HD Dual-Cam SO-100`（看高清双摄 CPU vs RTX NVDEC 瓶颈）
- **推荐断点与 Watch 观察点**：
  1. `src/embodied_formats/deep_dive/lerobot_v3.py` → `parse_mp4_sample_table()`
     - **Watch**：`btype`、`size`、`pos`、`keyframes_1based[:10]`、`sample_sizes[:10]`、`moov_offset`、`mdat_offset`
     - **看什么**：亲眼看 `ftyp → free → mdat → moov` 顶层 Box 递归进入 `stbl`，体会为什么 `moov` 在文件末尾会影响云端流式读取。
  2. `src/embodied_formats/deep_dive/lerobot_v3.py` → `MP4SampleTable.resolve_sample()`
     - **Watch**：`sample_idx_0based`、`kf_0based`、`offsets[sample_idx_0based]`、`decode_frames_needed`
     - **看什么**：验证如何仅靠 `stss` 二分查找 + `stsc/stco/stsz` 算出任意一帧的精确磁盘字节偏移，以及 `GOP=2` 下只需解码 1–2 帧。
  3. `src/embodied_formats/deep_dive/lerobot_v3.py` → `audit_v3_integrity()`（Chan/Welford 循环处）
     - **Watch**：`agg_mean`、`agg_m2`、`rec_std`、`ref_std`
     - **看什么**：验证多条 Episode 局部统计量如何通过并行方差公式无损还原全局 `stats.json`。
  4. `src/embodied_formats/formats/lerobot_v3_format.py` → `read_step()` / `read_window()`
     - **Watch**：`global_row`、`local_row`、`v_from`、`target_ts`
     - **看什么**：牢记 v3.0 最经典坑点——**全局行号必须减去分片起始行号才是 Parquet 局部索引**。
  5. `src/embodied_formats/deep_dive/lerobot_v3_hd.py` → `_benchmark_via_explicit_codec()`
     - **Watch**：`lat_ms`、`frames_per_seek`
     - **看什么**：对比 `ctx.flush_buffers()` 在 RTX NVDEC 上造成的 ~23 ms 硬件管线排空尖峰。
- **Hex Editor 字节对照**：
  - 打开 `data/real/lerobot_v3_pusht/data/chunk-000/file-000.parquet`：看开头 `0x00` 处的 `50 41 52 31 (PAR1)` 和文件末尾最后 8 字节（4B `footer_len` + `PAR1`）。
  - 打开 `data/real/lerobot_v3_pusht/videos/observation.image/chunk-000/file-000.mp4`：看 `0x00` 的 `ftyp isom av01`，跳转偏移 `6735984 (0x66C7B0)` 看到 `moov` 起始位置！

---

## 2. RLDS / TFRecord（无索引二进制帧 + Protobuf + `features.json`）

- **VS Code `F5` 入口**：`2. Deep Dive: RLDS / TFRecord (CRC32C + Protobuf + Read-Amp)`
- **推荐断点与 Watch 观察点**：
  1. `src/embodied_formats/deep_dive/rlds.py` → `audit_tfrecord_shard()`
     - **Watch**：`rec_start`、`length`、`hex(len_crc)`、`len_crc_ok`、`payload_crc_ok`
     - **看什么**：体会 TFRecord 没有任何目录/索引，必须逐条按 `12 + length + 4` 顺序跳帧扫描。
  2. `src/embodied_formats/formats/rlds_format.py` → `decode_tf_example()` 与 `_iter_pb_fields()`
     - **Watch**：`tag`、`field_num`、`wire_type`、`key`、`features_dict.keys()`
     - **看什么**：零 TensorFlow 依赖手解 Protobuf 线协议（varint + length-delimited + packed float32），观察为什么 `steps/observation/natural_language_embedding` 在二进制里只是扁平的 `9216` 个 float，必须靠 `features.json` 才能 reshape 成 `(18, 512)`。
  3. `src/embodied_formats/deep_dive/rlds.py` → `profile_random_read_degradation()`
     - **Watch**：`bytes_read_total / bytes_useful_total`（`89.1x` 读放大）、`nocache_ms` vs `cached_ms`
     - **看什么**：直观感受“为取 1 帧 32 KB PNG 却要反序列化整条 2–5 MB Episode Protobuf”的真实代价。
- **Hex Editor 字节对照**：
  - 打开 `data/real/rlds_berkeley_cable_routing/berkeley_cable_routing-train.tfrecord-00004-of-00064`：
    - `0x00..0x07`：`89 E0 20 00 00 00 00 00`（小端 `uint64` = `2,154,633`）
    - `0x08..0x0B`：`89 F3 DD 25`（小端 Masked CRC32C = `0x25DDF389`）
    - `0x17..`：直接看到明文 ASCII `steps/observation/...`，往后翻可看到无损 PNG 头 `89 50 4E 47 0D 0A 1A 0A`。

---

## 3. HDF5 / robomimic（Superblock + 内部 B-Tree + 自包含属性）

- **VS Code `F5` 入口**：`3. Deep Dive: HDF5 / robomimic (Superblock + B-Tree + MuJoCo XML)`
- **推荐断点与 Watch 观察点**：
  1. `src/embodied_formats/deep_dive/hdf5.py` → `parse_hdf5_superblock()`
     - **Watch**：`ver`、`offset_sz`、`length_sz`、`group_leaf_k`、`eof_addr`、`fsize`
     - **看什么**：看清 HDF5 单文件如何用 Superblock 管理整个内部虚拟文件系统。
  2. `src/embodied_formats/deep_dive/hdf5.py` → `audit_hdf5_storage_and_metadata()`
     - **Watch**：`total_datasets` (`6008`)、`total_stored` (`11.04 MB`)、`btree_overhead` (`23.96 MB / 68.47%`)、`ea`（`env_args` JSON）、`xml_str[:200]`
     - **看什么**：重点观察“为什么 11 MB 真实数据膨胀成 35 MB 文件”——大量小 Dataset 的 B-Tree 开销 + 200 份重复内嵌的 MuJoCo XML！
  3. `src/embodied_formats/deep_dive/hdf5.py` → `profile_hdf5_slicing()`
     - **Watch**：`act0_offset` (`20288`)、`seq_us`、`rand_us`
     - **看什么**：看 `contiguous` 无压缩数据集的物理连续偏移与 ~90 µs 微秒级切片延迟。
- **Hex Editor 字节对照**：
  - 打开 `data/real/hdf5_robomimic/demo_v15.hdf5`：
    - `0x00..0x07`：`89 48 44 46 0D 0A 1A 0A`（`\x89HDF\r\n\x1a\n`）
    - 跳转到 `act0_offset = 20288 (0x4F40)`：直接看到 `demo_0/actions` 连续排列的原始 IEEE 754 `float64` 字节！

---

## 4. Zarr / Diffusion Policy（分块目录树 + Blosc Header + 读放大）

- **VS Code `F5` 入口**：`4. Deep Dive: Zarr / Diffusion Policy (Blosc + Read-Amp + Workers)`
- **推荐断点与 Watch 观察点**：
  1. `src/embodied_formats/deep_dive/zarr.py` → `scan_blosc_chunks()`
     - **Watch**：`flags` (`0x94`)、`typesize` (`4`, float32!)、`nbytes` (`17805312`)、`cbytes` (`174925`)、`nbytes / cbytes`
     - **看什么**：看透 Blosc 16 字节二进制头，理解 BitShuffle + Zstd 如何把 2.65 GB float32 图像压成 28.4 MB。
  2. `src/embodied_formats/deep_dive/zarr.py` → `analyze_chunk_alignment_and_read_amp()`
     - **Watch**：`ep_ends[:5]`、`chunk0` (`161`)、`cross_chunk_eps` (`151/206`)、`chunk_uncomp_bytes` (`16.98 MB`)
     - **看什么**：理解 `chunks[0]` 选型对随机读的致命影响（`161x` 读放大）以及 `meta/episode_ends` 如何实现零拷贝扁平轨迹索引。
  3. `src/embodied_formats/deep_dive/zarr.py` → `profile_zarr_access_and_concurrency()`
     - **Watch**：`rand_ms`、`scaling`
     - **看什么**：对比 Zarr 单帧随机读（2.74 ms）vs 无锁多进程读取扩展行为。
- **Hex Editor / 文本对照**：
  - 文本打开 `data/real/zarr_pusht/pusht_cchi_v7_replay.zarr/data/img/.zarray` 看 JSON 控制面；
  - Hex Editor 打开 `data/real/zarr_pusht/pusht_cchi_v7_replay.zarr/data/img/0.0.0.0`：对照前 16 字节 `02 01 94 04 00 B0 0F 01 00 00 04 00 4D AB 02 00`。

---

## 5. ROS 2 bag / MCAP（Footer/Summary + 原始 CDR 解码 + 真机时钟抖动）

- **VS Code `F5` 入口**：`5. Deep Dive: ROS 2 bag / MCAP (Chunks + Raw CDR Decode + Clock Skew)`
- **推荐断点与 Watch 观察点**：
  1. `src/embodied_formats/deep_dive/mcap.py` → `CDRReader` 与 `decode_ros2_compressed_image_cdr()` / `decode_ros2_joint_state_cdr()`
     - **Watch**：`data[:16]`、`self.pos`（观察 `_align(4)` 与 `_align(8)` 的内存对齐步进！）、`fmt` (`"rgb8; jpeg compressed bgr8"`)、`names`、`positions`
     - **看什么**：零 ROS 2 安装直接手解 OMG CDR 二进制协议，C++ 工程师看一眼 `_align()` 就能秒懂 DDS 序列化内存布局。
  2. `src/embodied_formats/deep_dive/mcap.py` → `analyze_async_clocks_and_alignment()`
     - **Watch**：`dts_ms`、`per_topic_jitter`、`best_ms`、`alignment_skew`
     - **看什么**：亲眼看到真机三路相机 `147 ms` 掉帧尖峰、彼此 `10–70 ms` 相位差，以及 193 Hz 关节与 29 Hz 相机的最近邻对齐偏差。
  3. `src/embodied_formats/deep_dive/mcap.py` → `analyze_mcap_sidecars()`
     - **Watch**：`bag_info`、`out["robot_urdf"]`
     - **看什么**：看 `metadata.yaml` + `episode_info.json` + `robot.urdf` 如何与 `.mcap` 配合构成完整真机回放上下文。
- **Hex Editor 字节对照**：
  - 打开 `data/real/mcap_robotis_button_push/0_0.mcap`：
    - 文件首 8 字节与尾 8 字节均为 `89 4D 43 41 50 30 0D 0A`（`\x89MCAP0\r\n`）；
    - 倒数 37 字节处是 `02`（Footer Opcode），紧跟 `summary_start = 161,685,840`。

---

## 通用进阶调试技巧

- **钻入底层 C/Python 库内部**：在 VS Code 按 `F5` 选择 `9. Current Python File (Step into 3rd-party Libs)`（`justMyCode: false`），可直接单步进入 `mcap`、`zarr`、`pyarrow`、`av` 的 Python 封装层。
- **对比 HuggingFace 缓存层**：运行 `7. HF Hub Cache Anatomy (~/.cache/huggingface/hub)` 并在 `src/embodied_formats/lab/anatomy.py` 的 `inspect_hf_hub_cache()` 打断点，观察 `refs/main → snapshots/<sha> → blobs/<hash>` 及 Windows 降级拷贝逻辑。
