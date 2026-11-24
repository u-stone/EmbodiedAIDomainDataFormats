我建议把之前的机器人全景学习计划暂时降级为一个长期知识索引，而不是当前需要逐章执行的课程。

你眼下的目标应该调整为：

**具备独立开展 VLA 模型训练、调优、评估和问题诊断的工程能力。**

考虑到你有 C/C++ 系统编程经验，并且已经跑通 LeRobot、SmolVLA 的基础训练流程，继续从机器人学基础理论开始学习，效率并不高。更合适的方法是采用 **Code-First + Top-Down（代码优先、自顶向下）** 的学习方式。

先从一个实际运行的工程系统切入，沿着调用链、数据流、模型结构逐层深入。当遇到无法理解的技术问题时，再补充背后的数学、算法和机器人学知识。

# **一、重新划定学习范围**

暂时将整个机器人知识体系划分为三个优先级。

| 知识领域                                | 当前学习深度         |
| ----------------------------------- | -------------- |
| Python、PyTorch 训练相关 API             | 深入到代码级         |
| VLA 数据集格式、数据处理与转换                   | 深入到代码级         |
| LeRobot 框架与训练接口                     | 深入到源码级         |
| VLA 模型结构与动作生成                       | 深入到实现级         |
| 模型预训练、Fine-tuning、LoRA              | 深入到实践级         |
| 模型评估、实验对照、故障诊断                      | 深入到实践级         |
| 机器人仿真与闭环评估                          | 掌握必要接口         |
| 机器人状态、动作空间、坐标系                      | 按实际需要学习        |
| Transformer、Flow Matching、Diffusion | 结合模型源码学习       |
| ROS 2、运动学、动力学、控制论                   | 暂时按需学习         |
| 机器人硬件、实时系统、工业通信                     | 暂缓系统学习         |
| 机器人产业、厂商和产品                         | 保持关注，不占用主要学习时间 |

你不需要先成为一个机器人学专家，才有资格开始训练 VLA 模型。但必须逐渐理解模型的输入输出如何对应真实机器人的物理行为。

# **二、以一条工程调用链作为学习主线**

接下来的所有学习，都围绕下面这个系统展开。

**原始机器人数据**

图像、关节状态、动作、语言指令、Episode

**① Dataset / DataLoader**

Parquet、MP4、采样、Batch、时序数据

**② Preprocessor**

图像处理、Tokenizer、Normalization、Tensor 转换

#### **③ VLA Policy / Model**

Vision Encoder

Language Model

Robot State

**Action Expert / Action Head**

生成动作或计算动作学习目标

训练分支

**④ Loss**

**Backward**

**Optimizer**

**Checkpoint**

推理分支

**⑤ Action Chunk**

**Postprocessor**

**Robot / Simulator**

**Evaluation**

**⑥ 实验分析与模型调优**

分析失败 → 提出假设 → 修改数据、模型或训练策略 → 对照实验

VLA 训练与推理的概念性工程架构。不同模型内部实现不同，但这条主线具有较好的通用性。

当你能够理解每个模块的职责、输入输出、接口和内部实现，并且有能力替换其中的某个模块进行对照实验时，就具备了初步独立研究 VLA 模型的能力。

# **三、将学习计划压缩为六个工程阶段**

不再按照学科划分课程，而是按照需要解决的工程问题划分。

| 阶段 | 核心课题       | 实践成果                      | 建议投入  |
| -- | ---------- | ------------------------- | ----- |
| 01 | 解剖现有训练流程   | 训练调用链与最小训练程序              | 2–3 天 |
| 02 | 解剖训练数据流    | Dataset Inspector 与数据处理实验 | 2–3 天 |
| 03 | 解剖 VLA 模型  | SmolVLA 模型结构及张量流分析        | 3–4 天 |
| 04 | 模型训练与调优    | 参数冻结、学习率、LoRA 等对照实验       | 4–5 天 |
| 05 | 推理、仿真与评估   | 可重复的闭环评估系统                | 3–4 天 |
| 06 | 新场景适配与失败诊断 | 一个完整的模型调优案例               | 4–5 天 |

按照每天投入 3–4 小时计算，大约需要 **18–24 个学习日**。这只是第一轮学习的目标预算，不包含长时间训练任务及复杂的环境故障处理。

第一轮结束后，才考虑进入 OpenVLA、π 系列、GR00T 等不同模型的横向研究。

# **四、具体从哪里开始：解剖&#x20;**`lerobot-train`

我认为这是当前最合适的切入点。

你之前已经使用过类似这样的命令：

```
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=lerobot/svla_so100_pickplace \
    --policy.device=cuda \
    --batch_size=1 \
    --steps=100 \
    --wandb.enable=false
```

但这只是调用了一个封装好的训练入口。

我们接下来要做的是，把这个入口当作一个陌生的 C++ 工程来研究：找到程序入口、配置对象、核心类、工厂方法、执行流程，以及各个模块之间的数据契约。

## **4.1 先建立源码阅读地图**

以 LeRobot 官方仓库为参考：

[LeRobot GitHub 源码仓库](https://github.com/huggingface/lerobot?utm_source=chatgpt.com)

第一轮只需要重点关注以下部分，不要试图阅读整个仓库。

**lerobot/**

01

scripts/lerobot_train.py

训练入口、流程编排、参数更新

02

configs/

训练参数、模型配置及配置解析

03

datasets/

数据读取、Episode、采样、数据格式

04

processor/

归一化、Tokenizer、数据预处理

05

policies/factory.py

Policy 构建及工厂机制

06

policies/pretrained.py

Policy 基类及接口契约

07

policies/smolvla/

SmolVLA 模型的具体实现

08

optim/

优化器与学习率调度

09

scripts/lerobot_eval.py

模型评估与 Rollout

源码路径可能随 LeRobot 版本变化，因此实际学习必须以你的实验环境中安装的版本为准。

### **建议重点阅读的五份源文件**

| 阅读顺序 | 源文件                                                                                                                                                       | 阅读目标           |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| 1    | [examples/training/train_policy.py](https://github.com/huggingface/lerobot/blob/main/examples/training/train_policy.py?utm_source=chatgpt.com)            | 理解最简训练程序       |
| 2    | [lerobot_train.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_train.py?utm_source=chatgpt.com)                          | 理解完整训练框架       |
| 3    | [pretrained.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pretrained.py?utm_source=chatgpt.com)                               | 理解 Policy 接口定义 |
| 4    | [configuration_smolvla.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py?utm_source=chatgpt.com) | 理解模型参数         |
| 5    | [modeling_smolvla.py](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py?utm_source=chatgpt.com)           | 理解模型实现         |

尤其值得关注的是官方的 `examples/training/train_policy.py`，它展示的是 Diffusion Policy 的训练，而不是 SmolVLA，但相比完整的命令行训练入口更加精简，非常适合用来理解 LeRobot 的基本训练 API。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## **4.2 从 C/C++ 的知识体系迁移到模型训练工程**

你不必重新学习一遍软件工程。许多概念可以直接借助已有的编程经验理解。

| C/C++ 中熟悉的概念  | 模型训练工程中的对应概念                    |
| ------------- | ------------------------------- |
| 程序入口 `main()` | Training Entry Point            |
| 配置结构体         | Python Dataclass / Train Config |
| 抽象基类          | `PreTrainedPolicy`              |
| 工厂模式          | Policy Factory                  |
| 接口实现类         | `SmolVLAPolicy`                 |
| 数据结构与容器       | Tensor、Dict、Batch               |
| 数据读取器         | Dataset、DataLoader              |
| 函数调用          | Forward Pass                    |
| 插件机制          | Processor、Policy、Environment 扩展 |
| 对象序列化         | State Dict、Checkpoint           |
| 调试器与性能分析      | debugpy、PyTorch Profiler        |
| 回归测试          | Model Evaluation                |
| 性能基准测试        | Benchmark                       |
| 控制变量测试        | Ablation Study                  |

真正需要额外学习的是 Tensor、自动微分、模型参数更新、随机性、数据分布、训练目标以及模型泛化等概念。

这些知识都可以结合具体代码逐渐掌握。

# **五、后续课程的具体安排**

## **阶段一：模型训练框架解剖**

**核心问题：一个训练程序究竟在做什么？**

| 课题                   | 关键代码与知识                                            |
| -------------------- | -------------------------------------------------- |
| 1.1 训练程序入口           | `lerobot-train`、配置解析、执行流程                          |
| 1.2 Tensor 与 PyTorch | shape、dtype、device、维度操作                            |
| 1.3 模型对象             | `nn.Module`、`parameters()`、`state_dict()`          |
| 1.4 单次训练迭代           | `forward()`、Loss、`backward()`、`step()`             |
| 1.5 模型保存与恢复          | `save_pretrained()`、`from_pretrained()`、Checkpoint |
| 1.6 手写训练循环           | 脱离 CLI，直接调用训练 API                                  |

这一阶段结束后，要能够自己编写一个最小训练程序。

## **阶段二：机器人训练数据解剖**

**核心问题：模型实际学习的是什么数据？**

| 课题           | 关键代码与知识                            |
| ------------ | ---------------------------------- |
| 2.1 数据集文件结构  | Parquet、MP4、JSON、Metadata          |
| 2.2 数据读取     | `LeRobotDataset`、`__getitem__()`   |
| 2.3 Batch 构造 | `DataLoader`、Batch Size、Collate    |
| 2.4 机器人数据语义  | State、Action、Observation、Episode   |
| 2.5 时序数据     | FPS、Timestamp、Action Chunk、Padding |
| 2.6 数据预处理    | Normalization、Tokenizer、图像处理       |
| 2.7 数据转换     | HDF5 / RLDS → LeRobotDataset       |
| 2.8 数据质量     | 数据分布、错误数据、训练验证集划分                  |

这一阶段结束后，需要能够解释一个 Batch 中每一个 Tensor 的来源及物理意义。

## **阶段三：SmolVLA 模型解剖**

**核心问题：VLA 模型如何将视觉、语言和机器人状态转化为动作？**

| 课题                | 关键代码与知识                                                |
| ----------------- | ------------------------------------------------------ |
| 3.1 模型对象构成        | `SmolVLAPolicy`、`VLAFlowMatching`                      |
| 3.2 模型配置          | `SmolVLAConfig`                                        |
| 3.3 视觉编码          | 图像处理、Vision Encoder、Embedding                          |
| 3.4 语言编码          | Tokenizer、Token、Embedding                              |
| 3.5 Transformer   | Attention、Mask、位置编码                                    |
| 3.6 VLM Backbone  | 视觉与语言特征的处理                                             |
| 3.7 Action Expert | 机器人动作的表示与生成                                            |
| 3.8 Flow Matching | 加噪、速度场预测、动作生成                                          |
| 3.9 训练与推理差异       | `forward()`、`predict_action_chunk()`、`select_action()` |

这里才是深入 Transformer、Flow Matching 等理论的合适时机。

例如，在 SmolVLA 当前的实现中，训练阶段会构造带噪声的动作以及目标速度场，模型预测速度场，并以 MSE 作为学习目标的一部分，而不是简单地将预测动作与示教动作直接做 MSE。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



学习理论的直接目的，是能够读懂这些代码。

## **阶段四：模型训练调优**

**核心问题：如何有目的地改变模型能力？**

| 课题         | 关键代码与知识                               |
| ---------- | ------------------------------------- |
| 4.1 参数管理   | `named_parameters()`、`requires_grad`  |
| 4.2 参数冻结   | Freeze、Unfreeze、部分模块训练                |
| 4.3 全量微调   | Full Fine-tuning                      |
| 4.4 参数高效微调 | LoRA、PEFT、Adapter                     |
| 4.5 优化器    | AdamW、Learning Rate、Weight Decay      |
| 4.6 训练策略   | Warmup、Scheduler、Gradient Clipping    |
| 4.7 显存优化   | Mixed Precision、Gradient Accumulation |
| 4.8 训练诊断   | Loss、Gradient Norm、过拟合、欠拟合            |
| 4.9 实验对照   | Baseline、Ablation、随机种子                |

## **阶段五：模型推理与评估**

**核心问题：如何判断训练后的模型是否真的变好了？**

| 课题                | 关键代码与知识                                     |
| ----------------- | ------------------------------------------- |
| 5.1 Checkpoint 加载 | 模型与 Processor 的恢复                           |
| 5.2 离线推理          | 使用 Dataset 观察预测结果                           |
| 5.3 动作生成          | Action Chunk、Action Queue                   |
| 5.4 仿真环境          | `reset()`、`step()`                          |
| 5.5 闭环评估          | Observation → Policy → Action → Environment |
| 5.6 评价指标          | Success Rate、Latency、轨迹误差                   |
| 5.7 泛化能力          | 新物体、新位置、新指令、视觉变化                            |
| 5.8 失败分析          | 失败轨迹、视频、动作与状态分析                             |

特别注意：离线 Loss 下降不能直接证明机器人任务成功率提高，必须结合闭环评估。

## **阶段六：完整的新场景调优案例**

最终选定一个机器人和操作任务，完成以下完整过程：

```
确定目标机器人与操作任务
    ↓
确定评价协议
    ↓
选择基础模型
    ↓
评估基础模型（Baseline）
    ↓
收集或构造适配数据
    ↓
检查模型与数据的兼容性
    ↓
确定 Fine-tuning 策略
    ↓
训练并保存 Checkpoint
    ↓
进行闭环评估
    ↓
分析失败原因
    ↓
调整数据或训练策略
    ↓
重新训练、比较结果
```

这个案例才是真正意义上的 VLA 工程学习，而不是简单地运行一次训练命令。

# **六、第一课：现在就可以开始**

## **Lesson 01 — 训练入口与单次参数更新**

第一课不研究 Transformer，不研究机器人运动学，也不研究复杂的数据格式。

只做一件事：

**跟踪一次真实的 SmolVLA 训练，弄清楚模型参数究竟是如何被修改的。**

### **Step 1：确定当前实验环境中的源码位置**

进入你现有的 LeRobot Policy Python 环境，执行：

```
which python
which lerobot-train

python -c "import torch; print(torch.__version__)"
python -c "import lerobot; print(lerobot.__file__)"
```

然后执行以下脚本定位训练模块：

```
python - <<'PY'
import inspect
import importlib.metadata

import lerobot.scripts.lerobot_train as trainer

print("LeRobot version:")
print(importlib.metadata.version("lerobot"))

print("\nTraining module:")
print(inspect.getfile(trainer))

if hasattr(trainer, "train"):
    print("\nTrain function:")
    print(inspect.signature(trainer.train))

if hasattr(trainer, "update_policy"):
    print("\nUpdate policy function:")
    print(inspect.signature(trainer.update_policy))
PY
```

这里使用的 `inspect` 是 Python 标准库提供的反射与源码检查工具。

首先确定本地版本实际有哪些接口，避免 GitHub 的 `main` 分支与已安装版本不一致。

### **Step 2：阅读训练程序的主调用链**

打开 `lerobot_train.py`，先不要逐行理解。

搜索这些关键字：

```
train
dataset
dataloader
make_policy
preprocessor
optimizer
update_policy
backward
checkpoint
```

重点找到以下几处代码的位置：

| 代码位置              | 要回答的问题           |
| ----------------- | ---------------- |
| 配置解析              | 命令行参数怎样进入程序？     |
| Dataset 创建        | 数据集对象如何构建？       |
| Policy 创建         | SmolVLA 参数怎样被加载？ |
| DataLoader 创建     | 如何取出一个 Batch？    |
| Training Loop     | 每次迭代执行什么操作？      |
| `update_policy()` | 模型参数怎样被修改？       |
| Checkpoint        | 修改后的参数怎样保存？      |

当前官方 LeRobot 源码中的 `update_policy()` 承担了一次策略更新的主要职责，包括前向计算、反向传播、梯度裁剪和优化器更新，同时还涉及 Accelerator 等工程机制。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



### **Step 3：理解最核心的几行代码**

以下是概念性的最小 PyTorch 训练循环，省略了数据构建、混合精度、分布式训练等工程细节：

```
policy.train()for batch in dataloader:    batch = preprocessor(batch)    optimizer.zero_grad()    loss, output = policy.forward(batch)    loss.backward()    optimizer.step()
```

LeRobot 中大量的训练基础设施，最终都是围绕这一类核心过程展开的。

这里需要掌握的五个 PyTorch API：

| API                     | 实际作用               |
| ----------------------- | ------------------ |
| `policy.train()`        | 将模型设置为训练模式，本身不更新参数 |
| `policy.forward(batch)` | 执行前向计算，获得训练目标      |
| `optimizer.zero_grad()` | 清理之前累积的梯度          |
| `loss.backward()`       | 通过自动微分计算梯度         |
| `optimizer.step()`      | 根据优化算法修改可训练参数      |

### **Step 4：做一个最小 PyTorch 实验**

在真正调试 SmolVLA 之前，先运行以下代码。这个实验虽然非常简单，但可以清楚观察训练过程中参数发生的变化。

```
import torchtorch.manual_seed(42)model = torch.nn.Linear(1, 1)optimizer = torch.optim.SGD(    model.parameters(),    lr=0.01,)x = torch.tensor([[1.0], [2.0], [3.0]])y = torch.tensor([[2.0], [4.0], [6.0]])for step in range(5):    optimizer.zero_grad()    prediction = model(x)    loss = torch.nn.functional.mse_loss(prediction, y)    loss.backward()    print(f"Step: {step}")    print(f"Loss: {loss.item():.6f}")    print(f"Weight before: {model.weight.item():.6f}")    print(f"Gradient: {model.weight.grad.item():.6f}")    optimizer.step()    print(f"Weight after: {model.weight.item():.6f}")    print("-" * 40)
```

通过这个实验观察：

- Forward 执行后，权重有没有变化？
- Backward 执行后，权重有没有变化？
- `optimizer.step()` 执行后，权重发生了什么变化？
- 梯度与学习率怎样影响权重的更新？
- Loss 是怎样逐步下降的？

### **交互实验：观察学习率如何改变训练结果**

简化为无 Bias 的模型 y = wx，训练目标为 w = 2，使用 MSE 和 SGD。

Learning Rate

0.030

初始权重 w

0.50

初始 Loss

# **10.5000**

第 10 步 Loss

# **0.0147**

最终权重：**1.9438**

尚未收敛到目标

恢复默认参数

分别尝试较小、适中和过大的学习率，观察为什么有些训练过程收敛得很慢，有些会发生震荡甚至发散。

SmolVLA 的参数更新远比这个例子复杂，但自动微分和优化器更新的基本机制是一致的。

### **Step 5：调试一次真实的 SmolVLA 训练**

使用 VSCode 调试现有的训练程序，将训练步数设置为 1–5，Batch Size 设置为 1。

在以下几个位置设置断点：

1. Dataset 读取完成后。
2. Batch 预处理完成后。
3. `SmolVLAPolicy.forward()` 入口。
4. `loss.backward()` 或 `accelerator.backward()` 前后。
5. Optimizer Step 前后。

观察 Tensor 的：

```
tensor.shapetensor.dtypetensor.devicetensor.requires_grad
```

观察模型的参数及梯度：

```
for name, param in policy.named_parameters():    if param.requires_grad:        print(name, param.shape)
```

第一课不要求理解每个模型层的内部算法，只要求能够追踪一次完整的训练迭代，弄清楚数据在哪里进入模型、Loss 在哪里产生、梯度在哪里计算，以及模型参数在哪里发生变化。

# **七、学习过程的代码管理**

建议暂时不要继续大规模扩展 `vla-research-lab` 的基础设施，而是在现有项目中建立一个独立的学习实验区域。

```
vla-research-lab/
│
├── ...                     # 已有实验平台
│
└── learning-labs/
    ├── README.md
    │
    ├── 01-training-internals/
    │   ├── inspect_environment.py
    │   ├── minimal_pytorch_train.py
    │   ├── inspect_parameters.py
    │   ├── minimal_lerobot_train.py
    │   └── notes.md
    │
    ├── 02-dataset-internals/
    ├── 03-policy-internals/
    ├── 04-finetuning/
    ├── 05-inference-evaluation/
    └── 06-task-adaptation/
```

这里的代码是为了学习、验证和理解机制，不必立即按照生产级框架的要求设计。

等到某项实验验证了有价值的能力，再考虑将其集成到正式的实验平台中。这样能够防止你把大量时间花在框架重构、配置管理和基础设施建设上，而迟迟没有进入模型训练调优本身。

# **八、接下来我们的学习方式**

后续每一课都采用同一个结构：

| 环节                      | 内容                |
| ----------------------- | ----------------- |
| Engineering Question    | 提出一个具体的工程问题       |
| Source Code             | 找到相关第三方库及源码入口     |
| API Analysis            | 分析类、函数、输入、输出及接口契约 |
| Data Flow               | 跟踪真实数据和 Tensor    |
| Essential Theory        | 补充读懂代码所必需的理论      |
| Hands-on Lab            | 编写最小可运行实验         |
| Debugging               | 分析参数变化、错误与运行结果    |
| Engineering Application | 说明如何用于实际模型训练调优    |

**我建议接下来我们就正式进入第一课的源码级讲解：**`lerobot-train`**&#x20;背后的训练框架与核心 API。**

具体从 `lerobot_train.py` 入手，逐个拆解它如何构造 Dataset、Policy、Processor、Optimizer，以及如何进入 Training Loop。在理解这些内容后，编写我们自己的 `minimal_lerobot_train.py`，直接调用 LeRobot 的 Python API，完成一次 SmolVLA 模型微调。

这是从“会使用模型训练工具”转变为“能够开发模型训练程序”的第一步。