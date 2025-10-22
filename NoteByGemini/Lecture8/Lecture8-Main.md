# Lecture 8: 手撕大模型并行训练 (Distributed Training)

### 前言

在上周的课程中，我们探讨了单个 GPU 内部的并行性，重点在于通过算子融合（Fusion）和分块（Tiling）来利用极其快速但容量有限的 L1 缓存/共享内存，从而减少对较慢的高带宽内存（HBM）的读写。本周，我们将视野扩展到多 GPU 甚至多节点（Node）的**[分布式训练](./Lecture8-Distributed-Training.md)**。

这是一个从小而快到大而慢的**[硬件通信层级](./Lecture8-Hardware-Hierarchy.md)**。核心主题依然不变：**编排计算以避免数据传输瓶颈**。我们需要保持高算术强度以饱和 GPU 的计算能力。在分布式场景下，数据传输（跨 GPU 或跨节点）通常比本地内存访问慢得多，因此它成为了主要的瓶颈。为了克服这一点，我们需要跨设备复制或分片（Shard）模型参数、梯度和优化器状态。

本节课旨在将关于并行性的概念具体化为代码。我们将分为两部分：首先研究分布式通信的构建模块——**[集合通信操作](./Lecture8-Collective-Operations.md)**及其在 PyTorch 中的实现；然后，我们将动手实现三种核心的分布式训练策略：**[数据并行](./Lecture8-Data-Parallelism.md)**、**[张量并行](./Lecture8-Tensor-Parallelism.md)**和**[流水线并行](./Lecture8-Pipeline-Parallelism.md)**。所有的代码实现都将基于纯 PyTorch 从零开始构建，以深入理解底层机制。

---

### 拓展阅读指南

建议按照以下顺序结合理论与代码进行学习，以构建完整的分布式训练知识体系：

1.  **基础设施与原语**：首先阅读 **[硬件通信层级](./Lecture8-Hardware-Hierarchy.md)** 以理解物理限制。接着学习理论上的 **[集合通信操作](./Lecture8-Collective-Operations.md)**，然后深入 **[`torch.distributed` 模块解析](./Lecture8-Code-TorchDistributed.md)**，对照代码理解 All-Reduce 等原语的具体实现与基准测试。
2.  **并行策略实战**：
    *   **数据并行**：先理解 **[数据并行 (DDP)](./Lecture8-Data-Parallelism.md)** 的理论，再查阅 **[`data_parallelism_main` 实现解析](./Lecture8-Code-DataParallelism.md)**，看如何在标准 SGD 循环中注入梯度同步。
    *   **张量并行**：研读 **[张量并行 (TP)](./Lecture8-Tensor-Parallelism.md)** 了解矩阵切分方式，随后通过 **[`tensor_parallelism_main` 实现解析](./Lecture8-Code-TensorParallelism.md)** 学习在前向传播中如何切分计算并聚合激活值。
    *   **流水线并行**：最后学习 **[流水线并行 (PP)](./Lecture8-Pipeline-Parallelism.md)** 及其通过 **[微批次](./Lecture8-Micro-batches.md)** 缓解气泡的机制，并参考 **[`pipeline_parallelism_main` 实现解析](./Lecture8-Code-PipelineParallelism.md)** 理解点对点通信在层间数据传递中的应用。

---

### 正文

#### 第一部分：构建模块——集合通信操作 (Collective Operations)

**[集合通信操作](./Lecture8-Collective-Operations.md)**是分布式编程的原语。这里的“集合”意味着通信模式涉及多个（例如 4 个或更多）节点。这些原语提供了比手动管理点对点（Point-to-Point）通信更好的抽象。

主要的逻辑操作包括：
*   **Broadcast**：将数据从一个等级（Rank，即设备）发送到所有其他等级。
*   **Scatter**：将一个张量切分，并将不同的切片发送到不同的等级。
*   **Gather**：将不同等级的张量收集到一个等级上拼接起来。
*   **Reduce**：类似于 Gather，但不是拼接，而是执行某种满足结合律和交换律的运算（如求和、最大值）。
*   **All-Gather**：所有等级都收集所有其他等级的数据，最终每个等级都拥有完整的数据副本。
*   **[All-Reduce](./Lecture8-Collective-Operations.md#all-reduce)**：对所有等级的数据进行 Reduce 操作，并将结果分布到所有等级上。在逻辑上，**All-Reduce 等价于 Reduce-Scatter 加上 All-Gather**。

在硬件层面，现代数据中心为了应对深度学习负载，通常会绕过传统的 CPU 和以太网路径。NVIDIA 的 GPU 通过 **NVLink** 在单节点内直连，并通过 **NVSwitch** 实现跨节点直连，提供远超 PCIe 和以太网的带宽（例如 H100 的 NVLink 总带宽可达 900 GB/s）。

为了利用这些硬件，NVIDIA 提供了 **[NCCL](./Lecture8-NCCL.md)**（NVIDIA Collective Communications Library）。它负责检测硬件拓扑，优化 GPU 间的通信路径，并启动 CUDA 内核来收发数据。

在软件层面，我们通过 PyTorch 的 **[`torch.distributed` 模块](./Lecture8-Code-TorchDistributed.md)**来使用这些功能。底层的编程模型通常是 **[SPMD (单程序多数据)](./Lecture8-SPMD.md)**：我们启动多个进程（由 `world_size` 指定），每个进程运行相同的代码副本，但拥有唯一的 `rank`（从 0 到 `world_size - 1`），并根据 rank 操作不同的数据切片。

我们在代码中演示了这些原语的使用。例如，**[All-Reduce](./Lecture8-Code-TorchDistributed.md#3-核心逻辑-core-logic)** 操作会在所有进程中同步并将指定张量的值相加：

```python
# 每个 rank 上的 tensor 初始值不同
dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
# 操作后，所有 rank 上的 tensor 值变为初始值的总和
```

我们也通过基准测试验证了 All-Reduce 等价于 **[Reduce-Scatter](./Lecture8-Code-TorchDistributed.md#3-核心逻辑-core-logic)** 后接 **[All-Gather](./Lecture8-Code-TorchDistributed.md#3-核心逻辑-core-logic)**。在实际硬件上的基准测试显示，All-Reduce 传输的数据量是张量大小的两倍（发送输入，接收输出），而 Reduce-Scatter 则是一倍。

#### 第二部分：分布式训练策略

我们将通过手写代码在一个深层 MLP（多层感知机）上实现三种主要的并行策略。虽然 MLP 结构简单，但它代表了 Transformer 中前馈网络部分的计算瓶颈。

##### 1. 数据并行 (Data Parallelism)

在 **[数据并行](./Lecture8-Data-Parallelism.md)** 中，我们将数据沿批次（Batch）维度切分，而模型在每个 GPU 上完整复制。

实现的核心在于“劫持”标准的 SGD 训练循环。每个 Rank 获取属于自己的数据切片，进行独立的前向和反向传播，计算出局部梯度。在更新参数之前，我们插入一个 **[All-Reduce](./Lecture8-Collective-Operations.md#all-reduce)** 操作来同步（平均）所有 Rank 上的梯度。

具体的 **[`data_parallelism_main` 实现](./Lecture8-Code-DataParallelism.md)** 显示，尽管不同 Rank 上的损失（Loss）因数据不同而不同，但在执行 All-Reduce 后，所有 Rank 的梯度变得一致，从而保证了参数更新的一致性。这在数学上等价于在一个大批次上进行训练。

##### 2. 张量并行 (Tensor Parallelism)

当模型大到无法放入单个 GPU 的显存时，我们需要切分模型。**[张量并行](./Lecture8-Tensor-Parallelism.md)** 将每一层的权重矩阵沿隐藏维度（Hidden Dimension）切分。这意味着每个 Rank 只持有模型参数总量的 `1 / world_size`。

在前向传播中，每个 Rank 计算其拥有的部分权重的矩阵乘法，得到部分激活值。为了进行下一层的计算，必须通过 **[All-Gather](./Lecture8-Collective-Operations.md)** 操作在所有 Rank 间同步完整的激活向量。

我们的 **[`tensor_parallelism_main` 代码](./Lecture8-Code-TensorParallelism.md)** 展示了这一过程：计算部分结果，分配用于存储完整激活值的缓冲区，执行 All-Gather，然后拼接得到完整的输入以供下一层使用。由于每一层都需要通信，张量并行对设备间的互联带宽要求极高。

##### 3. 流水线并行 (Pipeline Parallelism)

另一种切分模型的方法是沿深度（层数）方向，这就是 **[流水线并行](./Lecture8-Pipeline-Parallelism.md)**。每个 Rank 负责模型的一组连续层。数据从第一个 Rank 流入，依次经过各个 Rank 处理。

朴素的实现会导致严重的硬件空闲，称为**[流水线气泡](./Lecture8-Pipeline-Parallelism.md#流水线气泡-pipeline-bubbles)**。为了缓解这一问题，我们将一个小批次（Mini-batch）进一步切分为多个 **[微批次](./Lecture8-Micro-batches.md)**。当前的 Rank 在处理完一个微批次并将其发送给下一个 Rank 后，可以立即开始处理下一个微批次，从而重叠计算。

在 **[`pipeline_parallelism_main` 实现](./Lecture8-Code-PipelineParallelism.md)** 中，我们使用点对点的 `dist.send` 和 `dist.recv` 原语在相邻 Rank 间传递激活值。Rank 0 负责切分数据，中间的 Rank 接收输入、计算并将结果发送给下一个 Rank。

#### 总结

本节课的实现是为了教学目的而进行的简化（Bare-bones）版本。在实际应用中（如 Megatron-LM 或 PyTorch FSDP），还需要处理更复杂的模型架构（如 Transformer 的注意力头切分）、仔细管理计算与通信的重叠（异步操作）、以及处理反向传播时的调度逻辑。

尽管硬件在不断进步，但模型规模的增长总会触及单设备的物理极限。因此，理解并掌握基于**[硬件通信层级](./Lecture8-Hardware-Hierarchy.md)**的分布式训练策略，将始终是构建大规模 AI 系统的核心能力。