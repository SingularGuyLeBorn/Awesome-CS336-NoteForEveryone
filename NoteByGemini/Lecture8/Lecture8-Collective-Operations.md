# 专题笔记：集合通信操作 (Collective Operations)

### 1. 定义
**集合通信操作 (Collective Operations)** 是一组用于并行计算的标准通信原语，涉及通信组内的所有参与者（Rank）。它们提供了比底层点对点消息传递更高级别的抽象，使得编写分布式程序更加容易和高效。

### 2. 核心术语
*   **World Size**: 通信组中的进程/设备总数。
*   **Rank**: 每个进程/设备在组内的唯一标识符，通常从 0 到 World Size - 1。

### 3. 常见原语图解

*   **Broadcast (广播)**: 将数据从根 Rank 发送到所有其他 Rank。
    *   `Rank 0 [A] -> Rank 0 [A], Rank 1 [A], Rank 2 [A], ...`

*   **Scatter (散射)**: 将根 Rank 上的一个张量切分，分发给不同的 Rank。
    *   `Rank 0 [A, B, C] -> Rank 0 [A], Rank 1 [B], Rank 2 [C]`

*   **Gather (收集)**: Scatter 的逆操作，将所有 Rank 上的数据收集并拼接到根 Rank。
    *   `Rank 0 [A], Rank 1 [B], Rank 2 [C] -> Rank 0 [A, B, C]`

*   **All-Gather (全收集)**: 所有 Rank 都收集所有数据，最终每个 Rank 都拥有完整数据的副本。
    *   `Rank 0 [A], Rank 1 [B] -> Rank 0 [A, B], Rank 1 [A, B]`
    *   常用于 **[张量并行](./Lecture8-Tensor-Parallelism.md)** 中收集激活值。

*   **Reduce (规约)**: 类似于 Gather，但对收集到的数据执行规约运算（如 Sum, Max, Min, Average），结果存放在根 Rank。
    *   `Rank 0 [A], Rank 1 [B] --(Sum)--> Rank 0 [A+B]`

*   **All-Reduce (全规约)**: 对所有 Rank 的数据执行规约运算，并将结果广播给所有 Rank。
    *   `Rank 0 [A], Rank 1 [B] --(Sum)--> Rank 0 [A+B], Rank 1 [A+B]`
    *   这是 **[数据并行](./Lecture8-Data-Parallelism.md)** 中同步梯度的核心操作。
    *   逻辑上的等价关系：**All-Reduce = Reduce-Scatter + All-Gather**。

*   **Reduce-Scatter (规约散射)**: 先对输入执行规约运算，然后将结果切分并散射到各个 Rank。
    *   `Rank 0 [A0, A1], Rank 1 [B0, B1] --(Sum)--> [A0+B0, A1+B1] --(Scatter)--> Rank 0 [A0+B0], Rank 1 [A1+B1]`

### 4. 实现
在深度学习中，这些操作通常由硬件厂商提供的库高效实现，如 NVIDIA 的 **[NCCL](./Lecture8-NCCL.md)**，并通过框架（如 PyTorch 的 **[`torch.distributed`](./Lecture8-Code-TorchDistributed.md)**）暴露给开发者。