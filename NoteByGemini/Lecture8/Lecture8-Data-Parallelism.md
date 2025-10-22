# 专题笔记：数据并行 (Data Parallelism, DDP)

### 1. 核心思想
**数据并行 (DDP)** 是最常见的分布式训练策略。当模型可以放入单个 GPU，但数据量太大导致训练太慢时使用。
*   **模型复制**：完整的模型副本存在于每一个 GPU (Rank) 上。
*   **数据分片**：全局的小批次（Global Mini-batch）沿批次维度被均匀切分为多个微批次，分发给各个 Rank。

### 2. 训练流程
1.  **前向传播 (Forward Pass)**：每个 Rank 使用其本地的数据分片和本地模型副本独立进行前向计算，得到损失值。此时，不同 Rank 上的损失值是不同的。
2.  **反向传播 (Backward Pass)**：每个 Rank 独立计算损失相对于本地参数的梯度。
3.  **梯度同步 (Gradient Synchronization)**：这是关键步骤。在更新参数之前，所有 Rank 必须通信以计算全局梯度的平均值。这通常通过 **[All-Reduce](./Lecture8-Collective-Operations.md#all-reduce)** (Op=SUM 或 AVG) 操作实现。
4.  **参数更新 (Parameter Update)**：每个 Rank 使用同步后的全局平均梯度更新其本地模型参数。
5.  **一致性保证**：由于所有 Rank 初始参数相同，且使用相同的平均梯度进行更新，因此在每一步结束时，所有 Rank 上的模型参数保持严格一致。

### 3. 数学等价性
在数学上，DDP 的训练过程完全等价于在单个 GPU 上使用全局批次大小（Global Batch Size = Local Batch Size $\times$ World Size）进行的顺序训练。

### 4. 通信开销
通信发生在反向传播之后。传输的数据量与模型参数量成正比，与批次大小无关。主要依赖高带宽的互联（如 NVLink 或高性能网络）来执行 **[All-Reduce](./Lecture8-Collective-Operations.md#all-reduce)**。代码实现可参见 **[`data_parallelism_main`](./Lecture8-Code-DataParallelism.md)**。