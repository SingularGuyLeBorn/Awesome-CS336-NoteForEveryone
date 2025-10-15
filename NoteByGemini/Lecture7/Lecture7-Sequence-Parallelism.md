### 概念: 序列并行 (Sequence Parallelism)

#### 1. 核心定义

序列并行是一种专门用于优化 Transformer 模型内存占用的并行技术, 特别是针对**激活值 (activations)** 的内存. 它的核心思想是利用 Transformer 中许多操作在**序列维度 (sequence dimension)** 上的独立性, 对这些操作进行并行化. 它是对**[张量并行](./Lecture7-Tensor-Parallelism.md)**的一个重要补充.

#### 2. 问题背景: 张量并行的局限性

**[张量并行](./Lecture7-Tensor-Parallelism.md)**非常擅长并行化模型中的大规模矩阵乘法 (如 `Linear` 层), 并且能够线性地减少这些操作产生的激活内存. 然而, Transformer 中还包含大量非矩阵乘法的逐点 (point-wise) 操作, 例如:
- LayerNorm
- Dropout
- 残差连接 (Residual Connections)

这些操作在张量并行中通常是在每个 GPU 上对整个张量重复执行的, 它们产生的激活值并没有被分片, 成为了激活内存中一个无法随并行度增加而缩减的“固定开销”. 随着序列长度的增加, 这部分内存占用会变得非常显著.

#### 3. 序列并行的工作原理

序列并行的关键洞察是: **上述逐点操作对于序列中的每个 token 的计算是完全独立的**. 对第 `i` 个 token 进行 LayerNorm, 与对第 `j` 个 token 进行 LayerNorm 没有任何关系.

基于此, 序列并行采取以下策略:

1.  **沿序列维度分片**: 将输入张量 (形状通常为 `[sequence_length, batch_size, hidden_dim]`) 沿着 `sequence_length` 维度进行切分. 如果张量并行度为 `T`, 则每个 GPU 只会得到 `[sequence_length / T, batch_size, hidden_dim]` 大小的输入块.

2.  **并行计算逐点操作**: 每个 GPU 在其本地的数据分片上独立地执行 LayerNorm, Dropout 等操作. 这样, 这些操作的激活内存就被成功地分片了, 每个 GPU 的内存占用减少为原来的 `1/T`.

3.  **数据布局转换**:
    - 模型中的计算并非全部都适合在序列分片上进行. 例如, **[张量并行](./Lecture7-Tensor-Parallelism.md)**的 `Linear` 层需要在 `hidden_dim` 维度上进行分片.
    - 因此, 在序列并行区域和张量并行区域之间, 需要进行数据布局的转换. 这通常通过**[集体通信操作](./Lecture7-Collective-Communication.md)**来完成.
    - **`All-Gather`**: 在进入张量并行区域之前, 对序列分片的数据执行 `All-Gather`, 以便在每个 GPU 上重建完整的序列张量.
    - **`Reduce-Scatter`**: 在从张量并行区域退出时, 对输出结果执行 `Reduce-Scatter`, 将其重新转换为按序列维度分片的布局.

#### 4. 效果与意义

序列并行通过补全张量并行的短板, 使得 Transformer 模型中几乎所有的计算和激活内存都可以随着并行度的增加而线性扩展. 它与张量并行和**[激活重计算](./Lecture7-Activation-Recomputation.md)**相结合, 是实现对超长上下文窗口 (如数十万甚至上百万 token) 进行高效训练的关键技术.