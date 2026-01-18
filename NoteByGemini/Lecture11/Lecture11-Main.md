# Lecture 11: Scaling – Case Study and Details

### 前言
在上一讲中，我们探讨了缩放定律（Scaling Laws）的理论基础，特别是 Chinchilla 论文提出的计算最优缩放比。然而，理论与工程实践之间往往存在鸿沟。本讲作为 CS336 课程关于缩放主题的延伸，深入探讨了 **[Scaling Laws](./Lecture11-Scaling-Laws.md)** 在现代模型开发中的实际应用。讲师通过详细拆解 Cerebras-GPT、MiniCPM 和 DeepSeek 等前沿模型的缩放配方（Recipe），揭示了如何通过 **[Maximal Update Parametrization (muP)](./Lecture11-muP-Theory.md)** 实现超参数的跨尺度迁移，以及如何利用 **[WSD Schedule](./Lecture11-WSD-Schedule.md)** 高效地拟合数据缩放曲线。这是一场关于如何将数学预测转化为工程现实的深度剖析。

### 1. 后 Chinchilla 时代的缩放格局
自 Chinchilla 论文发表及随后 ChatGPT 的爆发以来，大模型领域的竞争格局发生了剧变。前沿实验室对具体的缩放细节变得讳莫如深。因此，我们转向那些公开了详细缩放研究的“半生产级”模型——特别是来自 Cerebras、面壁智能（MiniCPM）和深度求索（DeepSeek）的成果。

这些研究主要关注两个核心问题：
1.  **超参数的稳定性**：随着模型规模扩大，如何避免昂贵的超参数重新搜索？
2.  **数据与模型的最优比**：Chinchilla 提出的 20:1 比例是否仍然是黄金法则？

### 2. 实战案例研究：不同的缩放哲学

#### Cerebras-GPT：拥抱稳定性
Cerebras 团队训练了一系列从 111M 到 13B 参数的模型。他们的核心发现是，通过采用 **[Maximal Update Parametrization (muP)](./Lecture11-muP-Theory.md)**，可以显著提高缩放过程的稳定性。与标准参数化（Standard Parametrization, SP）相比，muP 使得最优学习率在不同模型宽度下保持不变，从而极大地简化了超参数调整的难度。

#### MiniCPM：小模型的大智慧
MiniCPM 展示了如何训练高质量的小型模型（1.2B-2.4B）。他们采用了两项关键技术：
*   **muP 初始化**：用于稳定超参数。
*   **[WSD Schedule](./Lecture11-WSD-Schedule.md)**：即“预热-稳定-衰减”学习率调度。这种调度方式允许在单次训练运行中，通过在“稳定阶段”的不同点进行“衰减”操作，来模拟不同数据量的训练结果。这使得 **[IsoFLOPs Analysis](./Lecture11-IsoFLOPs.md)** 能够在极低的计算成本下完成。

值得注意的是，MiniCPM 拟合出的最优数据/参数比高达 **192:1**，远超 Chinchilla 的 20:1。这暗示了随着数据质量和训练技术的提升，我们应当在固定模型大小的情况下，通过增加数据量来挖掘更多潜力。

#### DeepSeek LLM：严谨的经验主义
DeepSeek 的方法则更为直接。他们没有使用 muP，而是通过在小规模模型上进行网格搜索，直接拟合关于 **[Critical Batch Size](./Lecture11-Critical-Batch-Size.md)** 和学习率的缩放定律。他们同样采用了 **[WSD Schedule](./Lecture11-WSD-Schedule.md)** 来高效获取 Chinchilla 曲线。DeepSeek 的成功表明，即使不依赖复杂的参数化技巧，只要进行严谨的缩放分析，也能精准预测大模型的性能。

### 3. 深入解析 Maximal Update Parametrization (muP)
本讲的一大重点是数学层面的 **[Maximal Update Parametrization (muP)](./Lecture11-muP-Theory.md)** 推导。在标准参数化下，随着模型变宽，最优学习率通常会发生漂移（Shift），迫使研究者在每个尺度上重新调参。

muP 基于两个核心的物理直觉（或称为谱条件）：
1.  **激活值稳定性**：在初始化时，激活值的尺度应保持为 $O(1)$，不随宽度 $n$ 发散。
2.  **更新量稳定性**：经过一次梯度步进后，激活值的变化量（$\Delta h$）也应保持为 $O(1)$。

为了满足这些条件，muP 要求我们将权重的初始化方差设置为 $1/n$（而非标准的 $1/\sqrt{n}$），并根据具体的优化器（SGD 或 Adam）调整每层的学习率缩放因子。具体的实施细节详见 **[muP Implementation](./Lecture11-muP-Implementation.md)** 笔记。

此外，一项独立的第三方复现研究（Lingle, 2024）表明，muP 在多种现代架构变体（如 SwiGLU, Squared ReLU）下依然有效，但对某些特定设置（如 **[RMSNorm](./Lecture11-RMSNorm.md)** 的可学习增益、Lion 优化器或强权重衰减）较为敏感。

### 4. 总结与推荐
在“野外”训练大模型时，我们面临着架构选择、优化器设置和计算资源限制的三重挑战。结合本讲案例，最佳实践建议如下：
1.  **考虑使用 muP** 或类似的参数化方法，以获得跨尺度的超参数稳定性。
2.  **采用 WSD 调度器**，它不仅能提供与余弦调度相当的性能，还能大幅降低拟合 Scaling Laws 的成本。
3.  **不要迷信 20:1**，现代模型（如 Llama 3）的数据/参数比往往更高，应根据实际算力和数据情况进行 **[IsoFLOPs Analysis](./Lecture11-IsoFLOPs.md)**。

### 拓展阅读
建议按以下顺序阅读笔记，以构建完整的知识体系：
1.  首先阅读 **[WSD Schedule](./Lecture11-WSD-Schedule.md)**，理解现代训练中高效拟合 Scaling Law 的核心工具。
2.  深入 **[Maximal Update Parametrization (muP)](./Lecture11-muP-Theory.md)**，掌握让超参数“一劳永逸”的数学原理。
3.  对照 **[muP Implementation](./Lecture11-muP-Implementation.md)**，查看具体的参数初始化和学习率设置规则。
4.  最后参考 **[Modern Scaling Recipes](./Lecture11-Modern-Scaling-Recipes.md)**，对比 Cerebras、MiniCPM 和 DeepSeek 的不同工程选择。