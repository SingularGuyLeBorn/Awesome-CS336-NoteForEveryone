# 专题笔记：WSD Schedule (预热-稳定-衰减)

### 1. 核心概念
**WSD (Warmup-Stable-Decay)** 是一种学习率调度策略，近年来在 **[MiniCPM](./Lecture11-CaseStudies.md)** 和 **[DeepSeek LLM](./Lecture11-CaseStudies.md)** 等模型的训练中被广泛采用。它的形状呈梯形，包含三个阶段：
1.  **Warmup (预热)**: 学习率从 0 线性增加到最大值。
2.  **Stable (稳定)**: 学习率保持在最大值不变，持续训练大部分时间（例如 80%-90% 的 Token）。
3.  **Decay (衰减)**: 在训练结束前，学习率迅速下降（通常是 1-cosine 或线性衰减）到 0 或最小值。

### 2. 为什么优于 Cosine Schedule？
传统的 Cosine Schedule（余弦退火）在过了预热期后，学习率就开始持续下降。这意味着：
*   **一次性**：每次训练的衰减曲线都是针对特定的总步数（Total Steps）固定的。如果你想在训练到一半时决定“这就够了”，此时的学习率还很高，模型并未收敛（Loss 依然很高）。
*   **无法复用**：如果要测试不同数据量下的模型性能，需要针对每个数据量重新跑完整的 Cosine 训练（因为形状不同）。

**WSD 的优势**在于其“可复用性”和“解耦性”：
*   **Scaling Law 拟合神器**: 在 **Stable** 阶段，模型处于持续学习状态。我们可以从 Stable 阶段的任意点（比如 10%、20%... 100% 数据量处）取出一个 Checkpoint，然后对其执行一个短期的 **Decay** 阶段（Cool-down）。
*   **一鱼多吃**: 这样，只需训练一个主干模型（Stable 阶段），加上多次低成本的 Decay 分支，就能获得该模型在不同数据量下的最终收敛性能。这使得拟合 **[IsoFLOPs Analysis](./Lecture11-IsoFLOPs.md)** 曲线的成本从 $O(N^2)$ 降低到了接近 $O(N)$。

### 3. 经验发现
*   **衰减的重要性**: 实验表明，模型 Loss 的大幅下降主要发生在 Decay 阶段。这被称为“冷却效应”（Cool-down effect）。即使 Stable 阶段 Loss 下降缓慢，一旦进入 Decay，Loss 会迅速收敛。
*   **性能对比**: 大多数研究表明，WSD 最终达到的收敛效果与经过完美调整的 Cosine Schedule 相当，甚至在某些长训练周期任务中表现更好。