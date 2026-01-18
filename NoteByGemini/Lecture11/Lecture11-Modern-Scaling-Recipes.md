# 专题笔记：Modern Scaling Recipes (Cerebras, MiniCPM, DeepSeek)

### 1. 概览
本笔记对比了三个后 Chinchilla 时代具有代表性的模型缩放策略（Recipe）。它们展示了在计算资源受限或需要极致优化时，不同的工程权衡。

### 2. Cerebras-GPT: 追求预测性
*   **目标**: 验证 Chinchilla 定律，并确立 Scaling 的可预测性。
*   **核心策略**:
    *   **架构**: 标准 GPT 架构。
    *   **参数化**: 全面采用 **[muP](./Lecture11-muP-Theory.md)**。
    *   **发现**: 确认了 muP 能够使最优学习率与模型宽度解耦。在 40M 参数的小模型上搜出的超参数，可以直接按规则迁移到 13B 模型上，且 Loss 预测极其精准。
*   **数据/模型比**: 严格遵循 Chinchilla 的 20:1。

### 3. MiniCPM: 小模型极致优化
*   **目标**: 在小参数规模（2B级别）下挖掘极致性能。
*   **核心策略**:
    *   **参数化**: 同样采用 **[muP](./Lecture11-muP-Theory.md)** 来稳定初始化和 LR。
    *   **调度器**: 引入并普及了 **[WSD Schedule](./Lecture11-WSD-Schedule.md)**，用于高效拟合数据缩放曲线。
    *   **数据缩放**: 通过 WSD 产生的数据点进行曲线拟合，他们发现对于 2B 级别的模型，最优的数据/参数比高达 **192:1**（远超 20:1）。这表明小模型如果“过度训练”（Over-train），可以达到非常高的性能，且在推理成本上极具优势。
*   **结论**: 摩尔定律在模型大小上可能失效，但在数据利用率上依然有效。

### 4. DeepSeek LLM: 直接与务实
*   **目标**: 构建强大的 7B 和 67B 开源模型。
*   **核心策略**:
    *   **参数化**: **不使用 muP**。DeepSeek 团队选择直接在小规模模型上对 **[Critical Batch Size](./Lecture11-Critical-Batch-Size.md)** 和学习率进行网格搜索。
    *   **Scaling Law**: 他们发现最佳 Batch Size 和 学习率与计算量（FLOPs）之间存在对数线性关系，并直接利用此规律外推到 67B 模型。
    *   **调度器**: 同样采用 WSD（分两阶段衰减）来复现 Chinchilla 分析。
*   **结论**: 即使没有复杂的参数化技巧，只要缩放分析做得足够细致（Brute-force scaling analysis），依然可以精准预测大模型的行为。

### 5. 对比总结
| 特性 | Cerebras-GPT | MiniCPM | DeepSeek |
| :--- | :--- | :--- | :--- |
| **超参数稳定性** | **muP** (核心卖点) | **muP** | 传统方法 + 缩放外推 |
| **学习率调度** | 传统 | **WSD** | **WSD** |
| **Scaling 重点** | 模型宽度缩放 | 数据量缩放 (Data-Hungry) | 综合 FLOPs 缩放 |
| **数据/参数比** | ~20:1 | ~192:1 | 动态分析 (IsoFLOPs) |