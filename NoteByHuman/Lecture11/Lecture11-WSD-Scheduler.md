# Elite Note: Warmup-Stable-Decay (WSD) Scheduler

**来源:** CS336 Lecture 11 Elite Extension
**核心概念:** Learning Rate Schedules, Chinchilla Analysis, Compute Efficiency, Data Scaling

---

## 1. 传统 Scaling Law 拟合的痛点

在 Chinchilla Scaling Law 的研究中，我们需要回答一个关键问题：**对于给定的模型大小 $N$，在数据量 $D$ 变化时，Loss 是如何变化的？**

为了回答这个问题，我们需要绘制一条 Loss vs Tokens 的曲线，其中每个点都代表一个**训练完全收敛**的模型。

*   **陷阱**：很多人试图只训练一个模型，然后取中间的 Checkpoint 作为“小数据量”的代表。这是**错误**的。
*   **原因**：传统的 **Cosine Decay** 调度器由 Token 总数 $T$ 决定。
    *   如果你计划训练 100B Token，Cosine 曲线会拉得很长，学习率下降得很慢。
    *   如果你只训练 10B Token，Cosine 曲线会迅速下降。
    *   因此，训练 100B Token 的模型在第 10B Token 时的学习率状态，与一个专门训练 10B Token 的模型在结束时的状态完全不同（前者 LR 很高，后者 LR 接近 0）。前者无法代表后者。

这意味着，为了获得 $K$ 个数据点的 Scaling 曲线，你需要从头训练 $K$ 次不同长度的模型。如果还要测试 $M$ 种模型大小，总成本是 $O(M \cdot K \cdot \text{Cost})$，这是一个昂贵的 $N^2$ 级操作。

## 2. WSD 解决方案：Rewind & Decay

**Warmup-Stable-Decay (WSD)** 调度器通过改变学习率曲线的形状，巧妙地解决了这个问题。

### 结构
1.  **Warmup**: 快速预热。
2.  **Stable Phase**: 学习率保持恒定（Constant High LR）。这占据了训练的大部分时间（如 80%-90%）。
3.  **Decay Phase**: 学习率在很短的步数内（如最后 10%）急剧下降到 0 或最小值。

### "Z-Shape" 实验与 Rewind 机制
由于 Stable Phase 的学习率是恒定的，它不依赖于总训练时长。这使得我们可以复用 Stable Phase 的轨迹。

**操作流程**:
1.  **主线训练**: 让模型一直以恒定学习率训练，直到达到最大的 Token 数 $D_{max}$。
2.  **生成数据点**: 假设我们想知道在数据量 $D_1, D_2, \dots$ ($D_i < D_{max}$) 时的最优性能。
3.  **Rewind (回退)**: 取出主线训练在 $D_i$ 时的 Checkpoint。
4.  **Decay (分支衰减)**: 从这个 Checkpoint 开始，执行一个短期的 Decay 阶段（通常只消耗总计算量的 10% 左右）。
5.  **记录**: Decay 结束后的 Loss，就是该模型在数据量 $D_i$ 下完全收敛的性能（Fully Converged Performance）。

### 成本分析
*   **传统方法**: 训练 5 个不同数据量的模型 (10%, 30%, 50%, 70%, 100%)。总计算量 $\approx 10+30+50+70+100 = 260\%$。
*   **WSD 方法**: 训练 1 个完整模型 (100%) + 4 次短 Decay (每次假设为 10% 的额外开销)。总计算量 $\approx 100 + 4 \times 10 = 140\%$。
*   **结论**: 极大地节省了计算资源，使得在一次运行中“扫描”整个数据 Scaling 轴成为可能。

## 3. WSD vs Cosine 性能对比

除了用于 Scaling 分析，WSD 本身作为一种训练策略也是可行的。

*   **MiniCPM 的发现**: 在相同的数据量下，WSD 最终达到的 Loss 通常**优于或等于** Cosine 调度。
*   **损失曲线特征**:
    *   在 Stable 阶段，WSD 的 Loss 下降比 Cosine 慢（因为 LR 保持较高，没有享受到 LR 降低带来的 Loss 下降红利）。
    *   但在进入 Decay 阶段的瞬间，WSD 的 Loss 会**垂直下坠**（Rapid Drop），并在极短时间内追上甚至超越 Cosine 的最终性能。
*   **DeepSeek 的策略**: DeepSeek 采用了类似的分段衰减（Step Decay），本质逻辑相同，也验证了这种策略的有效性。

**总结**: WSD 是现代大模型 Scaling 研究的标准配置，它兼具了工程上的**灵活性**（随时可以停止并 Decay 得到成品模型）和研究上的**高效性**（低成本拟合 Scaling Laws）。