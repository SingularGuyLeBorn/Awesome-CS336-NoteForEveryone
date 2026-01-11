# Lecture 10 Deep Dive: 投机采样 (Speculative Sampling) 的理论正确性

投机采样 (Speculative Sampling/Decoding) 最反直觉也最迷人的性质是：**虽然使用了弱模型（Draft Model）来生成 Token，但最终输出的概率分布与只使用强模型（Target Model）完全一致。**

这是一个**无损**的加速算法。本文将详细解释其背后的数学原理（拒绝采样）。

## 1. 核心问题

我们有两个模型：
*   **Target Model (q)**: 强但慢，概率分布为 $q(x|prefix)$。
*   **Draft Model (p)**: 弱但快，概率分布为 $p(x|prefix)$。
通常，$p$ 的计算成本远低于 $q$。

我们的目标是：从 $p$ 中快速采样，然后通过某种修正机制，使得最终输出的序列像是直接从 $q$ 中采样的一样。

## 2. 算法流程

假设 Draft Model 预测了 $K$ 个 token。对于每一个 token $x_{t}$：

1.  **采样**: Draft Model 给出概率 $p(x)$，我们从中采样得到 token $x$。
2.  **验证**: Target Model 并行计算该位置上所有词的概率 $q(x)$。
3.  **接受判定**: 我们计算接受率 $\alpha$:
    $$ \alpha = \min\left(1, \frac{q(x)}{p(x)}\right) $$
    生成一个随机数 $u \in [0, 1]$。
    *   如果 $u \le \alpha$，**接受**该 token。
    *   如果 $u > \alpha$，**拒绝**该 token，并终止当前 Speculation，使用 $q(x)$ 的修正分布重新采样下一个 token。

## 3. 为什么分布是无损的？ (数学证明)

我们要证明的是：经过上述接受/拒绝过程后，输出 token $x$ 的边缘概率分布 $P(x)$ 恰好等于 $q(x)$。

### Case 1: $q(x) \ge p(x)$
这意味着 Draft Model 低估了 $x$ 的概率。
*   接受率 $\alpha = \min(1, \frac{q(x)}{p(x)}) = 1$。
*   Draft Model 提出 $x$ 的概率是 $p(x)$。
*   一旦提出，必被接受。
*   所以这部分的贡献是 $p(x) \times 1 = p(x)$。
*   但这还不够 $q(x)$，缺少的 $q(x) - p(x)$ 部分将由**拒绝恢复机制**（Resampling）补足。

### Case 2: $q(x) < p(x)$
这意味着 Draft Model 高估了 $x$ 的概率。
*   接受率 $\alpha = \frac{q(x)}{p(x)}$。
*   Draft Model 提出 $x$ 的概率是 $p(x)$。
*   被接受的概率是 $\frac{q(x)}{p(x)}$。
*   所以最终输出 $x$ 的概率是：
    $$ P(\text{emit } x) = p(x) \times \frac{q(x)}{p(x)} = q(x) $$
*   **完美匹配！**

### 修正项 (Resampling)
当某个 token 被拒绝时，我们从修正后的分布中采样：
$$ p'(x) = \text{norm}(\max(0, q(x) - p(x))) $$
这个步骤保证了之前 Case 1 中“少采样”的部分被精确补回。

## 4. 结论

通过这种拒绝采样机制（Rejection Sampling），投机采样巧妙地利用了“小模型的猜测”来减少大模型的计算次数，同时**完全保留了大模型的“智商”（概率分布）**。

这与类似于 Beam Search 的近似搜索不同，Speculative Decoding 是**精确**的。如果你的应用需要严格的 Temperature=0 (Greedy) 或特定的 Top-P 采样，Speculative Decoding 都能保证结果 bit-level 一致。