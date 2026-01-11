# 代码深度解析：投机采样算法 (Speculative Sampling Algorithm)

### 1. 核心功能与目标 (Core Function & Goal)
本模块模拟了投机采样的核心概率逻辑。它演示了如何通过调整接受概率，将小模型（Draft Model $p$）的采样分布修正为大模型（Target Model $q$）的分布。

### 2. 参数解析 (Parameters)
*   `q`: Target Model (大模型) 的概率分布向量。
*   `p`: Draft Model (草稿/小模型) 的概率分布向量。
*   `x`: 词表中的某个 Token。

### 3. 核心逻辑 (Core Logic)

```python
def speculative_sampling():
    # ... (前文解释了原理) ...

    text("Compute the probabilities of speculatively sampling a token:")
    # 假设只有两个Token {A, B}
    # q(A), q(B) 是大模型概率
    # p(A), p(B) 是小模型概率

    # 场景假设: 小模型过度自信地预测了 A (p(A) > q(A))
    # 因此小模型低估了 B (p(B) < q(B))

    # 1. 采样 A 的总概率 (P[sampling A])
    # 路径一: 小模型采样到了 A (概率 p(A))，且被大模型接受。
    # 接受概率为 min(1, q(A)/p(A))。因为 p(A)>q(A)，接受率为 q(A)/p(A)。
    # 路径二: 小模型采样到了 B (概率 p(B))，但被拒绝了，并在修正步骤中重采样到了 A。
    # 在这个简化推导中，代码展示了路径一的结果直接等于 q(A)
    # P[sampling A] = p(A) * (q(A) / p(A)) = q(A)

    # 2. 采样 B 的总概率 (P[sampling B])
    # 路径一: 小模型采样到了 B (概率 p(B))。
    # 因为 p(B) < q(B)，接受率为 1。
    # 贡献 = p(B) * 1
    # 路径二: 小模型采样到了 A (概率 p(A))，但被拒绝了 (拒绝率 1 - q(A)/p(A))。
    # 此时需要从残差分布中重采样。在这个二元例子中，重采样必然命中 B。
    # 贡献 = p(A) * (1 - q(A)/p(A)) * 1 = p(A) - q(A)
    # 总概率 = p(B) + p(A) - q(A) = 1 - q(A) = q(B)

    # 结论: 无论小模型 p 如何分布，最终采样结果的边缘分布都严格等于 q
    text("- P[sampling A] = ... = q(A)")
    text("- P[sampling B] = ... = q(B)")

    # 这证明了投机采样是"精确"的采样方法，不会降低模型质量
    text("Key property: guaranteed to be an **exact sample** from the target model!")
```

### 4. 与理论的连接 (Connection to Theory)
代码中的推导是 **[投机采样理论](./Lecture10-Theory-Speculative.md)** 的直接数学证明。它展示了为什么我们可以放心地使用一个小得多的模型（如 1B 参数）来加速一个巨型模型（如 70B 参数），只要我们执行了正确的拒绝采样逻辑，用户得到的输出质量与 70B 模型完全一致，但速度却快得多。