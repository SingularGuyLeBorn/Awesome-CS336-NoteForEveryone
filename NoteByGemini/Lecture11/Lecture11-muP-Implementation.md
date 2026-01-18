# 代码实现深度解析: muP Scaling Rules

### 1. 核心功能与目标 (Core Function & Goal)
本笔记详细列出了 **[Maximal Update Parametrization (muP)](./Lecture11-muP-Theory.md)** 在 Transformer 架构中的具体实施规则。目标是将标准参数化（SP）转换为 muP，从而实现学习率的跨尺度迁移。

### 2. 关键参数定义 (Parameters)
*   $M$: 模型的宽度（Width），通常指 $d_{model}$。
*   $\alpha$: 基础学习率（Base Learning Rate），需要在小模型上进行调优。
*   $M'$: 代理模型（Proxy Model）的宽度，用于搜索超参数。
*   $n_{layers}$: 层数。

### 3. 核心逻辑与缩放规则 (Core Logic & Scaling Rules)

以下规则适用于 **AdamW** 优化器（这是 LLM 训练的标准配置）。

#### A. 权重形状分类
首先，需要识别参数张量的维度属性：
*   **Matrix-like (矩阵类)**: 具有两个无限维度（即随模型宽度 $M$ 增长的维度）。例如：MLP 的中间层权重、Attention 的投影矩阵。
*   **Vector-like / Output (向量类/输出)**: 具有一个无限维度。例如：Embedding 层、最后的 Logits 输出层。

#### B. 缩放表 (Scaling Table)

| 参数类型 (Parameter) | 初始化方差 (Init Variance) $\sigma^2$ | Adam 学习率 (Adam LR) | 说明 (Notes) |
| :--- | :--- | :--- | :--- |
| **Embedding** ($W^E$) | $1$ (常数) | $1$ (常数) | Embedding 层通常不随宽度缩放其初始化方差。 |
| **Attention Query/Key** ($W^Q, W^K$) | $1/M$ | $1/M$ | 注意力机制内部的投影矩阵。 |
| **Attention Value/Output** ($W^V, W^O$) | $1/M$ | $1/M$ | 注意力输出投影。 |
| **FeedForward (FFN)** ($W^{in}, W^{out}$) | $1/M$ | $1/M$ | MLP 层的权重。 |
| **Output / Logits** ($W^{U}$) | $1/M^2$ | $1/M$ | **注意**: 输出层的初始化方差缩放更为激进 ($1/M^2$)。 |
| **LayerNorm Gains** | - | - | 建议移除可学习增益，或对其 LR 进行特殊处理（通常 muP 对此敏感）。 |

#### C. 架构修改 (Architecture Changes)
除了参数初始化和学习率，muP 还要求修改 Attention 的缩放因子：

```python
# 标准 Transformer Attention Scaling
attention_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_head)

# muP Transformer Attention Scaling
# muP 理论要求缩放因子为 1/d 而非 1/sqrt(d)
attention_scores = (Q @ K.transpose(-2, -1)) / d_head
```

### 4. 与理论的连接 (Connection to Theory)
*   **$1/M$ 的初始化**: 直接对应 **[muP Theory](./Lecture11-muP-Theory.md)** 中的条件 A1，确保随着 $M \to \infty$，矩阵乘法的输出保持 $O(1)$。
*   **$1/M$ 的学习率**: 对应条件 A2。对于 Adam 优化器，由于其自适应地除以梯度的均方根（RMS），梯度的尺度变化被抵消了一部分。推导表明，为了保持更新量的 $O(1)$，学习率需按 $1/fan\_in$ 缩放。
*   **Cerebras-GPT 的应用**: Cerebras 团队严格遵循此表，发现 $d_{model}$ 的缩放系数直接决定了 LR 的缩放，使得他们能在小模型上搜出的最佳 LR 直接用于 13B 模型。