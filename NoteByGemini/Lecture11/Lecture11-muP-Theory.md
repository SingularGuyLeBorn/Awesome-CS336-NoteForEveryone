# 专题笔记：Maximal Update Parametrization (muP)

### 1. 核心概念与动机
**Maximal Update Parametrization (muP)** 是一种旨在解决深度神经网络在宽度缩放（Width Scaling）时超参数不稳定问题的参数化方法。

在标准参数化（Standard Parametrization, SP）中，随着神经网络宽度（$n$）的增加，保持模型收敛的最优学习率通常会发生变化。这意味着研究者无法在小模型上调优超参数，然后直接迁移到大模型上。muP 通过调整权重的初始化方差和学习率缩放因子，使得**最优超参数（特别是学习率）在不同宽度的模型上保持恒定**。

### 2. 两个核心谱条件 (Spectral Conditions)
muP 的推导基于两个关于神经网络行为的“物理”假设（或限制条件），旨在确保信号在网络中既不爆炸也不消失，且模型能进行有效的特征学习。

#### 条件 A1: 初始化时的激活值稳定性
随着网络宽度 $n \to \infty$，每一层的预激活值（pre-activations）和激活值向量的坐标元素大小应保持为 $O(1)$。
*   这意味着激活值向量的范数 $\|h\|_2$ 应随 $\sqrt{n}$ 增长。
*   为了满足此条件，权重矩阵 $W$ 的初始化方差通常需要缩放为 $1/n$（对于特定的层），这与标准的 He Initialization ($1/\sqrt{n}$) 有所不同。

#### 条件 A2: 梯度更新后的稳定性
在进行一次梯度更新步（step）后，激活值的**变化量**（$\Delta h$）相对于宽度 $n$ 应保持为 $O(1)$。
*   如果我们希望模型在训练初期就能学到特征（Feature Learning），权重矩阵的更新量 $\Delta W$ 必须足够大，以便对输出产生 $O(1)$ 的影响。
*   这导出特定的学习率缩放规则。对于 SGD 和 Adam，推导出的缩放规则是不同的。

### 3. 数学推导简述
考虑一个简单的线性层 $h_l = W_l h_{l-1}$。
*   **初始化**: 为了让 $\|h_l\| \approx \|h_{l-1}\|$，我们需要 $\|W_l\|_{op} \approx 1$。根据随机矩阵理论，高斯矩阵的算子范数与其元素方差 $\sigma^2$ 的关系约为 $\|W\| \propto \sigma \sqrt{n}$。因此，我们需要 $\sigma \propto 1/\sqrt{n}$。这是标准做法。
*   **更新**: 考虑 SGD 更新 $\Delta W = -\eta \nabla \mathcal{L} h^T$。激活值的变化量 $\Delta h = \Delta W h$。
    *   在 muP 中，为了最大化更新量同时保持稳定性，我们发现对于矩阵乘法层，学习率 $\eta$ 需要根据 fan_in ($n_{in}$) 和 fan_out ($n_{out}$) 进行调整。
    *   **SGD**: $\eta \propto \frac{n_{out}}{n_{in}}$
    *   **Adam**: 由于 Adam 对梯度的二阶矩进行了归一化，其更新量的尺度与梯度的尺度关系不同。在 muP 下，Adam 的学习率通常缩放为 $\eta \propto \frac{1}{n_{in}}$。

### 4. 局限性与鲁棒性
第三方研究（如 Lingle, 2024）表明：
*   **鲁棒性**: muP 能够很好地配合 SwiGLU、Squared ReLU 激活函数以及不同的 Batch Size。
*   **脆弱性**: 当引入 **[RMSNorm](./Lecture11-RMSNorm.md)** 的可学习增益（learnable gains）、使用 Lion 优化器或非常强的权重衰减时，muP 的超参数迁移特性可能会失效。
*   **实现细节**: 在 Transformer 中，muP 通常要求注意力分数的缩放因子从 $1/\sqrt{d}$ 改为 $1/d$（注意：这是一个关键的架构变更）。

### 关联代码笔记
*   **[muP Implementation](./Lecture11-muP-Implementation.md)**: 具体的初始化和学习率设置表。