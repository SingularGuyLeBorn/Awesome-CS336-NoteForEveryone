# Elite Note: Maximal Update Parametrization (muP) - Theory & Derivation

**来源:** CS336 Lecture 11 Elite Extension
**核心概念:** Tensor Programs, Spectral Conditions, Initialization, Learning Rate Scaling

---

## 1. 为什么我们需要 muP？

在深度学习中，当我们改变模型的宽度（Width, $n$）时，模型的许多行为都会发生变化。如果使用标准的参数化方法（Standard Parametrization, SP），即无论宽度如何都使用相同的初始化方差和学习率，我们会发现：
1.  **激活值爆炸或消失**：随着 $n \to \infty$，前向传播的信号可能变得极大或极小。
2.  **更新量不稳定**：梯度更新可能导致权重变化过大，破坏训练稳定性。
3.  **超参数漂移**：最优学习率 $\eta^*$ 会随着 $n$ 的变化而剧烈移动。这迫使我们在训练大模型（如 GPT-3, Llama）时，必须在昂贵的大规模设置下重新搜索超参数。

**Maximal Update Parametrization (muP)** 的目标是设计一套规则，使得在宽度 $n \to \infty$ 的极限下，模型的**特征学习（Feature Learning）**行为保持最大化且稳定。这使得我们可以从一个小模型（Proxy Model）无缝迁移最优超参数到大模型。

## 2. 核心数学假设：深度线性网络与谱条件

为了推导 muP，讲座中简化了一个**深度线性网络 (Deep Linear Network)** 模型：
$$ h_l = W_l h_{l-1} $$
其中 $W_l \in \mathbb{R}^{n_l \times n_{l-1}}$。

muP 的推导基于两个核心的**谱条件 (Spectral Conditions)**，这实际上是物理学中“重整化群（Renormalization Group）”思想在神经网络中的应用：确保物理量在尺度变换下保持有限。

### 条件 A1：激活值稳定性 (Stability of Activations)
**目标**: 无论宽度 $n$ 如何，激活值向量的坐标级数值应保持 $O(1)$。这意味着激活值向量的 $L_2$ 范数应为 $O(\sqrt{n})$。

**推导**:
假设输入 $h_0$ 满足 $\|h_0\|_2 = \Theta(\sqrt{n_0})$。
权重 $W_l$ 初始化为 $N(0, \sigma^2)$。
根据随机矩阵理论，高斯矩阵的算子范数（Operator Norm，即最大奇异值）集中在：
$$ \|W_l\|_* \approx \sigma (\sqrt{n_l} + \sqrt{n_{l-1}}) $$
前向传播的范数关系为：
$$ \|h_l\|_2 \le \|W_l\|_* \|h_{l-1}\|_2 $$
为了维持归纳假设 $\|h_l\|_2 = \Theta(\sqrt{n_l})$，我们需要 $\|W_l\|_* = \Theta(1)$。
代入算子范数公式：
$$ \sigma (\sqrt{n_l} + \sqrt{n_{l-1}}) = \Theta(1) \implies \sigma \propto \frac{1}{\sqrt{n}} $$
这解释了为什么标准的 Xavier/Kaiming Initialization ($Var \propto 1/n$) 是正确的，因为它保证了前向信号的稳定。

### 条件 A2：更新量稳定性 (Stability of Updates)
**目标**: 在一步梯度更新后，激活值的变化量 $\Delta h_l$ 也应保持与 $h_l$ 相同的量级，即 $O(\sqrt{n})$。这被称为“Maximal Update”，即我们在不导致发散的前提下，尽可能大地更新模型。

**推导 (SGD case)**:
SGD 的权重更新为秩-1 更新（Rank-1 Update）：
$$ \Delta W_l = -\eta \nabla_{W_l} \ell = -\eta (\nabla_{h_l} \ell) h_{l-1}^T $$
激活值的变化量（忽略高阶项）为：
$$ \Delta h_l \approx \Delta W_l h_{l-1} + W_l \Delta h_{l-1} $$
我们需要关注第一项 $\Delta W_l h_{l-1}$ 的量级。
代入 $\Delta W_l$：
$$ \Delta W_l h_{l-1} = -\eta (\nabla_{h_l} \ell) (h_{l-1}^T h_{l-1}) $$
注意 $(h_{l-1}^T h_{l-1}) = \|h_{l-1}\|_2^2 = \Theta(n_{in})$。
因此，更新量的量级大致为：
$$ \|\Delta h_l\| \propto \eta \cdot n_{in} \cdot \|\nabla_{h_l} \ell\| $$
为了让 $\|\Delta h_l\|$ 保持 $O(\sqrt{n})$（假设梯度项也是良态的），我们需要：
$$ \eta \cdot n_{in} = \Theta(1) \implies \eta \propto \frac{1}{n_{in}} $$

**关键修正 (The Adam Difference)**:
上述推导是基于 SGD 的。对于 **Adam**，情况完全不同。
Adam 的更新步长大致仅取决于梯度的符号（或标准化后的梯度），它消除了梯度幅度的影响。
$$ \Delta W_{l, \text{Adam}} \approx -\eta \cdot \text{sign}(\nabla_{W_l} \ell) $$
对于高斯矩阵，矩阵元素的更新不再与 $n$ 成正比，而是更加均匀。详细的 Tensor Program 分析表明，为了让 $\Delta W_l h_{l-1}$ 保持稳定，我们需要：
$$ \eta_{\text{Adam}} \propto \frac{1}{n} $$

## 3. 实施细节：Scaling Table

在 Transformer 的具体实现中（如 Cerebras-GPT），muP 的缩放规则如下：

| 参数类型 | 初始化方差 (Init Var) | Adam 学习率 (LR) | 说明 |
| :--- | :--- | :--- | :--- |
| **Embedding** | 1 (or specific scale) | 1 (Fixed) | Embedding 层通常不缩放，因为它是 One-hot 查找 |
| **Matrix Weights** | $1/n$ (Fan-in) | $1/n$ (Fan-in) | 核心权重层 (Attention, MLP) |
| **Output/Readout** | $1/n^2$ | $1/n$ | 输出层通常需要更小的初始化以防止 Logits 爆炸 |

**注意**: Cerebras-GPT 的实现中特别提到，Standard Parametrization (SP) 的学习率是全局常数，而 muP 要求每层（Per-Layer）的学习率根据其输入维度 $n$ 进行 $1/n$ 的缩放。

## 4. 为什么 "A2" 被称为 "Maximal Update"?
如果学习率比 muP 建议的更小（例如 $1/n^2$），则 $\Delta h$ 会随着 $n \to \infty$ 趋向于 0，模型在初始化附近无法有效学习（Feature Learning 退化为 Kernel Regime/Neural Tangent Kernel）。
如果学习率比 muP 建议的更大（例如 $O(1)$），则 $\Delta h$ 会爆炸，导致训练发散。
因此，muP 定义的是**在保持训练稳定的前提下，理论上允许的最大学习率缩放比例**，从而最大化特征学习的效率。