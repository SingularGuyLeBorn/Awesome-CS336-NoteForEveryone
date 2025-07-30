### 专题笔记:层归一化 (Layer Normalization)

#### 1. 核心功能

层归一化(Layer Normalization, LayerNorm)是一种在深度神经网络中广泛使用的技术,旨在**稳定训练过程**. 其核心功能是在网络的每一层,对该层神经元的激活值进行归一化处理,使其具有零均值和单位方差. 这有助于:

-   **缓解内部协变量偏移 (Internal Covariate Shift)**:确保每一层输入的分布相对稳定,从而加速模型的收敛. 
-   **平滑损失曲面**:使得优化过程更加容易,允许使用更高的学习率. 
-   **提升训练稳定性**:尤其是在训练非常深的网络(如 Transformer)时,LayerNorm 是防止梯度爆炸或消失的关键组件之一. 

#### 2. 关键变体与演进

在 **[Transformer 架构](./Lecture3-Transformer-Architecture.md)** 的发展历程中,层归一化的应用方式和具体实现也经历了重要的演进. 

##### **A. Pre-Norm vs. Post-Norm**

这是关于 LayerNorm 在 Transformer 模块中**放置位置**的两种不同策略. 

-   **Post-Norm (原始设计)**:
    -   **位置**:在每个子模块(如自注意力和 FFN)**之后**,并且是在与残差连接相加之后进行归一化. 
    -   **公式**:`x = LayerNorm(x + SubLayer(x))`
    -   **问题**:LayerNorm 被置于残差主路径上,可能会干扰梯度的直接传播. 这使得训练深层模型时非常不稳定,通常需要配合精细调整的学习率预热(Warmup)策略. 

-   **Pre-Norm (现代共识)**:
    -   **位置**:在每个子模块**之前**进行归一化. 
    -   **公式**:`x = x + SubLayer(LayerNorm(x))`
    -   **优势**:LayerNorm 不再位于主路径上,保证了从顶层到底层的“清洁”梯度流. 这极大地提升了**[训练稳定性](./Lecture3-Training-Stability-Tricks.md)**,使得模型可以容忍更高的学习率,并且不再严重依赖 Warmup. **几乎所有现代 LLM 都采用 Pre-Norm 结构. **

##### **B. RMSNorm (Root Mean Square Normalization)**

RMSNorm 是对标准 LayerNorm 的一种简化和优化,旨在提升计算效率. 

-   **标准 LayerNorm 公式**:
    `y = (x - E[x]) / sqrt(Var[x] + ε) * γ + β`
    它同时对均值(`E[x]`)和方差(`Var[x]`)进行归一化. 

-   **RMSNorm 公式**:
    `y = x / sqrt(RMS[x] + ε) * γ`,其中 `RMS[x] = mean(x^2)`
    -   **核心改动**:RMSNorm **移除了均值中心化步骤**(即减去均值 `E[x]`)和**偏置项 `β`**. 它只通过均方根(Root Mean Square)来重新缩放激活值. 
    -   **优势**:计算量更少,参数更少. 虽然 FLOPs 的减少微不足道,但它显著降低了内存的读写量,这在以内存带宽为瓶颈的现代 GPU 上,能带来可观的训练速度提升. 实践证明,这种简化几乎不会影响模型性能. 因此,RMSNorm 已成为 **[LLaMA](./Lecture3-LLaMA-Architecture.md)** 等现代模型的主流选择. 

##### **C. Double Norm**

这是最新的发展趋势之一,旨在进一步增强稳定性. 
-   **做法**:在 Pre-Norm 的基础上,在每个子模块的输出端、与残差相加之前,再增加一次 LayerNorm. 
-   **目的**:对子模块的输出进行二次控制,为训练更大、更不稳定的模型提供额外的“安全保障”. Grok、Gemma 2 等模型已开始采纳此设计. 