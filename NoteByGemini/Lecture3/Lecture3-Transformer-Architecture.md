### 专题笔记:Transformer 架构 (Transformer Architecture)

#### 1. 核心思想

Transformer 架构由 Vaswani 等人在 2017 年的开创性论文 **[`Attention Is All You Need`](./Lecture3-Attention-Is-All-You-Need.md)** 中首次提出,其核心是彻底摒弃了以往序列建模中广泛使用的循环(Recurrence)和卷积(Convolution)结构,完全依赖于“自注意力机制”(Self-Attention)来捕捉输入序列内的依赖关系. 这一设计极大地提升了计算的并行度,为训练更大、更深的模型铺平了道路,并最终引爆了大型语言模型(LLM)的革命. 

#### 2. 原始架构的关键组件

经典的 Transformer 架构主要由编码器(Encoder)和解码器(Decoder)两部分构成,每个部分都是由若干个相同的层堆叠而成. 

*   **输入嵌入与位置编码 (Input Embedding & Positional Embedding)**
    *   **词嵌入**:将输入的离散词元(tokens)转换为连续的向量表示. 
    *   **[位置编码](./Lecture3-Positional-Embedding.md)**:由于自注意力机制本身不包含序列顺序信息,必须额外引入位置编码来向模型注入 token 的位置信息. 原始论文使用的是正弦和余弦函数(Sine & Cosine)的组合. 

*   **编码器 (Encoder)**
    每个编码器层包含两个核心子层:
    1.  **多头自注意力 (Multi-Head Self-Attention)**:这是模型的核心. 它允许输入序列中的每个位置都能“关注”到序列中的所有其他位置,并计算出一个加权的表示. 多头(Multi-Head)则允许模型在不同的表示子空间中并行地学习不同的注意力模式. 
    2.  **位置全连接前馈网络 (Position-wise Feed-Forward Network)**:这是一个简单的两层全连接网络,独立地应用于每个位置的输出上. 它为模型增加了非线性变换能力. 
    *   **残差连接与层归一化**:每个子层都包裹在残差连接(Residual Connection)中,其后紧跟着一个**[层归一化](./Lecture3-Layer-Normalization.md)**(Layer Normalization). 这种设计被称为 **Post-Norm**,有助于缓解梯度消失问题并稳定训练. 

*   **解码器 (Decoder)**
    每个解码器层在编码器层的基础上,增加了一个额外的子层:
    1.  **掩码多头自注意力 (Masked Multi-Head Self-Attention)**:与编码器类似,但增加了“掩码”(Masking)机制. 在生成第 `i` 个位置的输出时,掩码会阻止注意力机制关注到 `i` 之后的位置,以确保模型的自回归(auto-regressive)特性,即预测未来时只能依赖于过去的输出. 
    2.  **编码器-解码器注意力 (Encoder-Decoder Attention)**:这个子层允许解码器的每个位置关注到编码器输出的所有位置. 这是连接编码器和解码器的桥梁,使得解码器能够利用完整的输入序列信息. 
    3.  **位置全连接前馈网络**:与编码器中的 FFN 相同. 
    *   同样,每个子层也都采用了残差连接和 Post-Norm LayerNorm. 

*   **输出层 (Output Layer)**
    解码器栈的最终输出会经过一个线性层(Linear Layer)和一个 Softmax 层,以生成在整个词汇表上的概率分布,用于预测下一个词元. 

#### 3. 架构的演进与现代变体

尽管原始 Transformer 架构非常成功,但经过多年的发展,业界已经形成了一套新的“标准实践”,即以 **[LLaMA](./Lecture3-LLaMA-Architecture.md)** 为代表的现代架构. 这些变体在保持核心思想不变的同时,做出了一系列优化,以提升性能、稳定性和效率. 主要变化包括:

-   **Pre-Norm LayerNorm**: 将层归一化移至每个子模块之前. 
-   **RMSNorm**: 使用更高效的 RMSNorm 替代 LayerNorm. 
-   **SwiGLU**: 在 FFN 中使用 SwiGLU 激活函数. 
-   **RoPE**: 采用旋转位置编码. 
-   **无偏置项**: 在大多数线性层中移除偏置项. 

这些改进共同构成了当今大多数高性能 LLM 的架构基础. 