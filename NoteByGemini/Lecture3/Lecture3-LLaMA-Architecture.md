### 专题笔记:LLaMA 架构

#### 1. 背景与意义

LLaMA(Large Language Model Meta AI)是由 Meta AI 发布的一系列大型语言模型. 它的出现具有里程碑式的意义,并非因为它在架构上有颠覆性的创新,而是因为它**整合并验证了一套高效、稳定且性能卓越的“最佳实践”**,为开源社区提供了一个高质量、可复现的基准. 自 LLaMA 发布以来,其架构设计已成为事实上的行业标准,许多后续的开源模型(如 Mistral, Yi, Qwen 等)都沿用了或微调了其核心设计,形成了所谓的“LLaMA-like”架构. 

#### 2. LLaMA 架构的核心组件

LLaMA 是一个仅解码器(Decoder-only)的 **[Transformer 架构](./Lecture3-Transformer-Architecture.md)**,其设计哲学是追求在给定计算预算下的最佳性能. 相较于原始 Transformer,LLaMA 做出了一系列关键的改进,这些改进共同构成了现代 LLM 的主流范式:

1.  **Pre-Norm 和 RMSNorm**:
    *   LLaMA 采用了 **Pre-Norm** **[层归一化](./Lecture3-Layer-Normalization.md)**,即在每个 Transformer 子模块(注意力和 FFN)之前进行归一化. 这被证明是提升训练稳定性的关键. 
    *   它使用 **[RMSNorm](./Lecture3-Layer-Normalization.md)** 替代了传统的 LayerNorm. RMSNorm 更计算高效,因为它省略了均值中心化步骤,从而在不牺牲性能的前提下加速了训练. 

2.  **SwiGLU 激活函数**:
    *   在前馈网络(FFN)中,LLaMA 使用了 **[SwiGLU](./Lecture3-Activation-Functions.md)** 激活函数,这是一种**[门控线性单元 (GLU)](./Lecture3-Activation-Functions.md)**. 相比于 ReLU,SwiGLU 提供了更强的表达能力,并被广泛证明可以提升模型性能. 
    *   为了保持与使用 ReLU 的 FFN 参数量相当,LLaMA 将 FFN 的隐藏层维度设置为 `2/3 * 4 * d_model`,即 `(8/3) * d_model`,而不是传统的 `4 * d_model`. 

3.  **旋转位置编码 (RoPE)**:
    *   LLaMA 采用了**[旋转位置编码 (RoPE)](./Lecture3-Positional-Embedding.md)** 来注入序列的位置信息. RoPE 是一种相对位置编码方案,它通过旋转 Query 和 Key 的嵌入向量来编码位置,具有良好的外推性能,能够更好地处理比训练时更长的序列. 

4.  **移除偏置项**:
    *   与许多现代模型一样,LLaMA 在其网络的大多数线性层(包括注意力投影和 FFN)中移除了偏置项(bias terms). 这被认为有助于提升**[训练稳定性](./Lecture3-Training-Stability-Tricks.md)**并略微减少参数量. 

#### 3. 演进:从 LLaMA 到 LLaMA 3

-   **LLaMA 2** 引入了**[分组查询注意力 (GQA)](./Lecture3-Attention-Variants.md)**,这是一个在标准多头注意力和**[多查询注意力 (MQA)](./Lecture3-Attention-Variants.md)**之间的折中方案,旨在显著降低推理时的内存带宽消耗,从而在几乎不影响模型性能的情况下提升推理速度. 
-   **LLaMA 3** 延续了 LLaMA 2 的核心架构,但在 tokenizer 和训练数据上做了重大升级,使用了更大的词汇表(128k)和更高质量、更大规模的训练数据,从而取得了性能上的巨大飞跃. 

总之,学习 LLaMA 架构是理解现代大型语言模型设计的最佳切入点. 