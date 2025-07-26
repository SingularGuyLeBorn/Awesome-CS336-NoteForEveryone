# 专题：Transformer
## 1. 核心思想
Transformer 是 Google 在 2017 年的论文《Attention Is All You Need》中提出的一个深度学习模型架构。它的出现彻底改变了序列处理任务（尤其是自然语言处理）的技术格局，并成为当今几乎所有大规模语言模型（如 **[GPT-4](./Lecture1-GPT-4.md)**, **[BERT](./Lecture1-BERT.md)**）的基石。
Transformer 的核心思想是：**完全摒弃循环神经网络（RNN）的顺序依赖结构，仅依赖注意力机制（Attention Mechanism）来捕捉序列中的长距离依赖关系。**
这种设计带来了两大优势：
1.  **强大的并行计算能力:** 由于没有 RNN 的时序依赖，模型可以同时处理整个输入序列的所有 token，极大地提高了训练效率。
2.  **卓越的长距离依赖捕捉能力:** **[注意力机制](./Lecture1-Self-Attention.md)**允许模型直接计算序列中任意两个位置之间的关联度，路径长度为 O(1)，从而有效解决了 RNN 难以捕捉长距离依赖的问题。
## 2. 核心架构
Transformer 模型主要由编码器（Encoder）和解码器（Decoder）两部分组成，每一部分都是由多个相同的层（Block）堆叠而成。
### 2.1 编码器 (Encoder)
每个编码器层包含两个核心子层：
1.  **多头自注意力层 (Multi-Head Self-Attention):** 这是模型的核心。它让输入序列中的每个 token 都能“关注”到序列中的所有其他 token，并根据相关性计算出一个加权的上下文表示。所谓“多头”，是指并行地运行多个独立的**[注意力机制](./Lecture1-Self-Attention.md)**，每个“头”可以学习到不同方面的依赖关系（如语法、语义等），然后将结果拼接起来，增强了模型的表达能力。
2.  **位置前馈网络 (Position-wise Feed-Forward Network):** 这是一个简单的全连接前馈网络，它被独立地应用于每个 token 的表示上。它为模型引入了非线性，增加了模型的深度和复杂性。
每个子层之后都跟着一个残差连接（Residual Connection）和层归一化（Layer Normalization，如 **[RMSNorm](./Lecture1-RMSNorm.md)**），这对于训练深度网络至关重要。
### 2.2 解码器 (Decoder)
每个解码器层包含三个核心子层：
1.  **掩码多头自注意力层 (Masked Multi-Head Self-Attention):** 与编码器中的自注意力层类似，但增加了一个“掩码”（Mask）。在生成第 `t` 个 token 时，掩码会阻止模型看到 `t` 位置之后的信息，确保了模型的自回归（auto-regressive）特性，即预测只能依赖于已生成的部分。
2.  **编码器-解码器注意力层 (Encoder-Decoder Attention):** 这一层允许解码器的每个 token “关注”编码器输出的所有 token。这是连接编码器和解码器的桥梁，让解码器在生成输出时能够充分利用输入序列的信息。
3.  **位置前馈网络 (Position-wise Feed-Forward Network):** 与编码器中的相同。
### 2.3 位置编码 (Positional Encoding)
由于**[注意力机制](./Lecture1-Self-Attention.md)**本身不包含位置信息（它是一种集合操作），为了让模型理解 token 的顺序，必须显式地引入位置信息。原始 Transformer 使用正弦和余弦函数来创建**[位置编码](./Lecture1-Positional-Encoding.md)**向量，并将其加到输入嵌入上。现代模型则更多采用如**[旋转位置编码 (RoPE)](./Lecture1-Rotary-Positional-Embeddings.md)**等更先进的方法。
## 3. 关键的架构演进
自 2017 年以来，原始的 Transformer 架构经历了一系列重要改进，这些改进共同构成了现代大语言模型的基础：
*   **归一化位置:** 从 Post-LN (在残差连接后) 改为 Pre-LN (在残差连接前)，使训练更稳定。
*   **归一化方法:** 从 LayerNorm 演进为更高效的 **[RMSNorm](./Lecture1-RMSNorm.md)**。
*   **激活函数:** 从 ReLU 演进为 **[SwiGLU](./Lecture1-SwiGLU.md)** 等更平滑、表现更好的函数。
*   **位置编码:** 从绝对正弦编码演进为相对的**[旋转位置编码 (RoPE)](./Lecture1-Rotary-Positional-Embeddings.md)**。
*   **专家混合:** 在前馈网络层引入**[混合专家模型 (MoE)](./Lecture1-Mixture-of-Experts.md)**，以极低的计算成本大幅增加模型参数量。
## 4. 影响力
Transformer 不仅仅是一个模型，它是一种全新的、基于注意力的建模范式。它的成功证明了，通过大规模并行计算和有效的长距离依赖建模，可以构建出前所未有强大的**[语言模型](./Lecture1-Language-Models.md)**。