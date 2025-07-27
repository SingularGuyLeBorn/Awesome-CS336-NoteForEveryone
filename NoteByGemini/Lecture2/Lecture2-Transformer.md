# 专题笔记: Transformer

### 1. 核心概念

**Transformer** 是 Google 在 2017 年的论文《Attention Is All You Need》中提出的一种革命性的深度学习模型架构. 它的出现彻底改变了自然语言处理(NLP)领域,并成为当今几乎所有大型语言模型(LLM),如 GPT、BERT、Llama 等的基础. 

Transformer 最核心的创新在于,它**完全摒弃了传统的循环(Recurrence, 如RNN/LSTM)和卷积(Convolution, 如CNN)结构**,仅依赖于一种名为**自注意力(Self-Attention)**的机制来处理序列数据. 

### 2. Transformer 的关键优势

*   **强大的并行计算能力**: 传统的 RNN 需要按顺序逐个处理序列中的 token,这限制了其并行能力. 而 Transformer 的自注意力机制可以一次性计算出序列中所有 token 之间的相互关系,使得模型能够充分利用现代 GPU 的大规模并行计算能力,极大地提升了训练效率. 
*   **出色的长距离依赖建模**: RNN 在处理长序列时,信息在传递过程中容易丢失,导致难以捕捉长距离的依赖关系(例如,一句话开头和结尾的词语关联). Transformer 通过自注意力机制,可以直接计算任意两个位置之间的关联强度,使得任意两个 token 之间的路径长度都是 O(1),从而完美地解决了长距离依赖问题. 

### 3. 核心架构: 编码器-解码器 (Encoder-Decoder)

原始的 Transformer 模型遵循一个经典的**编码器-解码器**架构,主要用于序列到序列(Seq2Seq)任务,如机器翻译. 

![Transformer Architecture](https://raw.githubusercontent.com/g-make-ai/asset-storage/main/CS336/transformer_architecture.png)

#### a. 编码器 (Encoder)
*   **作用**: 负责接收并“理解”输入序列(例如,源语言句子),将其编码成一系列富含上下文信息的向量表示(embeddings). 
*   **结构**: 由 N 个相同的编码器层堆叠而成. 每个编码器层包含两个核心子层: 
    1.  **多头自注意力层 (Multi-Head Self-Attention Layer)**: 这是模型的核心. 它让输入序列中的每个 token 都能“关注”到序列中的所有其他 token,并计算出它们之间的相关性得分,然后根据这些得分加权聚合所有 token 的信息,从而生成新的、包含了上下文信息的 token 表示. 
    2.  **位置前馈网络 (Position-wise Feed-Forward Network)**: 一个简单的全连接前馈网络,被独立地应用于每个 token 的表示上,为其增加非线性变换能力. 
*   每个子层后面都跟着一个**残差连接 (Residual Connection)** 和**层归一化 (Layer Normalization)**,这对于训练深度 Transformer 模型至关重要,可以有效防止梯度消失. 

#### b. 解码器 (Decoder)
*   **作用**: 接收编码器的输出和已经生成的部分目标序列,然后自回归地(autoregressively)生成下一个目标 token. 
*   **结构**: 同样由 N 个相同的解码器层堆叠而成. 每个解码器层比编码器层多一个子层: 
    1.  **带掩码的多头自注意力层 (Masked Multi-Head Self-Attention Layer)**: 与编码器中的自注意力类似,但增加了一个“掩码(mask)”. 这个掩码确保了在预测第 `i` 个位置的 token 时,只能关注到它前面(1 到 `i-1`)已经生成的 token,而不能“看到”未来的 token. 这是保证自回归生成过程正确的关键. 
    2.  **编码器-解码器注意力层 (Encoder-Decoder Attention Layer)**: 这是连接编码器和解码器的桥梁. 它允许解码器中的每个 token “关注”编码器输出的所有 token 表示,从而从源序列中提取最相关的信息来指导目标序列的生成. 
    3.  **位置前馈网络**: 与编码器中的完全相同. 
*   同样,每个子层后也有残差连接和层归一化. 

### 4. 核心组件详解

*   **自注意力机制 (Self-Attention)**: 对于每个输入 token,自注意力机制会创建三个向量: **查询(Query)**、**键(Key)**和**值(Value)**. 它通过计算一个 token 的 Query 向量与所有其他 token 的 Key 向量的点积来得到注意力得分,然后用 Softmax 对得分进行归一化,最后将这些归一化的得分作为权重,对所有 token 的 Value 向量进行加权求和,得到该 token 的新表示. 
*   **多头注意力 (Multi-Head Attention)**: 与其只进行一次自注意力计算,不如将 Query、Key、Value 向量线性投影到多个独立的、更低维度的“头(head)”中,并行地在每个头中执行自注意力计算,然后将所有头的结果拼接起来并再次进行线性投影. 这使得模型能够同时从不同的表示子空间中学习信息. 
*   **位置编码 (Positional Encoding)**: 由于自注意力机制本身不包含任何关于序列顺序的信息,Transformer 必须额外引入位置信息. 这是通过在输入 embedding 中加入一个**位置编码向量**来实现的. 原始论文使用 `sin` 和 `cos` 函数来生成这种独特的、能够表示绝对和相对位置的编码. 

### 5. Transformer 的演进

原始的 Encoder-Decoder 架构后来演化出了两种主流的变体: 
*   **仅编码器模型 (Encoder-only)**: 如 BERT、RoBERTa. 这类模型擅长理解上下文,非常适合做自然语言理解(NLU)任务,如文本分类、命名实体识别. 
*   **仅解码器模型 (Decoder-only)**: 如 GPT 系列、Llama. 这类模型是强大的文本生成器,通过自回归的方式生成连贯的文本,成为当今大语言模型的主流架构. 

---
**关联知识点**
*   [模型并行 (Model Parallelism)](./Lecture2-Model-Parallelism.md)
*   [FLOPS (浮点运算)](./Lecture2-FLOPS.md)