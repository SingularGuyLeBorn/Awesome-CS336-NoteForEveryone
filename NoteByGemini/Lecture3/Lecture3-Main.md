# 第三讲:你不想知道的关于语言模型架构与训练的一切

## 前言

欢迎来到第三讲. 本次课程我们将深入探讨大型语言模型(LLM)架构与训练中那些通常被忽略的“细枝末节”. 不同于关注顶层理论,我们将采取一种数据驱动、甚至可以说是“演化分析”的视角,通过梳理从 2017 年至今发布的近二十个重要模型的技术报告,来探究哪些架构选择是昙花一现,哪些又经受住了时间的考验,并最终汇聚成了现代 LLM 的主流设计. 

本讲的核心主旨是:**最佳的学习方式是亲身实践,而次佳的方式则是借鉴他人的宝贵经验. ** 我们将一起剖析那些成功模型背后的架构决策,从层归一化、激活函数到超参数设置,为你揭示构建高效、稳定的大型语言模型所需遵循的那些不成文的“潜规则”. 

## 正文

### 1. 从“原始”到“现代”的 Transformer

我们的起点是**[Transformer 架构](./Lecture3-Transformer-Architecture.md)**的经典设计. 你可能在其他课程中已经熟悉了它的结构:输入端带有**[位置编码](./Lecture3-Positional-Embedding.md)**,核心由多头注意力(Multi-Head Attention)和前馈网络(Feed-Forward Network, FFN)堆叠而成,并通过残差连接(Residual Stream)贯穿始终. 在每个子模块之后,会进行一次**[层归一化](./Lecture3-Layer-Normalization.md)**(LayerNorm),这被称为“Post-Norm”结构. 

然而,你在课程作业中实现的并非这个原始版本,而是一个更现代的变体. 主要区别在于:
1.  **Pre-Norm LayerNorm**:**[层归一化](./Lecture3-Layer-Normalization.md)**被移到了每个模块(如注意力和FFN)的前面. 
2.  **RoPE 位置编码**:使用**[旋转位置编码 (Rotary Position Embeddings)](./Lecture3-Positional-Embedding.md)**来注入位置信息. 
3.  **SwiGLU 激活函数**:前馈网络层采用了**[SwiGLU](./Lecture3-Activation-Functions.md)**,而不是传统的 ReLU. 
4.  **无偏置项**:线性层和层归一化层中通常会省略偏置项(bias terms). 

你可能会问,为何要做这些修改？这些看似微小的调整,实际上是过去几年无数模型迭代和实验后沉淀下来的共识. 

### 2. 架构的演进:达成共识与持续探索

通过分析从 GPT、BERT 到 LLaMA、Gemma 等一系列模型,我们可以清晰地看到架构演进的趋势. 

#### 2.1 层归一化:唯一的共识与新变化

- **Pre-Norm vs. Post-Norm**:几乎所有现代模型都达成了共识:**使用 Pre-Norm 结构**. 原始的 Post-Norm 会将 LayerNorm 放在残差连接的主路径上,这可能干扰梯度的直接传播,导致训练不稳定,需要配合精细的学习率预热(Warmup)才能正常工作. 而 Pre-Norm 将 LayerNorm 移出主路径,极大地提升了训练的**[稳定性](./Lecture3-Training-Stability-Tricks.md)**,使得模型可以使用更大的学习率,并且不再强制依赖 Warmup. 
- **RMSNorm**:近年来,**[RMSNorm](./Lecture3-Layer-Normalization.md)**已基本取代了传统的 LayerNorm. RMSNorm 简化了计算,它不计算均值,也不添加偏置项 beta,从而减少了参数和计算量. 尽管这些操作在总浮点运算(FLOPs)中占比极小,但它们涉及大量的内存读写. 在现代 GPU 架构中,**内存带宽是比计算能力更宝贵的资源**,因此 RMSNorm 带来的内存效率提升,转化为了实实在在的训练速度增益,并且模型性能丝毫不受影响. 
- **Double Norm**:近期,如 Grok 和 Gemma 2 等模型引入了“双重归一化”(Double Norm)的概念. 它们不仅在模块前进行 Pre-Norm,还在模块(如 FFN 和注意力层)之后、残差连接相加之前,再增加一个 LayerNorm. 这被认为是进一步提升**[训练稳定性](./Lecture3-Training-Stability-Tricks.md)**的有效手段. 

#### 2.2 激活函数:门控线性单元(GLU)的胜利

- **从 ReLU 到 SwiGLU**:早期的模型多使用 ReLU 或其平滑版本 GELU. 然而,实验和实践一致表明,**[门控线性单元(Gated Linear Units, *GLU)](./Lecture3-Activation-Functions.md)**变体,尤其是 **SwiGLU**,能带来持续且显著的性能提升. 
- **门控机制**:GLU 的核心思想是为前馈网络的隐藏层增加一个“门控”. 它使用输入信号的另一个线性变换来逐元素地调节(“门控”)主路径上的激活值. 这种动态机制增强了模型的表达能力. 几乎所有 2023 年后发布的先进模型,如 **[LLaMA](./Lecture3-LLaMA-Architecture.md)**、PaLM 等,都采用了 SwiGLU 或其类似的 GeGLU. 

#### 2.3 并行层 vs. 串行层

标准的 Transformer 模块是**[串行的](./Lecture3-Parallel-vs-Serial-Layers.md)**:先计算注意力,再计算 FFN. 而 GPT-J 和 PaLM 等模型探索了**[并行层](./Lecture3-Parallel-vs-Serial-Layers.md)**结构,即同时计算注意力和 FFN,然后将两者的输出相加. 理论上,这种并行化可以通过算子融合(Kernel Fusion)提升 GPU 的利用率,从而加速训练. 尽管如此,目前大多数模型仍然采用更传统的串行结构. 

#### 2.4 位置编码:RoPE 的统一

在**[位置编码](./Lecture3-Positional-Embedding.md)**方面,早期存在多种方案,如绝对位置编码、相对位置编码等. 但如今,**旋转位置编码(RoPE)**已成为绝对的主流. 
- **核心思想**:RoPE 的目标是让注意力得分只依赖于词向量的相对位置. 它通过一个巧妙的方式实现:将词向量在不同维度上进行“旋转”,旋转的角度由其绝对位置决定. 由于内积运算对于同步旋转具有不变性,因此任意两个位置的词向量之间的内积(即注意力得分的关键部分)将只与它们的相对位置差有关. 
- **实现方式**:RoPE 在计算 Query 和 Key 向量之后,但在计算注意力得分之前,对它们应用旋转变换. 这一设计已被证明在性能和外推能力(即处理比训练时更长的序列)上都表现出色. 

### 3. 超参数:那些约定俗成的“魔法数字”

在**[模型超参数](./Lecture3-Model-Hyperparameters.md)**的选择上,同样存在着一些惊人的一致性. 

- **FFN 隐藏层大小**:对于非门控的 FFN(如 ReLU),隐藏层维度 `d_ff` 通常是模型维度 `d_model` 的 **4 倍**. 对于门控的 FFN(如 SwiGLU),由于增加了一个门控线性层,为了保持参数量大致不变,这个比例通常调整为 **8/3 ≈ 2.67 倍**. 尽管像 **[T5](./Lecture3-T5-Model.md)** 这样的模型曾大胆尝试过 64 倍的比例,但后续版本也回归到了更标准的设置. 根据 **[Scaling Laws](./Lecture3-Scaling-Laws.md)** 的研究,这个范围(1-10倍)确实是一个性能最优的“甜点区”. 
- **模型纵横比 (`d_model` / `n_layer`)**:模型的深度(层数 `n_layer`)与宽度(模型维度 `d_model`)之间也存在一个最佳平衡点. 过深的模​​型难以并行,延迟高;过宽的模型则可能效率低下. 经验表明,`d_model / n_layer` 的比值在 **100-200** 之间是一个“甜点区”. 例如,GPT-3、LLaMA 等许多成功模型的这个比值都在 128 左右. 
- **词汇表大小**:对于单语(主要是英语)模型,词汇表大小通常在 **30k-50k** 之间. 而对于多语言或生产系统,为了更好地覆盖多种语言和特殊字符(如 emoji),这个数字会显著增加到 **100k-250k**. 

### 4. 正则化与稳定性

- **正则化**:在预训练阶段,模型通常只在海量数据上过一遍(one epoch),过拟合风险很低. 因此,**[Dropout](./Lecture3-Regularization.md)** 已经基本被弃用. 然而,**[权重衰减(Weight Decay)](./Lecture3-Regularization.md)**却被广泛使用. 这并非为了防止过拟合,而是因为它与学习率调度(尤其是余弦退火)之间存在复杂的相互作用,能够在训练后期隐式地加速优化,从而达到更低的训练损失. 
- **训练稳定性**:随着模型规模和训练时长的增加,**[训练稳定性](./Lecture3-Training-Stability-Tricks.md)**变得至关重要. 不稳定的训练表现为损失曲线突然出现尖峰(spike),梯度范数爆炸. 为了解决这个问题,研究者们开发了多种技巧,且大多针对模型中最不稳定的部分——Softmax. 
    - **Z-loss**:应用于输出层的 Softmax. 它通过一个辅助损失项,鼓励 Softmax 的归一化因子 Z 的对数值接近于 0,从而使计算更稳定. 
    - **QK-Norm**:应用于**[注意力机制](./Lecture3-Attention-Variants.md)**中的 Softmax. 在计算 Query 和 Key 的内积之前,对它们分别进行 LayerNorm. 这能有效控制输入到 Softmax 的数值范围,防止其因数值过大而行为异常. 
    - **Logit Soft-capping**:另一种控制注意力 logits 的方法,通过 `tanh` 函数对 logits 进行“软裁剪”,限制其最大值. 

### 5. 注意力机制的现代变体

- **MQA & GQA**:在推理(Inference)阶段,逐个生成 token 的方式使得**[KV 缓存](./Lecture3-Attention-Variants.md)**成为性能瓶颈. **[多查询注意力(MQA)和分组查询注意力(GQA)](./Lecture3-Attention-Variants.md)**应运而生. MQA 让所有头共享同一份 Key 和 Value,而 GQA 则是介于标准多头注意力和 MQA 之间的折中方案. 它们通过减少 KV 缓存的大小,极大地降低了推理时的内存带宽需求,从而提升了吞吐量. 
- **滑动窗口注意力**:为了处理超长上下文,模型如 Mistral 采用了**[滑动窗口注意力(Sliding Window Attention)](./Lecture3-Attention-Variants.md)**. 每个 token 只关注其邻近的一个固定大小的窗口内的其他 token,这使得计算复杂度从二次方降低到线性. 
- **长距离与短距离注意力的结合**:最新的模型如 LLaMA 4 和 Cohere Command A 采用了一种更精巧的混合策略. 它们交替使用两种类型的注意力层:大部分层使用带 RoPE 的滑动窗口注意力来处理**短距离依赖**;每隔几层,则插入一个不带任何位置编码的**全注意力层**来捕捉**长距离依赖**. 这种设计兼顾了效率和处理超长上下文的能力. 

## 拓展阅读

为了更好地消化本讲的内容,我们为您规划了一条学习路径,助您将理论与行业实践融会贯通. 

**推荐阅读策略:**
1.  **建立宏观认知**:首先,回顾**[Transformer 架构](./Lecture3-Transformer-Architecture.md)**的原始设计,并阅读其奠基性论文**[`Attention Is All You Need`](./Lecture3-Attention-Is-All-You-Need.md)**,理解其核心思想. 
2.  **把握现代脉络**:接着,深入研究**[LLaMA 架构](./Lecture3-LLaMA-Architecture.md)**的笔记. LLaMA 系列代表了当前业界的主流设计范式,理解它有助于你把握现代 LLM 的核心. 同时,可以对照阅读 **[T5 模型](./Lecture3-T5-Model.md)** 的笔记,了解其在超参数选择上的“大胆”创新,这有助于拓宽思路. 
3.  **逐个击破核心技术点**:按照以下顺序,深入探索构成现代 Transformer 的关键技术模块. 这个顺序反映了它们在架构演变中的重要性:
    *   **[层归一化 (Layer Normalization)](./Lecture3-Layer-Normalization.md)**:理解从 Post-Norm 到 Pre-Norm,再到 RMSNorm 的演进,这是稳定训练的基础. 
    *   **[激活函数 (Activation Functions)](./Lecture3-Activation-Functions.md)**:了解为何 SwiGLU 等门控单元能够胜出. 
    *   **[位置编码 (Positional Embedding)](./Lecture3-Positional-Embedding.md)**:探究 RoPE 如何巧妙地解决了相对位置编码问题. 
4.  **关注性能与效率**:最后,将目光投向那些直接影响模型训练和推理效率的设计:
    *   **[训练稳定性技巧 (Training Stability Tricks)](./Lecture3-Training-Stability-Tricks.md)**:了解 Z-loss 和 QK-Norm 如何驯服不稳定的 Softmax. 
    *   **[注意力机制变体 (Attention Variants)](./Lecture3-Attention-Variants.md)**:学习 MQA/GQA 和滑动窗口注意力如何赋能高效推理和超长上下文处理. 

通过这条路径,您不仅能理解“是什么”,更能洞察“为什么”,从而在自己的实践中做出更明智的架构选择. 