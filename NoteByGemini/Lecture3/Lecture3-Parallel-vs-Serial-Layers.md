### 专题笔记:并行层与串行层 (Parallel vs Serial Layers)

#### 1. 定义

并行层与串行层描述的是 **[Transformer 架构](./Lecture3-Transformer-Architecture.md)**内部,注意力(Attention)子层和前馈网络(FFN)子层之间两种不同的计算流组织方式. 

##### **A. 串行层 (Serial Layers)**

-   **计算流**:这是 **[Transformer 原始论文](./Lecture3-Attention-Is-All-You-Need.md)**中提出的标准设计. 在一个 Transformer 块中,输入首先流经注意力子层,其输出再作为输入流经 FFN 子层. 计算是顺序进行的. 
-   **公式 (Pre-Norm 形式)**:
    1. `x_attn = x + Attention(LayerNorm(x))`
    2. `x_ffn = x_attn + FFN(LayerNorm(x_attn))`
-   **特点**:
    -   **表达能力**:理论上,由于是函数的复合(Composition),`FFN(Attention(x))` 的表达能力可能比两者的简单相加更强. 
    -   **主流选择**:绝大多数现代模型,包括 **[LLaMA](./Lecture3-LLaMA-Architecture.md)** 系列和 GPT 系列,都采用串行层结构. 

##### **B. 并行层 (Parallel Layers)**

-   **计算流**:在这种设计中,注意力和 FFN 子层接收相同的输入,并同时进行计算. 然后,将它们的输出直接相加到原始的输入残差上. 
-   **公式 (Pre-Norm 形式)**:
    `x_out = x + Attention(LayerNorm(x)) + FFN(LayerNorm(x))`
-   **特点**:
    -   **并行效率**:理论上可以提升计算效率. 因为注意力和 FFN 的计算可以并行执行,并且它们共享同一个 `LayerNorm(x)` 的输入,可以减少一次 LayerNorm 计算. 更重要的是,在一些硬件上,两个并行的矩阵乘法可以被融合成一个更高效的计算核(Fused Kernel),从而提升 GPU 的利用率和训练速度. 
    -   **应用案例**:由 GPT-J 首次引入,并在 Google 的 PaLM 模型中得到了大规模应用. 近期的 Cohere Command 系列和 Falcon 2 模型也采用了并行层设计. 

#### 2. 对比与权衡

| 特性 | 串行层 (Serial) | 并行层 (Parallel) |
| :--- | :--- | :--- |
| **计算流** | 顺序:Attention -> FFN | 并行:Attention + FFN |
| **表达能力** | 理论上更强(函数复合) | 理论上较弱(函数相加) |
| **计算效率** | 标准 | 更高(可并行、可融合) |
| **应用广泛度** | **非常广泛(业界主流)** | 相对较少,但被一些重要模型采用 |

#### 3. 结论

尽管并行层在理论上具有更高的计算效率优势,并且在一些大规模模型中成功应用,但串行层结构凭借其简单、有效以及可能更强的表达能力,至今仍然是绝大多数大型语言模型的首选. 

在实践中,选择哪种结构取决于具体的性能目标、硬件特性以及对模型表达能力与训练速度之间的权衡. 对于大多数研究者和开发者而言,遵循主流的串行层设计是一个稳妥且经过充分验证的选择. 