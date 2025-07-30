### 专题笔记:激活函数 (Activation Functions)

#### 1. 核心功能

激活函数是神经网络中的关键组件,它被应用于每个神经元的输出,以引入**非线性 (non-linearity)**. 没有激活函数,一个多层神经网络本质上只是一个线性模型,无法学习复杂的数据模式. 在 **[Transformer 架构](./Lecture3-Transformer-Architecture.md)**中,激活函数主要存在于位置全连接前馈网络(FFN)中. 

#### 2. 关键变体与演进

##### **A. ReLU (Rectified Linear Unit)**

-   **公式**:`ReLU(x) = max(0, x)`
-   **特点**:计算极其简单、高效,能够有效缓解梯度消失问题. 
-   **应用**:是 **[Transformer](./Lecture3-Transformer-Architecture.md)** 原始论文中使用的激活函数,在许多早期模型(如 T5、Gopher)中都有应用. 
-   **缺点**:存在“Dying ReLU”问题,即如果一个神经元的输入恒为负,它将永远不会被激活,相应的权重也无法更新. 

##### **B. GELU (Gaussian Error Linear Unit)**

-   **公式**:`GELU(x) ≈ x * Φ(x)`,其中 `Φ(x)` 是高斯分布的累积分布函数(CDF). 
-   **特点**:GELU 可以看作是 ReLU 的一个平滑近似. 与 ReLU 的硬截断不同,GELU 在零点附近是平滑的,并且允许负值输入产生非零输出. 这种随机正则化的解释使其在实践中表现更好. 
-   **应用**:被 GPT 系列模型(GPT, GPT-2, GPT-3)广泛采用,在 BERT 中也得到了应用. 

##### **C. 门控线性单元 (Gated Linear Units, *GLU) 与 SwiGLU**

门控线性单元是近年来在 LLM 领域取得巨大成功的一类激活函数. 它们不是单一的函数,而是一种结构. 

-   **核心思想**:GLU 通过一个“门控”机制来动态地调节信息流. 标准的 FFN 层是 `FFN(x) = Activation(xW₁)W₂`. 而 GLU 变体将其修改为:
    `FFN_GLU(x) = (Activation(xW₁) ⊗ (xV)) * W₂`
    其中 `xV` 产生一个“门控”向量,通过逐元素乘法 `⊗` 来控制 `Activation(xW₁)` 的输出. 这个额外的门控 `xV` 增加了模型的表达能力,使其能够根据输入内容选择性地让信息通过. 

-   **SwiGLU**:
    -   **构成**:SwiGLU 是 GLU 的一个具体变种,它使用 **Swish** 函数作为其内部的激活函数. Swish 函数的定义是 `Swish(x) = x * sigmoid(x)`. 
    -   **公式**:`SwiGLU(x) = (Swish(xW₁) ⊗ (xV)) * W₂` (在 **[LLaMA](./Lecture3-LLaMA-Architecture.md)** 中,`V` 和 `W₁` 被合并为一个矩阵,`xV` 直接使用 `xW₂` 的另一部分,以提升效率). 在 PaLM 的实现中,`Swish(xW) ⊗ (xV)` 的形式更为常见,其中 `V` 和 `W` 是不同的权重. 简化版的 SwiGLU 则是 `(Swish(xW)) ⊗ (xW₂)`. 更具体来说,在 Llama 中,FFN 的形式是 `FFN(x) = W₂(Swish(xW₁) ⊗ xW₃)`,其中 `W₁` 和 `W₃` 是门控层的两个线性变换. 
    -   **优势**:实践证明,SwiGLU 能够持续且显著地提升模型性能. 它结合了 Swish 的平滑特性和 GLU 的门控能力. 
    -   **应用**:**几乎所有 2023 年后发布的先进模型都采用了 SwiGLU**,包括 **[LLaMA](./Lecture3-LLaMA-Architecture.md)**、PaLM、Mistral 等,它已成为现代 LLM 架构的标配. 

-   **GeGLU**:
    -   与 SwiGLU 类似,只是将内部的 Swish 函数换成了 GELU. 
    -   **应用**:在 **[T5-v1.1](./Lecture3-T5-Model.md)**、LaMDA、Phi-3 等模型中被采用. 

**结论**:从 ReLU 到 GELU,再到 SwiGLU/GeGLU,激活函数的演进体现了对更强表达能力和更优性能的追求. **门控机制**的引入是其中最重要的一步,它已成为现代高性能 LLM 的核心组件之一. 