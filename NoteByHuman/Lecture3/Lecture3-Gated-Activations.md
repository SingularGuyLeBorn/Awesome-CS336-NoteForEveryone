# 精英笔记:门控激活 (GLU)

在Transformer的前馈网络(Feed-Forward Network, FFN)中,激活函数的选择对模型性能有重要影响. 虽然ReLU和GeLU等简单非线性函数在早期模型中被广泛使用,但现代大型语言模型几乎一致地转向了**门控线性单元(Gated Linear Units, GLU)**及其变体,如SwiGLU和GeGLU.

### 1. 从简单非线性到门控机制

一个标准的FFN层(无偏置项)通常如下:

`FFN(x) = Activation(x * W1) * W2`

其中 `Activation` 可以是 ReLU, GeLU等. 这里的 `x * W1` 将输入`x`从`d_model`维度映射到中间维度`d_ff`.

GLU的核心思想是,不应让所有信息都无差别地通过非线性激活,而应引入一个**数据依赖的门(gate)**来动态地控制信息流.

### 2. GLU的结构与变体

GLU通过引入第二个线性变换来创建这个门. 它将标准的FFN结构修改为:

`FFN_GLU(x) = (Activation(x * W) ⊗ (x * V)) * W2`

这里的符号 `⊗` 代表逐元素相乘(element-wise multiplication).

![](https://storage.googleapis.com/static.a-b-c/project-daedalus/L3-P22.png)

让我们拆解这个公式:

- `x * W`: 和标准FFN一样,这是主要的线性变换.
- `x * V`: 这是一个新增的、并行的线性变换,其输出维度与 `x * W` 相同. 这个输出就是所谓的“门”.
- `⊗`: `x * W` 的结果会与“门” `x * V` 进行逐元素相乘. 这意味着,`x * V` 中的每个元素都决定了 `x * W` 中对应位置的元素有多大比例可以通过. 如果门控值为0,信息就被完全阻断;如果为1,信息则完全通过.

由于这个门是输入 `x` 的函数,因此它是**动态的**,使得网络可以根据当前输入内容,选择性地激活或抑制FFN中的某些神经元.

根据 `Activation` 函数的不同,GLU有多种流行的变体:

- **SwiGLU**: `Activation` 使用Swish函数 (`swish(x) = x * sigmoid(x)`). 这是目前**最流行**的选择,被Llama系列、PaLM、Mistral等广泛采用.
- **GeGLU**: `Activation` 使用GeLU函数. 被T5-v1.1、LaMDA、Gemma等模型使用.
- **ReGLU**: `Activation` 使用ReLU函数.

### 3. 为什么门控机制有效？

门控机制的成功并非偶然,它在深度学习的许多领域(如LSTM、GRU)都证明了其价值.

- **增强的表达能力**: 门控为网络提供了更复杂的、非线性的交互方式. 模型可以学习到更精细的特征组合,而不仅仅是简单的激活或不激活.
- **动态信息路由**: 模型可以根据上下文决定哪些信息是重要的,应该被保留和传递,哪些是噪音,应该被抑制. 这对于处理复杂且多变的语言现象尤为重要.
- **缓解梯度问题**: 像sigmoid这样的门控函数(在SwiGLU中使用)可以将梯度平滑地缩放到0和1之间,可能有助于更稳定的训练.

经验证据非常充分. 多项研究(如Shazeer 2020, Narang et al. 2020)的评估结果一致表明,GLU变体(特别是ReGLU和SwiGLU)在各种NLP任务上都优于非门控的对应版本.

![](https://storage.googleapis.com/static.a-b-c/project-daedalus/L3-P25.png)

### 4. 实现中的考量:参数量匹配

由于GLU引入了额外的参数矩阵`V`,为了与非门控的FFN保持大致相同的参数总量,通常需要调整中间维度 `d_ff`.

一个常见的做法是,如果非门控FFN的中间维度是`d_ff`,那么在使用GLU时,矩阵`W`和`V`的输出维度会设为 `(2/3) * d_ff`. 这样,总的参数量 (`d_model * (2/3)*d_ff * 2`) 就约等于非门控版本 (`d_model * d_ff`).

这就是为什么当讨论超参数时,GLU模型的`d_ff / d_model`比例通常是`(8/3) ≈ 2.67`,而不是非门控模型的`4`. 这个细节对于公平比较不同架构和复现论文结果至关重要.