# CS336 - 第2讲: 手把手用PyTorch搭建大语言模型

## 前言

在上一讲中,我们宏观地探讨了什么是语言模型、从零开始构建它们的意义以及为何要这样做. 今天,我们将深入实践,亲手搭建一个模型. 我们会系统地学习构建模型所需的 [PyTorch](./Lecture2-PyTorch.md) 核心组件,从基础的[张量(Tensors)](./Lecture2-Tensors.md)出发,逐步构建模型、[优化器(Optimizers)](./Lecture2-Optimizers.md)以及完整的训练循环. 

本讲将贯穿一个核心主题: **效率**. 我们将密切关注如何高效地利用计算资源,特别是内存和算力. 这不仅仅是理论探讨,更是为了培养一种“资源核算”的思维模式. 在当今大模型时代,了解你的模型在硬件上消耗了多少资源,直接关系到项目的成本和可行性. 我们将通过“餐巾纸数学”(Napkin Math)来快速估算训练成本和资源需求,让你对这些庞大的数字有切实的体感. 

## 正文

### 1. 核心心法: 资源估算与效率思维

在深入代码之前,我们先来建立一种“餐巾纸数学”的直觉. 这是一种在几分钟内估算出大模型训练关键指标的能力,例如: 

- **问题1: ** 在1024张 [NVIDIA H100](./Lecture2-NVIDIA-H100.md) 显卡上,用15万亿(trillion)个token训练一个700亿参数的稠密Transformer模型,需要多长时间？

  - **估算逻辑: ** 关键在于一个经验公式: 总计算量([FLOPS](./Lecture2-FLOPS.md))约等于 `6 * 模型参数量 * 训练Token数`. 这个“6”的由来,我们稍后会详细拆解. 知道了总计算量,再除以硬件集群在特定**[模型FLOPS利用率(MFU)](./Lecture2-MFU.md)**下的每日有效算力,就能得出所需天数. 通过这个简单的计算,我们能估算出大约需要144天. 

- **问题2: ** 如果你只有8张H100显卡,并且使用 [AdamW](./Lecture2-Adam-AdamW.md) 优化器,在不使用任何内存优化技巧的情况下,能训练的最大模型是多大？

  - **估算逻辑: ** H100有80GB显存. 对于标准的混合精度训练,每个参数大约需要16个字节来存储模型参数本身、**[梯度](./Lecture2-Gradients.md)**和优化器状态. 用总显存除以单位参数所需内存,就能得出大约可以容纳400亿个参数. 这个估算虽然粗略(忽略了激活值等动态内存开销),但它为我们设定了一个明确的资源边界. 


掌握这种估算能力,意味着你从“实现一个模型”的工程师,转变为“高效设计和部署一个模型”的架构师. 因为在大模型领域,每一分效率的提升,都直接转化为金钱和时间的节省. 

### 2. 内存核算: 张量的成本

深度学习的一切都构建在**[张量](./Lecture2-Tensors.md)**之上,它们是存储参数、梯度、优化器状态和数据的基本单元. 一个张量占用的内存由其元素数量和数据类型共同决定. 

#### 2.1 浮点数精度

- [FP32 (float32)](./Lecture2-FP32-FP16-BF16-FP8.md): 单精度浮点数,是传统的“黄金标准”,拥有32位(4字节). 它提供了较大的动态范围和较高的精度,但内存和计算开销也最大. 
- [FP16 (float16)](./Lecture2-FP32-FP16-BF16-FP8.md): 半精度浮点数,占用16位(2字节),能将内存占用减半,并通常带来更快的计算速度. 但其动态范围非常有限,在训练大型模型时容易出现上溢(overflow)或下溢(underflow)问题,导致训练不稳定. 
- [BF16 (bfloat16)](./Lecture2-FP32-FP16-BF16-FP8.md): 由Google Brain开发,同样占用16位,但它拥有与FP32相同的动态范围(8位指数位),牺牲了部分精度(7位小数位). 实践证明,对于深度学习,动态范围比极高的精度更重要,因此BF16成为了现代大模型训练的主流选择. 
- [FP8](./Lecture2-FP32-FP16-BF16-FP8.md): 最新的8位浮点格式,由NVIDIA在H100中引入,进一步压缩了内存和计算. 它提供了两种变体,分别侧重于动态范围和精度,是追求极致性能的前沿技术. 

#### 2.2 混合精度训练

为了平衡稳定性与效率,业界发展出了**[混合精度训练](./Lecture2-Mixed-Precision-Training.md)**策略. 其核心思想是: 

- **参数和优化器状态**: 使用高精度(如FP32)存储,因为它们需要长期累积信息,对精度要求高. 
- **前向和反向传播计算**: 在计算密集型操作(如矩阵乘法)中,将参数临时转换为低精度(如BF16)进行计算,以利用硬件的加速能力. 计算完成后,梯度会转换回FP32进行累积. 

### 3. 计算核算: 操作的代价

#### 3.1 硬件与数据迁移

默认情况下,PyTorch张量创建于CPU内存中. 为了利用GPU加速,必须显式地将数据从CPU的RAM通过PCIe总线迁移到GPU的HBM(高带宽内存)上. 这个数据传输过程是有成本的,因此在设计流程时,应尽量减少不必要的CPU-GPU数据往返. 

#### 3.2 理解计算量: FLOPS

[FLOPS](./Lecture2-FLOPS.md) (Floating Point Operations) 是衡量计算量的单位. 在深度学习中,绝大多数计算量来自矩阵乘法(MatMul). 一个 `(B, D) x (D, K)` 的矩阵乘法,其计算量约为 `2 * B * D * K` FLOPS. 这里的“2”代表每个元素的乘法和加法操作. 

这个简单的公式是进行“餐巾纸数学”的基石. 对于一个模型,我们可以近似地认为其前向传播(Forward Pass)的计算量为 `2 * P * D_seq`(其中P为参数量,D_seq为序列长度或批次大小),这个结论可以从线性模型推广到 [Transformer](./Lecture2-Transformer.md) 这类更复杂的模型中. 

#### 3.3 模型FLOPS利用率 (MFU)

硬件厂商会提供一个理论上的峰值FLOPS(例如 [NVIDIA H100](./Lecture2-NVIDIA-H100.md) 在BF16下约有1000 TFLOPS/s). 然而,实际应用中由于数据加载、内存访问、通信开销等因素,不可能达到理论峰值. 

[MFU (Model FLOPs Utilization)](./Lecture2-MFU.md) 是一个衡量硬件利用效率的指标,其定义为: 
  
`MFU = (模型实际有效FLOPS / 运行时间) / 硬件理论峰值FLOPS`

一个高于50%的MFU通常被认为是良好的. 如果MFU过低,则说明你的训练流程中存在瓶颈,大部分时间GPU都在空闲等待,而不是在进行有效的计算. 

### 4. PyTorch核心操作与技巧

#### 4.1 张量视图与存储

在PyTorch中,很多操作(如 `transpose`, `view`)并不会创建新的内存拷贝,而是创建一个指向原始数据存储区的“视图”(View). 它们通过修改**[张量步长(Tensor Strides)](./Lecture2-Tensor-Strides.md)**来实现. Stride定义了在内存中为了到达下一个维度元素需要跳过的字节数. 

- **优点**: 视图操作几乎是零成本的,可以自由使用以提高代码可读性. 
- **注意点**: 修改视图会改变原始张量. 此外,某些操作(如 `view`)要求张量是**[内存连续的](./Lecture2-Contiguous-vs-Non-contiguous-Tensors.md)**. 如果一个张量(例如经过转置后)变得不连续,你需要先调用 `.contiguous()` 创建一个内存连续的副本,然后才能继续操作. 

#### 4.2 Einops: 优雅的张量操作

在处理高维张量时,使用索引(如 `-1`, `-2`)来指定维度很容易出错且难以阅读. [Einops (爱因斯坦求和约定)](./Lecture2-Einops.md) 提供了一种更直观、更强大的方式来操作张量. 它使用具名维度来描述操作,例如: 

```python
# 传统方式
output = torch.matmul(x, y.transpose(-2, -1))

# Einops 方式
output = einops.einsum(x, y, 'b seq1 hidden, b seq2 hidden -> b seq1 seq2')
```

Einops的 `rearrange`, `reduce`, `repeat` 等函数极大地提高了代码的可读性和健壮性,是现代深度学习项目中的推荐实践. 

### 5. 构建与训练一个简单模型

#### 5.1 模型定义与参数初始化

在PyTorch中,我们通常通过继承 [nn.Module](./Lecture2-nn-Module.md) 来定义自己的模型. 模型的核心是其可学习的参数(`nn.Parameter`). 

**[参数初始化](./Lecture2-Parameter-Initialization.md)**至关重要. 如果初始化不当(例如,值过大),模型在训练初期可能会梯度爆炸,导致不稳定. 一个常用的策略(如Xavier或He初始化)是将权重根据其输入维度进行缩放,例如除以 `sqrt(input_dim)`,以确保初始输出的方差保持在1附近. 

#### 5.2 梯度计算: 反向传播的成本

训练模型不仅有前向传播的计算成本,还有计算梯度的**[反向传播](./Lecture2-Backpropagation.md)**成本. PyTorch通过 [Autograd](./Lecture2-Autograd.md) 引擎自动完成这个过程. 

一个关键的经验法则是: 

- **前向传播 (Forward Pass)**: 需要约 `2 * N * P` FLOPS (N=数据点数, P=参数量). 
- **反向传播 (Backward Pass)**: 需要约 `4 * N * P` FLOPS. 

因此,一次完整的前向+反向传播,总计算量约为 `6 * N * P` FLOPS. 这解释了我们最初估算公式中“6”的来源. 反向传播的计算量是前向传播的两倍,因为它不仅要计算关于权重的梯度,还要计算关于输入的梯度,以便将梯度链式传播到更早的层. 

#### 5.3 优化器与状态内存

[优化器](./Lecture2-Optimizers.md)(如 [Adam](./Lecture2-Adam-AdamW.md))负责根据计算出的梯度来更新模型参数. 不同的优化器需要额外的内存来存储其状态. 

- [SGD](./Lecture2-Stochastic-Gradient-Descent.md): 最简单,几乎不需要额外内存. 
- [Momentum](./Lecture2-Momentum.md): 需要存储一份动量(与参数大小相同),即每个参数需要额外4字节(FP32). 
- [RMSProp](./Lecture2-RMSProp.md): 需要存储一份梯度的平方均值(额外4字节/参数). 
- [Adam/AdamW](./Lecture2-Adam-AdamW.md): 结合了Momentum和RMSProp,需要同时存储动量和梯度平方均值,因此每个参数需要额外8字节. 

综上,对于一个使用AdamW和混合精度训练的模型,每个参数总的内存占用大致为: 

- 参数本身(FP16存储,FP32加载): 4字节
- 梯度(FP16计算,FP32累积): 4字节
- AdamW状态(动量+方差,FP32): 8字节
- **总计: 约16字节/参数**

这就是我们在第二个“餐巾纸数学”问题中使用的数字. 

#### 5.4 训练循环、数据加载与检查点

一个标准的训练循环包含以下步骤: 

1. 从**[数据加载器](./Lecture2-Data-Loading-and-Memmap.md)**获取一个批次的数据. 对于超大规模数据集(如数TB),使用内存映射文件(`memmap`)可以避免一次性将所有数据读入RAM. 
2. 执行模型的前向传播,计算损失. 
3. 调用 `loss.backward()` 执行反向传播,计算梯度. 
4. 调用 `optimizer.step()` 更新模型权重. 
5. 调用 `optimizer.zero_grad()` 清除旧的梯度. 

由于训练过程漫长且可能中断,定期保存**[模型检查点](./Lecture2-Checkpointing.md)**至关重要. 一个完整的检查点应包括: 模型的状态字典、优化器的状态字典以及当前的训练步数/周期数. 

### 6. 展望: 后训练优化与未来方向

当模型训练完成后,为了在推理时达到更高的效率,通常会进行**[模型量化](./Lecture2-Quantization.md)**. 量化是指将模型的浮点数参数(如FP16)转换为更低位的整数(如INT8甚至INT4). 这可以极大地减小模型体积、降低内存带宽需求并利用专门的硬件指令加速推理,尽管这可能会带来轻微的精度损失. 

本讲通过构建一个简单的线性模型,系统地剖析了资源核算的各个方面. 在作业1中,你将把这些概念应用到真正的 [Transformer](./Lecture2-Transformer.md) 模型上,为你深入理解和驾驭大语言模型打下坚实的基础. 未来的话题,如**[模型并行](./Lecture2-Model-Parallelism.md)**和使用 [torch.compile](./Lecture2-torch-compile.md) 进行即时编译,将是在此基础上的进一步延伸. 

***

### 拓展阅读

为了更好地吸收本讲的知识并为后续课程做准备,我们推荐以下阅读策略和资源. 这些笔记由课程内容和我们的主动扩展构成,旨在构建一个更完整的知识体系. 

#### **推荐阅读策略**

1. **第一步: 巩固基石**

   - 首先深入理解 [PyTorch](./Lecture2-PyTorch.md) 的核心哲学以及它与 [NumPy](./Lecture2-NumPy.md) 的关系. 
   - 接着,彻底搞懂 [张量](./Lecture2-Tensors.md) 的内部机制,特别是 [张量步长](./Lecture2-Tensor-Strides.md) 和 [连续与非连续张量](./Lecture2-Contiguous-vs-Non-contiguous-Tensors.md) 的概念,这是理解性能优化的关键. 
   - 最后,学习 [nn.Module](./Lecture2-nn-Module.md)、[反向传播](./Lecture2-Backpropagation.md) 和 [Autograd](./Lecture2-Autograd.md),了解模型构建和自动求导的魔法是如何发生的. 

2. **第二步: 拥抱效率思维**

   - 阅读 [FLOPS](./Lecture2-FLOPS.md) 和 [MFU](./Lecture2-MFU.md) 的笔记,将“餐巾纸数学”内化为你的第二天性. 
   - 研究 [FP32 / FP16 / BF16 / FP8](./Lecture2-FP32-FP16-BF16-FP8.md),了解不同数值精度之间的权衡. 
   - 阅读 [The Bitter Lesson](./Lecture2-The-Bitter-Lesson.md),从哲学层面理解为何拥抱大规模计算是AI发展的必然趋势. 

3. **第三步: 掌握高级工具与技术**

   - 深入探索 [优化器](./Lecture2-Optimizers.md) 的世界,理解从 [SGD](./Lecture2-Stochastic-Gradient-Descent.md) 到 [AdamW](./Lecture2-Adam-AdamW.md) 的演进过程. 
   - 学习 [Einops](./Lecture2-Einops.md),让你的张量操作代码变得优雅而高效. 
   - 了解 [混合精度训练](./Lecture2-Mixed-Precision-Training.md) 和 [torch.compile](./Lecture2-torch-compile.md),这是当前PyTorch性能优化的两大支柱. 

4. **第四步: 面向未来**

   - 最后,阅读关于 [模型并行](./Lecture2-Model-Parallelism.md) 和 [模型量化](./Lecture2-Quantization.md) 的笔记,了解如何将单个GPU上学到的知识扩展到训练和部署真正庞大的模型. 


#### **知识库链接**

- **核心框架与概念**

  - [PyTorch](./Lecture2-PyTorch.md)
  - [张量 (Tensors)](./Lecture2-Tensors.md)
  - [梯度 (Gradients)](./Lecture2-Gradients.md)
  - [反向传播 (Backpropagation)](./Lecture2-Backpropagation.md)
  - [Autograd](./Lecture2-Autograd.md)
  - [nn.Module](./Lecture2-nn-Module.md)
  - [NumPy](./Lecture2-NumPy.md)

- **性能与效率**

  - [NVIDIA H100](./Lecture2-NVIDIA-H100.md)
  - [FLOPS (浮点运算)](./Lecture2-FLOPS.md)
  - [MFU (模型FLOPS利用率)](./Lecture2-MFU.md)
  - [FP32 / FP16 / BF16 / FP8](./Lecture2-FP32-FP16-BF16-FP8.md)
  - [混合精度训练 (Mixed Precision Training)](./Lecture2-Mixed-Precision-Training.md)
  - [张量步长 (Tensor Strides)](./Lecture2-Tensor-Strides.md)
  - [连续与非连续张量](./Lecture2-Contiguous-vs-Non-contiguous-Tensors.md)
  - [torch.compile](./Lecture2-torch-compile.md)
  - [The Bitter Lesson](./Lecture2-The-Bitter-Lesson.md)

- **模型构建与训练技术**

  - [优化器 (Optimizers)](./Lecture2-Optimizers.md)
  - [Adam / AdamW](./Lecture2-Adam-AdamW.md)
  - [随机梯度下降 (SGD)](./Lecture2-Stochastic-Gradient-Descent.md)
  - [Momentum](./Lecture2-Momentum.md)
  - [RMSProp](./Lecture2-RMSProp.md)
  - [Einops (爱因斯坦求和约定)](./Lecture2-Einops.md)
  - [参数初始化 (Parameter Initialization)](./Lecture2-Parameter-Initialization.md)
  - [数据加载与内存映射 (Data Loading & Memmap)](./Lecture2-Data-Loading-and-Memmap.md)
  - [模型检查点 (Checkpointing)](./Lecture2-Checkpointing.md)

- **前沿与未来**

  - [Transformer](./Lecture2-Transformer.md)
  - [模型并行 (Model Parallelism)](./Lecture2-Model-Parallelism.md)
  - [模型量化 (Quantization)](./Lecture2-Quantization.md)
