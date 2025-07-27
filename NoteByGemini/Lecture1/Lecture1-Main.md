# CS336 - 讲座 1：课程概述与 Tokenization
## 前言：为何要从零构建？
欢迎来到 CS336 课程！本课程旨在带领大家端到端地体验构建[语言模型](./Lecture1-Language-Models.md)的全过程,内容涵盖数据、系统和建模. 
  
我们正处在一个特殊的时代. 研究人员与底层技术的联系日益脱节. 过去,研究者会亲手实现和训练自己的模型. 如今,许多人只需调用专有模型的 API 即可. 这本身并非坏事,更高层次的抽象能让我们完成更多工作. 然而,与编程语言或操作系统不同,[语言模型](./Lecture1-Language-Models.md)的抽象是“泄露的”. 我们并不完全理解其内部机制. 要进行基础性研究,就必须深入技术栈的内部,协同设计数据、系统和模型的各个方面. 
  
本课程的理念是：**要理解它,就必须亲手构建它. **
  
但这面临一个巨大挑战：[语言模型](./Lecture1-Language-Models.md)**的工业化. 像** [GPT-4](./Lecture1-GPT-4.md) **这样的前沿模型,据说拥有 1.8 万亿参数,训练成本高达数亿美元. 对于学术界而言,复现这样的**[开放与闭源模型](./Lecture1-Open-vs-Closed-Models.md)**中的前沿模型是不现实的. 我们只能构建小型模型,但这可能无法完全代表大规模模型的行为. 例如,小模型与大模型在计算瓶颈(Attention vs MLP)上存在差异,更重要的是,许多关键能力,如上下文学习,是只在巨大规模下才会出现的**[涌现能力](./Lecture1-Emergent-Behavior.md). 
  
尽管如此,我们依然能学到宝贵知识,主要集中在三个层面：
1. **机制 (Mechanics):** 事物如何工作的原理. 例如,[Transformer](./Lecture1-Transformer.md)**的结构,或**[模型并行](./Lecture1-Model-Parallelism.md)如何高效利用 GPU. 
2. **心态 (Mindset):** 像 OpenAI 的先驱者那样,认真对待规模化,并致力于榨干硬件的每一分性能. 这是一种追求极致效率的思维方式,它比任何具体技术都更重要. 
3. **直觉 (Intuitions):** 关于哪些数据和模型决策能带来好模型. 这部分我们只能部分传授,因为在不同规模下,最优解可能不同. 
  
   许多人误解了 Rich Sutton 的[《The Bitter Lesson》](./Lecture1-The-Bitter-Lesson.md),认为它意味着“规模就是一切,算法不重要”. 正确的解读是：**规模化的算法才是关键**. 模型的最终精度是效率和资源的乘积. 效率的提升,尤其是在算法层面,其速度甚至超过了摩尔定律. 因此,我们课程的核心问题是：**在给定的计算和数据预算下,如何构建最好的模型？**
## 课程核心支柱
本课程围绕五大支柱展开,它们共同构成了构建[语言模型](./Lecture1-Language-Models.md)的完整流程. 
#### 1. 基础 (Basics)
目标是打通一个最基础的端到端训练流程. 这包括实现：
- [Tokenization (令牌化)](./Lecture1-Tokenization.md)**:** 将原始文本字符串转换为模型可以处理的整数序列. 我们将重点学习和实现目前仍被广泛使用的[字节对编码 (BPE)](./Lecture1-Byte-Pair-Encoding.md)算法. 
- **模型架构:** 我们将从最初的 [Transformer](./Lecture1-Transformer.md) 架构出发,并逐步引入一系列自 2017 年以来的重要改进,例如 [SwiGLU](./Lecture1-SwiGLU.md) 激活函数、[旋转位置编码 (RoPE)](./Lecture1-Rotary-Positional-Embeddings.md) 和 [RMSNorm](./Lecture1-RMSNorm.md) 等. 
- **训练:** 使用 [AdamW 优化器](./Lecture1-AdamW-Optimizer.md) 设置一个完整的训练循环. 
#### 2. 系统 (Systems)
目标是榨干硬件的每一分性能. 
- [GPU 核函数 (Kernels)](./Lecture1-GPU-Kernels.md)**:** 深入 GPU 硬件层面,理解数据流动的瓶颈. 我们将使用 [Triton](./Lecture1-Triton.md) 语言编写自定义核函数,以最小化数据移动,最大化计算效率. 像 [FlashAttention](./Lecture1-FlashAttention.md) 这样的技术就是这一思想的极致体现. 
- **并行计算:** 当模型大到单张 GPU 无法容纳时,就需要[模型并行](./Lecture1-Model-Parallelism.md)、数据并行等技术,将计算任务协同地分布在数百甚至数千张 GPU 上. 
- [推理 (Inference)](./Lecture1-Inference.md)**:** 模型的训练是一次性的,而推理成本则与使用次数成正比. 我们将学习如何优化推理过程,包括像[推测解码](./Lecture1-Speculative-Decoding.md)这样的前沿技术. 
#### 3. 伸缩法则 (Scaling Laws)
目标是在小规模实验的指导下,预测大规模训练的结果. 
- [伸缩法则](./Lecture1-Scaling-Laws.md)是连接小规模实验和大规模部署的桥梁. 它帮助我们回答一个核心问题：给定一个计算预算,模型应该设多大？数据应该用多少？DeepMind 的 [是连接小规模实验和大规模部署的桥梁. 它帮助我们回答一个核心问题：给定一个计算预算,模型应该设多大？数据应该用多少？DeepMind 的 ](./Lecture1-Chinchilla-Optimal.md) 提供了著名的“20:1”规则,即模型参数量的 20 倍约等于最佳训练数据量. 
#### 4. 数据 (Data)
“Garbage in, garbage out.” 数据是模型的灵魂. 
- **评估:** 我们如何判断一个模型的好坏？除了标准的[困惑度 (Perplexity)](./Lecture1-Perplexity.md),还有 MMLU 等标准化测试. 
- [数据管理](./Lecture1-Data-Curation.md)**:** 人们常说“模型在互联网上训练”,这是一个巨大的误解. 高质量的数据需要主动获取、清洗、去重和过滤. 我们将深入研究如何处理像 [Common Crawl](./Lecture1-Common-Crawl.md) 这样的原始网络数据,将其转化为模型能够学习的高质量文本. 
#### 5. 对齐 (Alignment)
目标是让一个只会“文字接龙”的基础模型变得有用、听话且安全. 
- [监督式微调 (SFT)](./Lecture1-Supervised-Fine-Tuning.md)**:** 通过高质量的“指令-回答”对,教会模型遵循人类的指令. 
- [从人类反馈中强化学习 (RLHF)](./Lecture1-RLHF.md)**:** 当 SFT 不足时,可以利用更轻量级的反馈信号(如偏好数据 A>B)来进一步优化模型. 这其中涉及 [PPO](./Lecture1-PPO.md) 和更简单的 [DPO](./Lecture1-DPO.md) 等算法. 
## 核心技术：Tokenization
[Tokenization (令牌化)](./Lecture1-Tokenization.md) 是将原始文本(一串字符)转换为模型能理解的数字序列(一串整数/token)的过程. 
#### 1. 朴素方法及其缺陷
- **基于字符 (Character-based):** 将每个 Unicode 字符映射到一个整数. 问题在于词汇表可能非常大,且无法有效利用常见字符组合. 
- **基于字节 (Byte-based):** 将文本编码为 UTF-8 字节序列. 词汇表大小固定为 256. 虽然优雅,但会导致序列过长(压缩率仅为 1),对于依赖[注意力机制](./Lecture1-Self-Attention.md)的 [Transformer](./Lecture1-Transformer.md) 来说,计算成本是灾难性的. 
- **基于词 (Word-based):** 用空格或正则表达式分割单词. 问题在于会遇到大量未登录词(Out-of-Vocabulary, OOV),导致模型无法处理新词. 
#### 2. 字节对编码 (Byte-Pair Encoding, BPE)
为了解决上述问题,现代[语言模型](./Lecture1-Language-Models.md)(从 [GPT-2](./Lecture1-GPT-4.md) 开始)普遍采用 [BPE](./Lecture1-Byte-Pair-Encoding.md) 或其变体. 
  
[BPE](./Lecture1-Byte-Pair-Encoding.md) 是一种数据压缩算法,其核心思想是：**迭代地合并最常见的一对相邻 token**. 
  
**训练过程：**
1. **初始化:** 将文本转换为字节序列. 此时,词汇表就是所有可能的 256 个字节. 
2. **迭代合并:**
  
   a. 统计当前文本序列中所有相邻 token 对的出现频率. 
  
   b. 找到频率最高的一对(例如,`'t'` 和 `'h'` 对应的字节经常一起出现). 
  
   c. 将这对 token 合并成一个新的 token(例如,`'th'`),并将其加入词汇表. 
  
   d. 在整个文本中,用这个新 token 替换所有出现的原始 token 对. 
3. **重复:** 重复步骤 2,直到达到预设的词汇表大小. 
  
   通过这种方式,[BPE](./Lecture1-Byte-Pair-Encoding.md) 能够根据语料库的统计特性,自适应地学习出一个词汇表. 常见词(如 "hello")会被表示为单个 token,而罕见词或生造词(如 "Supercalifragilisticexpialidocious")则会被拆分为多个 subword token(如 "Super", "cali", "fragi", ...),从而完美地解决了 OOV 问题,同时保持了较高的压缩率. 
## 总结与展望
本节课我们鸟瞰了从零开始构建一个现代[语言模型](./Lecture1-Language-Models.md)**所需的技术全景,并深入探讨了第一步——**[Tokenization](./Lecture1-Tokenization.md). 我们理解到,每一个设计决策,从架构选择到数据处理,最终都服务于一个核心目标：**在有限的资源下最大化效率**. 
  
下一讲,我们将深入 PyTorch,学习如何精确地核算模型训练中的资源消耗,为后续的系统优化打下坚实的基础. 
***
### 拓展阅读
为了更好地消化本讲内容并为后续课程做准备,我们为您规划了一条战略性的学习路径. 
  
**推荐阅读策略：**
1. **奠定基石 (Laying the Foundation):** 首先,请确保您对现代语言模型的核心架构有清晰的认识. 这是理解一切后续内容的基础. 
   - 阅读 [Transformer](./Lecture1-Transformer.md) 笔记,了解其整体架构. 
   - 深入钻研 [注意力机制 (Self-Attention)](./Lecture1-Self-Attention.md),这是 Transformer 的心脏. 
   - 学习 [位置编码 (Positional Encoding)](./Lecture1-Positional-Encoding.md),理解模型如何处理序列顺序. 
2. **理解“规模”的教训 (Understanding the Lessons of Scale):** 现代 LLM 的发展离不开对“规模”的深刻洞见. 
   - 阅读 [《The Bitter Lesson》](./Lecture1-The-Bitter-Lesson.md),理解为何通用、可扩展的方法最终会胜出. 
   - 学习 [伸缩法则 (Scaling Laws)](./Lecture1-Scaling-Laws.md) 和 [Chinchilla Optimal](./Lecture1-Chinchilla-Optimal.md),了解模型性能如何随规模增长而变化,以及如何平衡模型大小与数据量. 
3. **深入核心算法 (Diving into Core Algorithms):** 掌握本讲介绍的关键技术. 
   - 仔细阅读 [Tokenization (令牌化)](./Lecture1-Tokenization.md) 和 [字节对编码 (BPE)](./Lecture1-Byte-Pair-Encoding.md),这是数据预处理的第一步. 
   - 回顾 [N-gram 模型](./Lecture1-N-gram-%E6%A8%A1%E5%9E%8B.md) 和 [Seq2Seq 模型](./Lecture1-Seq2Seq-%E6%A8%A1%E5%9E%8B.md),了解 BPE 和 Transformer 出现之前的技术演进. 
4. **探索前沿与未来 (Exploring the Frontier & Future):** 了解当前的研究热点和行业格局. 
   - 阅读 [开放与闭源模型](./Lecture1-Open-vs-Closed-Models.md),思考两种模式的利弊和未来走向. 
   - 了解 [推理 (Inference)](./Lecture1-Inference.md) 优化技术,如 [推测解码](./Lecture1-Speculative-Decoding.md),这在模型部署中至关重要. 
   - 关注 [从人类反馈中强化学习 (RLHF)](./Lecture1-RLHF.md),这是让模型与人类价值观对齐的关键技术. 
