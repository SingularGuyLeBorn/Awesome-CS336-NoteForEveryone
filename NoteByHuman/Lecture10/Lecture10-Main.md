# Lecture 10: 推理 (Inference)

**主讲人**: CS336 Instructor
**核心议题**: 模型推理的效率瓶颈、算术强度分析、KV Cache优化架构、非Transformer架构、量化与剪枝、投机采样、系统级优化。

---

## 1. 推理概览 (Landscape)

推理（Inference）是指给定一个**固定的、已训练好的模型**，根据提示词（Prompts）生成响应的过程。它不仅出现在聊天机器人或代码补全（如Cursor）中，还贯穿于模型评估、测试时计算（Test-time compute，即“思考”过程）以及强化学习的采样阶段。

### 1.1 为什么效率至关重要

训练是一次性成本，而推理是重复发生的。

* **OpenAI**: 每天生成约 1000 亿单词。
* **Cursor**: 每天生成约 10 亿行被采纳的代码。

> **OpenAI (Sam Altman)**: "OpenAI now generates about 100 billion words per day. (all people on earth: ~100 trillion)"
>
> **Cursor**: "Cursor users now generate 1 billion lines of accepted code per day."

这些巨大的数字表明，推理成本（Inference Cost）正在随着模型使用量的爆发而线性甚至指数级增长。优化推理效率不仅仅是省钱，更是规模化服务的生存关键。

### 1.2 核心指标 (Metrics)

* **首字延迟 (Time-to-first-token, TTFT)**: 用户在看到任何输出前需要等待的时间。这对交互式应用至关重要。
* **延迟 (Latency, seconds/token)**: Token生成的速率。决定了用户阅读生成的流畅度。
* **吞吐量 (Throughput, tokens/second)**: 系统单位时间内能处理的总Token数。这对于批量处理（Batch Processing）非常关键。**高吞吐量并不等同于低延迟**。

### 1.3 训练与推理的关键区别

* **训练 (Training)**: 你能看到所有 Token，可以在序列维度上并行化（Transformer 的核心优势）。
* **推理 (Inference)**: 必须**顺序生成**（Sequential Generation）。生成第 $t$ 个 Token 依赖于前 $t-1$ 个 Token。这种自回归特性使得推理难以充分利用计算资源，且高度受限于内存。

---

## 2. 理解推理负载：算术强度分析 (Understanding the Workload)

为了理解为什么推理通常是**内存受限（Memory-limited）**而非**计算受限（Compute-limited）**，我们需要引入**算术强度（Arithmetic Intensity）**的概念。

### 2.1 基础概念复习

* **FLOPs**: 浮点运算次数。
* **Bytes Transferred**: 内存读写字节数。
* **算术强度**: $\frac{\text{FLOPs}}{\text{Bytes Transferred}}$。我们希望这个值尽可能高。
* **H100 规格**:
  * 算力: 989 TFLOPs (FP16/BF16)
  * 内存带宽: 3.35 TB/s
  * **加速器强度 (Accelerator Intensity)**: $\approx 295$。这意味着我们需要每传输 1 个字节做至少 295 次运算，才能饱和 H100 的计算能力。

### 2.2 矩阵乘法 vs. 生成过程

在代码 `lecture_10.py` 中，我们定义了符号变量来模拟这一过程。

* **矩阵乘法 (Training/Prefill)**: 算术强度通常为 Batch Size ($B$)。只要 $B > 295$，我们就处于计算受限状态（好）。
* **生成过程 (Generation)**: 当 $B=1$（单请求生成）时，也就是矩阵-向量乘法，算术强度约为 1。
  * **结论**: 1 远小于 295。生成过程严重受限于内存带宽。

**[深入探讨: Transformer推理的算术强度与性能建模](./Lecture10-Arithmetic-Intensity-Analysis.md)**
*(点击上方链接查看基于 SymPy 代码对 MLP 和 Attention 层算术强度的完整数学推导与 FLOPs 计数)*

### 2.3 朴素推理 vs. KV Cache

* **朴素推理**: 每生成一个 Token，都把整个历史序列重新喂给 Transformer。复杂度 $O(T^2)$ 甚至 $O(T^3)$。
* **KV Cache**: 观察到前缀的计算是冗余的。我们将每一层的 Key 和 Value 向量缓存到 HBM（高带宽内存）中。
  * **Prefill (预填充)**: 并行编码 Prompt，计算受限（快）。
  * **Generation (生成)**: 逐个生成，读取 KV Cache，内存受限（慢）。

#### 直观对比 (Visual Intuition)

```text
朴素推理 (Naive Inference):
Step 1: [A] -> Model -> [B]
Step 2: [A, B] -> Model -> [C]       (重复计算 A)
Step 3: [A, B, C] -> Model -> [D]    (重复计算 A, B)
复杂度: O(T^2)

KV Cache 推理:
Step 1: [A] -> Model -> [B]          (存 A 的 KV)
Step 2: [B] -> Model(with Cache A) -> [C]    (只算 B, 存 B 的 KV)
Step 3: [C] -> Model(with Cache A,B) -> [D]  (只算 C, 存 C 的 KV)
复杂度: O(T)
```

*(注: KV Cache 的核心直觉是用空间换时间。所有的自回归生成模型本质上都在维护一个随时间增长的状态)*

### 2.4 推理的两个阶段 (Two Stages)

1. **Prefill**: $T=S$（Prompt长度）。算术强度 $\approx S/2$。容易达到计算受限。
2. **Generation**: $T=1$。算术强度 $< 1$。无法通过增加 Batch Size $B$ 来优化 Attention 层，因为每个序列都有独立的 KV Cache（详见精英笔记分析）。

---

## 3. 走捷径 (Taking Shortcuts): 有损优化 (Lossy)

既然推理是内存受限的，核心思路就是**减少 KV Cache 的大小**。但这可能会降低精度，所以是“有损”的。

### 3.1 减少 KV Cache 大小

内存占用公式（基于 `compute_transformer_stats` 函数）：

$$
\text{Memory} = B \times \text{KV\_Size} + \text{Model\_Params}
$$

$$
\text{KV\_Size} = S \times (K \cdot H) \times L \times 2 \times 2 \text{ (bytes)}
$$

#### A. 分组查询注意力 (**Grouped-Query Attention, GQA**)

* **原理**: 多头注意力 (MHA) 中 $K=N$（Key头数=Query头数）。GQA 让 $K < N$，即多个 Query 头共享一组 Key/Value 头。
* **效果**: KV Cache 减少 $N/K$ 倍。
* **实例**: Llama 2 (70B) 和 Llama 3 全系使用了 GQA。代码模拟显示，使用 GQA 后可以将 Batch Size 从 64 提升到 256，从而显著提升吞吐量。
* **精度**: 几乎无损。

  * **实例**: Llama 2 (70B) 和 Llama 3 全系使用了 GQA。代码模拟显示，使用 GQA 后可以将 Batch Size 从 64 提升到 256，从而显著提升吞吐量。
  * **精度**: 几乎无损。

#### B. 多头潜在注意力 (**Multi-Head Latent Attention, MLA**)

* **来源**: DeepSeek V2/V3。
* **原理**: 不直接减少头的数量，而是将 Key/Value 向量投影到一个低维潜在空间（Latent Space）。MLA 通过低秩压缩（Low-Rank Compression）大幅减少 KV Cache，同时保持了比 MQA/GQA 更强的表达能力。

![DeepSeek MLA Architecture](images/mha_gqa_mla_comparison.svg)

> **图示详细解读**:
>
> * **MHA (左一)**: 每个 Query Head (蓝色) 都有自己独立的 Key/Value Head (黄色/橙色)。存储开销最大 ($d_h n_h$)。
> * **GQA (左二)**: 多个 Query Head 共享一组 Key/Value Head。存储开销降低 ($d_h \cdot \text{group\_size}$)。
> * **MQA (左三)**: 所有 Query Head 共享**唯一**的一组 Key/Value Head。存储开销最小 ($d_h$)，但表达能力受限。
> * **MLA (右一)**: DeepSeek 的核心创新。它不直接存储 Key/Value，而是存储一个**低维压缩向量** (Compressed KV, 灰色细柱)。
>   * 在推理时，这个压缩向量可以被“解压”还原为 Key/Value 参与计算。
>   * **优势**: 实现了比 GQA 更极致的压缩（KV Cache 极小），同时保留了类似 MHA 的多头表达能力（矩阵秩更高）。

#### C. 跨层注意力 (**Cross-Layer Attention, CLA**)

* **原理**: 不仅在头之间共享 KV（如GQA），还在**层（Layers）**之间共享 KV。
* **效果**: 进一步压缩内存，提升了精度/内存的帕累托前沿。

![Cross-Layer Attention](images/cross_layer_attention.png)

> **图示详细解读 (CLA)**:
>
> * **传统 Transformer**: 每一层都有自己独立的 KV Cache。
> * **Cross-Layer Attention (CLA)**: 利用了相邻层之间激活模式的高度相似性。
>   * **机制**: 第 $i$ 层和第 $i+1$ 层共享同一份 Key/Value 矩阵。
>   * **收益**: KV Cache 内存直接减半（2层共享），或者减少更多（多层共享）。这是在不牺牲太多精度的情况下压缩内存的有效手段。

#### D. 局部注意力 (**Local Attention**)

* **原理**: 只关注最近的 $K$ 个 Token。
* **优势**: KV Cache 大小变为 $O(1)$ 常数，与序列长度无关。
* **混合模式**: 如 Character.ai，每 6 层使用 1 层全局注意力，其余为局部注意力。

---

## 4. 替代 Transformer 的架构 (Alternatives to Transformer)

如果 Attention + 自回归本质上就是内存低效的，我们可以换架构吗？

### 4.1 状态空间模型 (**State-Space Models, SSMs**)

* **S4 / Mamba / Jamba**:
  * **思想**: 将 $O(T)$ 的 KV Cache 替换为 $O(1)$ 的固定大小状态（State）。
  * **缺点**: 在“联想回忆（Associative Recall）”任务上表现不佳（例如：根据键找值）。
  * **现状**: **混合架构**是主流。例如 Jamba (Transformer + Mamba) 或 MiniMax-01 (Linear Attention + Full Attention)。仅保留少量 Full Attention 层即可维持高精度。

![S4 Model Architecture](images/s4_architecture.png)

> **图示详细解读 (S4 Model)**:
>
> * **HiPPO Matrix**: 图中通过特定的矩阵结构（HiPPO）将长序列压缩到固定大小的状态中。
> * **连续 -> 离散**: S4 起源于连续时间下的微分方程控制理论，通过离散化（Discretization）应用到深度学习序列建模中。
> * **线性复杂度**: 相比 Transformer 的 $O(N^2)$ (Attention Map)，S4 实现了 $O(N)$ 的推理复杂度。

#### 4.1.1 SSM 的弱点：联想回忆 (Associative Recall)

![Mamba vs Transformer Architecture](images/mamba_ssm_architecture.png)

> **图示详细解读 (SSM Weakness)**:
>
> * **任务**: 联想回忆任务（例如：Given "Key: A, Value: 1", later ask "Key: A", expect "1"）。
> * **Transformer (Attention)**: 就像一本“完美的电话簿”，可以直接查找（Query）历史记录中的任何位置，轻松解决此任务。
> * **传统 SSM**: 状态 $h_t$ 容量有限，随着时间推移，旧信息（Key: A）会被新信息“冲刷”掉（Exponential Decay），导致难以回忆起很久之前的精确关联。
> * **改进**: Mamba 引入了**选择性扫描 (Selective Scan)** 机制，允许模型动态决定“记住”什么和“遗忘”什么，从而部分解决了这个问题。

### 4.2 扩散模型 (**Diffusion Models**)

* **原理**: 不再是自回归（一个接一个），而是**并行生成所有 Token**，然后通过多次迭代“去噪”精炼。
* **优势**: 极高的生成速度（Inception Labs 演示）。
* **潜力**: 彻底改变推理的游戏规则。

![Diffusion-LM Generation](images/diffusion_lm.png)

> **图示详细解读 (Diffusion-LM Process)**:
>
> * **非自回归 (Non-Autoregressive)**: 与 GPT 逐个字生成不同，Diffusion-LM **同时生成所有 Token**。
> * **去噪过程 (Denoising)**:
>   1. **Start**: 从纯高斯噪声开始（图中左侧乱码）。
>   2. **Steps**: 经过几百步的迭代去噪，原本模糊的向量逐渐清晰，坍缩成具体的 Word Embedding。
>   3. **End**: 最终映射回离散的 Token（右侧连贯句子）。
> * **可控性**: 这种生成方式由于是在连续空间操作，非常容易施加额外的约束（如“必须包含某个词”、“必须押韵”等），做**可控生成 (Controllable Generation)** 比自回归模型容易得多。

---

## 5. 量化与剪枝 (Quantization & Pruning)

### 5.1 量化 (**Quantization**)

* **精度**: FP32 (4B) -> BF16 (2B, 推理默认) -> INT8 (1B) -> INT4 (0.5B)。
* **LLM.int8()**:
  * **问题**: 异常值（Outliers）会破坏量化精度。
  * **解法**: 分离异常值用 FP16 计算，其余用 INT8。
* **AWQ (Activation-aware Weight Quantization)**: 根据激活值的重要性来决定哪些权重需要保留高精度。

### 5.2 模型剪枝 (**Model Pruning**)

* **NVIDIA 方法**:
  1. 识别重要的层/头/维度。
  2. 移除不重要的部分。
  3. **蒸馏 (Distillation)**: 用原模型教剪枝后的模型恢复精度。
* **结果**: 15B 参数剪枝到 8B，精度几乎不降。

---

## 6. 使用捷径但反复检查 (Lossless): 投机采样

**投机采样 (Speculative Sampling)** 是一种**无损 (Lossless)** 加速技术。它利用了“验证比生成快”的特性。

* **直觉**: Prefill（验证）是并行的（快），Generation 是串行的（慢）。
* **流程**:

  1. 使用一个小且快的**草稿模型 (Draft Model)** $p$ 快速生成 $K$ 个 Token。
  2. 使用大模型 (Target Model) $q$ 并行验证这些 Token。
  3. 根据特定概率接受或拒绝。
* **核心性质**: **数学上保证**最终输出分布严格等同于大模型 $q$ 的分布。
* **核心性质**: **数学上保证**最终输出分布严格等同于大模型 $q$ 的分布。

![Speculative Decoding Diagram](images/speculative_decoding_cover.png)

> **图示详细解读 (Speculative Decoding Process)**:
> 上图展示了“草稿-验证”的流水线过程：
>
> 1. **Drafting (绿色部分)**: 小模型（Draft Model）快速自回归生成了一串 Token (例如: "The", "quick", "brown", "fox")。因为小模型很快，这一步延迟很低。
> 2. **Verification (蓝色/红色部分)**: 大模型（Target Model）接收这串 Token 作为输入，进行**一次并行前向传播 (One Parallel Forward Pass)**。
>    * 大模型计算出每个位置的真实概率分布 $q(x)$。
>    * **对比**: 将 $q(x)$ 与小模型的分布 $p(x)$ 对比。
> 3. **Outcome (结果)**:
>    * **Accepted (蓝色)**: "The", "quick", "brown" 被大模型认可（接受）。这些 Token 直接输出，无需大模型逐个生成。
>    * **Rejected (红色)**: "fox" 被拒绝。大模型在该位置采样出了 "dog"。
>    * **Correction**: 最终输出序列修正为 "The quick brown dog"，并丢弃后续草稿。使用了 $K=4$ 的算力，实际上生成了 3 个有效 Token，实现了 $3\times$ 加速。

**[深度分析: 投机采样的概率理论与正确性证明](./Lecture10-Speculative-Sampling-Theory.md)**
*(点击上方链接查看关于为何该算法能保证精确采样及其拒绝采样逻辑的数学证明)*

---

## 7. 处理动态负载 (Handling Dynamic Workloads)

在实际服务（Serving）中，请求到达时间不同、长短不一，导致批处理（Batching）非常困难。

### 7.1 连续批处理 (**Continuous Batching**)

* **问题**: 传统的静态 Batching 会因为等待最长序列完成而浪费计算资源。
* **解法**: 迭代级调度 (Iteration-level scheduling)。一旦某个序列生成结束，立刻插入新请求，无需等待整个 Batch 结束。

### 7.2 PagedAttention

* **来源**: vLLM 项目。
* **问题**: 传统内存分配导致严重的**碎片化 (Fragmentation)**。

  * 内部碎片：预分配了 Max Length 但没用完。
  * 外部碎片：不同请求之间的空隙。
* **解法**: 借鉴操作系统的**分页 (Paging)** 机制。

  * 将 KV Cache 切分为非连续的块 (Blocks)。
  * 逻辑上连续，物理上离散。
* **优势**: 内存利用率接近 100%，且支持**前缀共享 (Prefix Sharing)**（如系统提示词共享），通过写时复制 (Copy-on-Write) 实现。

  * 将 KV Cache 切分为非连续的块 (Blocks)。
  * 逻辑上连续，物理上离散。
* **优势**: 内存利用率接近 100%，且支持**前缀共享 (Prefix Sharing)**（如系统提示词共享），通过写时复制 (Copy-on-Write) 实现。

![PagedAttention Animation](images/paged_attention.gif)

> **动画详细解读**:
>
> * **Logical KV Blocks (左侧)**: 模型看到的 KV Cache 是连续的序列 (Token 1, 2, 3...)，被划分为逻辑块 (Logical Block 0, 1...)。
> * **Block Table (中间)**: 类似于操作系统的页表 (Page Table)。它记录了 `Logical Block 0 -> Physical Block 7` 这样的映射关系。
> * **Physical KV Blocks (右侧)**: 在显存 (HBM) 中实际存储数据的地方。注意这些物理块是**完全离散**的，哪里有空位就插哪里。
> * **动态分配**: 当生成新 Token 需要空间时，vLLM 只需要在显存中找一个空闲物理块，填入数据，并更新 Block Table。**没有任何内存需要预留，也没有内存空洞（碎片）。**这是 vLLM 能够支持超大 Batch Size 的根本原因。

---

## 8. 总结 (Summary)

* **重要性**: 推理是模型产生价值的环节，且成本巨大。
* **瓶颈**: 与训练不同，推理（生成阶段）是**内存受限**的，且具有高度动态性。
* **技术栈**:
  * **架构创新**: GQA, MLA, SSMs (Mamba), Diffusion。这是潜力最大的方向。
  * **近似计算**: 量化 (INT8/4), 剪枝。
  * **算法优化**: 投机采样 (Speculative Decoding)。
  * **系统优化**: Continuous Batching, PagedAttention。
