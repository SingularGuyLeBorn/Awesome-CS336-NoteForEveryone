# 第10讲：大模型推理效率 (Inference Efficiency)

### 前言
在完成了模型的训练后，我们面临着一个全新的挑战：推理（Inference）。与训练不同，推理是给定一个**固定模型**，根据提示生成响应的过程。本讲我们将深入探讨推理工作负载的特殊性，特别是它是如何从训练时的“计算受限”转变为“内存受限”的。我们将结合 **[算术强度](./Lecture10-Code-Intensity.md)** 的理论分析，探讨通过架构调整（如 **[GQA](./Lecture10-Code-Stats.md)**）和算法创新（如 **[投机采样](./Lecture10-Theory-Speculative.md)**）来提升效率的方法。

### 1. 理解推理工作负载
推理不仅仅是聊天机器人的对话，它广泛存在于代码补全、批量数据处理、模型评估，甚至强化学习的样本生成中。

#### 1.1 关键指标
评估推理性能主要关注三个指标：
*   **首词延迟 (TTFT)**：用户等待第一个token生成的时间。
*   **延迟 (Latency)**：生成每个后续token所需的时间（秒/token）。
*   **吞吐量 (Throughput)**：单位时间内系统处理的token总量。

#### 1.2 训练与推理的本质差异
*   **训练**：我们可以看到所有token，因此可以在序列维度上并行计算，这充分利用了GPU的计算能力（Compute-Limited）。
*   **推理**：生成是自回归的（Sequential），为了生成第 $t$ 个token，必须依赖前 $t-1$ 个token。这种逐个生成的特性使得计算无法并行，导致推理过程往往是 **[内存受限](./Lecture10-Theory-ArithmeticIntensity.md)** 的。

### 2. 算术强度与内存墙
为了量化这种限制，我们引入了 **[算术强度](./Lecture10-Code-Intensity.md)** 的概念，即每传输一个字节的数据所进行的浮点运算次数（FLOPs/Byte）。
*   **Prefill 阶段**（处理Prompt）：类似于训练，算术强度高，计算受限。
*   **Generation 阶段**（生成Token）：由于每次只处理一个token，且需要加载巨大的权重和KV Cache，算术强度极低（接近1），甚至远低于硬件的加速器强度。这意味着GPU大部分时间在等待内存传输，而非进行计算。

### 3. KV Cache：推理的显存杀手
为了避免重复计算历史token的表示，我们引入了 **KV Cache**。虽然它将计算复杂度从 $O(T^2)$ 降低到了 $O(T)$，但代价是巨大的显存占用。我们通过 **[Transformer统计模型](./Lecture10-Code-Stats.md)** 可以精确计算出KV Cache随序列长度、层数和Batch Size增长的公式。

### 4. 架构层面的捷径（有损优化）
既然内存是瓶颈，减小KV Cache就是关键。
*   **GQA (Grouped-Query Attention)**：通过让多个查询头（Query Heads）共享一组键值头（KV Heads），显著减少了KV Cache的大小，从而提升吞吐量。
*   **MLA (Multi-head Latent Attention)**：DeepSeek提出的技术，将KV投影到低维潜在空间。
*   **量化 (Quantization)**：将FP16转换为INT8甚至INT4，以精度换速度。

### 5. 算法层面的捷径（无损优化）
我们可以利用“验证比生成快”的非对称特性。
**[投机采样](./Lecture10-Theory-Speculative.md)** 利用一个小型的“草稿模型”快速生成候选token，然后由大模型并行验证。通过巧妙的数学设计（基于拒绝采样），我们可以在保证输出分布与大模型完全一致的前提下，显著提升生成速度。具体的概率计算逻辑在 **[Speculative 算法代码](./Lecture10-Code-Speculative.md)** 中有详细模拟。

### 6. 系统层面的优化
*   **Continuous Batching**：解决请求到达时间不同步的问题，允许在Batch中动态插入新请求。
*   **PagedAttention**：借鉴操作系统的虚拟内存分页思想，解决KV Cache的显存碎片化问题（vLLM的核心技术）。

### 拓展阅读
*   建议首先阅读 **[算术强度理论笔记](./Lecture10-Theory-ArithmeticIntensity.md)**，理解推理慢的物理本质。
*   结合 **[Transformer统计代码](./Lecture10-Code-Stats.md)**，尝试修改参数，观察Llama 2在不同Batch Size下的显存和延迟变化。
*   深入研究 **[投机采样代码](./Lecture10-Code-Speculative.md)**，理解其如何保证无损生成的数学原理。