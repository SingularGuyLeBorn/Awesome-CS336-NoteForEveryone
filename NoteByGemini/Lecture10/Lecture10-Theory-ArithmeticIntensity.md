# 理论专题：算术强度与内存墙 (Arithmetic Intensity & The Memory Wall)

### 1. 核心概念
算术强度（Arithmetic Intensity）是衡量计算任务特性的核心指标，定义为：
$$ \text{Intensity} = \frac{\text{Total FLOPs}}{\text{Total Bytes Accessed}} $$

它反映了算法每从内存中读取一个字节的数据，能够进行多少次浮点运算。

### 2. Roofline 模型
硬件性能通常受限于两个峰值：
1.  **峰值计算能力 (Peak Compute)**：GPU每秒能进行的浮点运算次数 (FLOPS)。
2.  **峰值内存带宽 (Peak Memory Bandwidth)**：GPU显存每秒能传输的数据量 (Bytes/s)。

硬件的**加速器强度 (Accelerator Intensity)** 定义为：
$$ I_{device} = \frac{\text{Peak FLOPS}}{\text{Peak Bandwidth}} $$
例如，对于NVIDIA H100：
*   FP16 Tensor Core FLOPS $\approx 989 \times 10^{12}$
*   Memory Bandwidth $\approx 3.35 \times 10^{12}$ Bytes/s
*   $I_{H100} \approx 295$

**判定规则**：
*   若算法的算术强度 $> I_{device}$，则为**计算受限 (Compute-Limited)**。
*   若算法的算术强度 $< I_{device}$，则为**内存受限 (Memory-Limited)**。

### 3. 推理中的两重天
在Transformer推理中，我们面临两种截然不同的负载：

#### A. Prefill 阶段 (Prompt Processing)
*   **操作**：一次性处理用户输入的整个Prompt（长度为 $S$）。
*   **矩阵运算**：$ [B, S, D] \times [D, F] $。
*   **算术强度**：与 Batch Size ($B$) 和序列长度 ($S$) 成正比。由于 $S$ 通常较大，算术强度通常远超 295，属于**计算受限**。此时GPU利用率高。

#### B. Generation 阶段 (Token Generation)
*   **操作**：逐个生成Token（长度为 1）。
*   **矩阵运算**：$ [B, 1, D] \times [D, F] $（实际上是矩阵-向量乘法）。
*   **算术强度**：
    *   读取权重矩阵：$O(D \cdot F)$
    *   计算量：$O(D \cdot F)$
    *   强度 $\approx 1$。
*   **结论**：1 远小于 295。生成阶段严重**内存受限**。GPU核心大部分时间在空转，等待权重从HBM显存加载到SRAM中。

### 4. 优化方向
由于Generation阶段受限于内存带宽，单纯增加计算单元（更多的Tensor Cores）无法提升速度。唯一的优化路径是：
1.  **减少数据传输量**：量化（使用int8/fp8减半权重体积）、模型剪枝、GQA（减少KV Cache读取）。
2.  **增加单次读取的计算量**：增大Batch Size（让多个请求共享权重的读取），这也是为什么高并发下吞吐量会提升的原因。