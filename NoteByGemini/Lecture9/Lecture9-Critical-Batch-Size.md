# 专题笔记: 临界批量大小 (Critical Batch Size)

### 1. 概念定义
**Critical Batch Size (临界批量大小)** 是训练效率的一个转折点。它衡量了数据并行（Data Parallelism）的有效性边界。
*   **当 Batch Size < Critical Size**: 增加 Batch Size 几乎线性地减少所需的训练步数（Steps）。此时每增加一个样本都在有效降低梯度估计的噪声。这是**时间效率最优**区域。
*   **当 Batch Size > Critical Size**: 增加 Batch Size 带来的收益急剧递减。额外的样本并没有显著提供更好的梯度方向，因为梯度噪声已经小于优化地形本身的曲率（Curvature）影响。这是**计算资源浪费**区域。

### 2. 梯度噪声尺度 (Gradient Noise Scale)
Critical Batch Size 本质上与 **Gradient Noise Scale (梯度噪声尺度)** 有关。简单来说，如果梯度的方差很大（噪声大），我们需要更大的 Batch 来平均它；如果梯度很准，大 Batch 也是浪费。

OpenAI 的研究发现，Critical Batch Size ($B_{crit}$) 与梯度的统计特性有关：
$$ B_{crit} \approx \frac{\text{Tr}(\Sigma)}{|G|^2} $$
其中 $\Sigma$ 是梯度协方差矩阵，$G$ 是真实梯度。

### 3. 动态变化
讲座中提到的一个关键点是：**随着训练进行（Loss 降低），Critical Batch Size 会变大。**
*   **原因**: 随着模型接近局部最优解，真实的梯度向量变小（变得微妙），信噪比降低。为了“看清”下降的方向，需要更多的样本来平均噪声。
*   **工程应用**: 这解释了为什么在 Llama 3 等大模型训练中，工程师会采用 **Dynamic Batch Scheduling**，即随着训练进行逐步增大 Batch Size。

### 4. 与代码的连接
在实际训练中，可以通过测量不同 Batch Size 下的 Gradient Variance 来估算这个值。具体实现逻辑请参考 **[CriticalBatchEstimator 类](./Lecture9-Code-BatchAnalysis.md)**。