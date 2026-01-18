# Lecture 11: Scaling - Case Study and Details

**课程:** CS336
**讲师:** 
**主题:** 大模型 Scaling 的最佳实践：案例研究与 muP 详解

---

## 1. 引言：Scaling 的现状与挑战

上一讲我们讨论了 **Chinchilla Scaling Laws**，建立了关于模型参数量与训练数据量之间权衡的理论基础。然而，在 ChatGPT 爆发后，前沿实验室（如 OpenAI, Anthropic）对 Scaling 的细节变得讳莫如深。

今天的核心动机是回答一个实际问题：**在实践中 Scaling 一个大模型的最佳范式是什么？**

这就引出了一系列需要验证的问题：

* Chinchilla 的方法在实际操作中真的有效吗？
* 如果在对数坐标图上拟合 IsoFlops 曲线，它真的能告诉我们正确的 Token 权衡吗？
* 我们可以利用 Scaling Laws 来设定最优学习率（Learning Rate）吗？
* 我们是否应该选择特定的架构或参数化方法（Parametrization）来确保模型能够平滑地 Scaling？

由于缺乏西方顶尖实验室的公开数据，本讲座将深入剖析几个**公开且执行力极高**的 Scaling 案例，主要来自中国的研究团队（DeepSeek, MiniCPM）以及 Cerebras。它们代表了 2023-2025 年间 Scaling 研究的“黄金标准”。

---

## 2. 案例研究：Scaling 在现实世界中的应用

我们将重点分析三个模型：**Cerebras-GPT**、**MiniCPM** 和 **DeepSeek (V1)**。此外，还会简要提及 **Llama 3**、**Hunyuan Large** 和 **Minimax 01** 的相关发现。

### 2.1 Cerebras-GPT (2023)

Cerebras 团队发布了从 111M 到 13B 参数的一系列模型，完全遵循 Chinchilla 的配方（即 Token 数约为参数量的 20 倍）。

* **核心发现**：他们引入了 **Maximal Update Parametrization (muP)** 来稳定 Scaling 过程。
* **muP 的优势**：
  * 在传统的参数化（Standard Parametrization, SP）下，随着模型变大，最优学习率会发生显著漂移（通常需要变小），导致大模型训练初期容易出现震荡（Oscillations），难以精确预测 Scaling 曲线。
  * 使用 **muP** 后，他们发现 Scaling 曲线更加平滑，且更贴近理论预测。最重要的是，**最优超参数（特别是学习率）在不同规模的模型间保持了惊人的稳定性**。

`![Cerebras-GPT 损失曲线对比](placeholder.png)`
`**[插入图片: lecture_11.pdf, 第6页, 图表, 描述: 展示 Standard Parametrization (蓝色) 与 muP (橙色) 在 Pile 数据集上的测试损失。蓝色曲线在 Scaling 时表现出震荡，而 muP 曲线非常平滑且紧贴 Scaling Law 预测线。]**`

* **实施策略**：
  * 他们采取了**激进的小规模代理（Proxy）搜索**策略。在 40M 参数的小模型上进行详尽的超参数搜索（网格搜索），找到最优值。
  * 利用 muP 的特性，直接将这些超参数（如学习率）迁移到大模型上，无需重新搜索。

### 2.2 MiniCPM (2024)

MiniCPM 是由清华大学团队（ModelBest）发布的高性能小模型（1.2B - 2.4B）。他们的目标是用大量的计算资源训练出极高质量的小模型。

* **Scaling 策略**：
  1. 同样使用 **muP** 来稳定初始化和学习率，确保超参数可迁移。
  2. 固定模型的长宽比（Aspect Ratio），只调整整体模型大小。
  3. **WSD (Warmup-Stable-Decay) Learning Rate Scheduler**：这是他们推广的一项重要技术。

`![WSD 学习率调度示意图](placeholder.png)`
`**[插入图片: lecture_11.pdf, 第18页, 图15, 描述: 左侧展示了标准的 Cosine 学习率曲线（黄色）与 WSD 曲线（绿色）的对比。WSD 包含 Warmup、一个长且平坦的 Stable 阶段，以及最后的急剧 Decay 阶段。]**`

* **WSD 与 Chinchilla 分析**：

  * 传统的 **Cosine** 调度一旦设定了目标 Token 数，其衰减曲线就固定了，无法中途改变。要测试不同数据量的效果，必须从头训练多次（$N^2$ 成本）。
  * **WSD** 允许模型在 Stable 阶段一直训练。如果想知道“如果只训练 60% 的数据会怎样？”，只需将模型回退（Rewind）到该点，然后执行一个快速的 Decay 阶段即可。这使得**在一次主训练过程中完成多组 Chinchilla 数据点采样**成为可能。**[深入探讨: WSD (Warmup-Stable-Decay) 学习率调度策略](./Lecture11-WSD-Scheduler.md)**
* **拟合结果 (Method 3)**：

  * 他们使用联合拟合（Joint Fit）方法，得出了一个极高的数据/参数比：**192 tokens per parameter**。这远高于 Chinchilla 的 20:1。
  * 虽然 192:1 可能是一个异常值（Outlier），但这强烈暗示了**随着优化水平的提高，我们应该在比 Chinchilla 建议的更多的数据上训练模型**。

`![MiniCPM 临界 Batch Size 分析](placeholder.png)`
`**[插入图片: lecture_11.pdf, 第14页, 图表, 描述: 展示不同模型尺寸下，Loss 与 Batch Size 的关系。红线标出了每个数据规模下的最优（临界）Batch Size，呈现出随着 Loss 降低，最优 Batch Size 多项式级增加的趋势。]**`

### 2.3 DeepSeek LLM (2024)

DeepSeek (V1) 在 7B 和 67B 模型上展示了极高的 Scaling 分析水平。

* **策略差异**：
  * DeepSeek **没有使用 muP**。
  * 他们采用“暴力”但科学的方法：直接在大、小两个尺度上对 **Batch Size** 和 **Learning Rate** 进行网格搜索。
  * 他们发现最优 Batch Size 和 Learning Rate 与计算量（Compute）之间存在 Scaling Law 关系，并直接拟合这些曲线来预测大模型的超参数。

`![DeepSeek 学习率 Scaling 分析](placeholder.png)`
`**[插入图片: lecture_11.pdf, 第29页, 图(b), 描述: 展示 DeepSeek 拟合的最优学习率 Scaling 曲线。虽然讲师认为该拟合线略显牵强（点有些重叠），但 DeepSeek 确实以此确定了大模型的学习率。]**`

* **Chinchilla 复现**：
  * 他们也使用了 WSD 风格的调度器（虽然是分段阶梯式衰减）来高效获取数据。
  * 他们精确地复现了 IsoFlops 分析，展示了完美的预测能力——利用小规模实验准确预测了 7B 和 67B 模型的最终 Loss。

### 2.4 其他模型简述

* **Llama 3 (2024)**: 复现了 IsoFlops 分析，得出了约 **39:1** 的 Token-Parameter 比例。更重要的是，他们尝试建立 **Log Loss (Perplexity)** 与 **下游任务准确率 (Downstream Accuracy)** 之间的关联，以便直接针对任务性能进行 Scaling。
* **Hunyuan Large (2024)**: 针对 MoE (Mixture of Experts) 模型进行了 Scaling 分析，得出了 **96:1** 的 Data-to-Active Parameter 比例。
* **Minimax 01 (2025)**: 这是一个线性 Attention（Linear Attention）模型。他们利用 Scaling Law 证明了 Linear Attention 和 Hybrid 架构在 Scaling 趋势上与标准 Softmax Attention 一致，从而验证了长上下文架构的可行性。

---

## 3. 深度解析：Maximal Update Parametrization (muP)

通过案例我们看到，能够跨越规模迁移超参数（Scale-Invariant Hyperparameters）是 Scaling 的圣杯。**muP** 旨在通过特定的参数化方式，使得最优学习率在模型宽度（Width）变化时保持不变。

### 3.1 muP 的核心思想与推导

muP 的推导基于两个核心的**谱条件 (Spectral Conditions)**，或者说关于量级的断言（Assertion）：

1. **条件 A1 (初始化稳定性)**: 随着模型宽度 $n$ 增加，激活值（Activations）的坐标级数值应保持 $O(1)$，不应爆炸也不应消失。
2. **条件 A2 (更新稳定性)**: 在进行一步梯度下降后，激活值的**变化量**（Change in Activation, $\Delta h$）也应保持 $O(1)$。

#### 推导 A1：初始化 (Initialization)

考虑一个简单的深度线性网络 $h_l = W_l h_{l-1}$。
假设 $W_l$ 初始化为高斯分布 $N(0, \sigma^2)$。根据随机矩阵理论，矩阵的算子范数（Operator Norm） $\|W_l\|_*$ 会集中在 $\sigma \cdot \sqrt{n}$ 附近。

为了保持激活值的范数 $\|h_l\|_2 \approx \sqrt{n}$ （即坐标级为 $O(1)$），我们需要：

$$
\|h_l\|_2 \approx \|W_l\|_* \|h_{l-1}\|_2
$$

代入 $\sqrt{n}$ 的归纳假设，我们需要 $\|W_l\|_* \approx 1$。
因此，我们需要设置初始化的方差 $\sigma$ 为：

$$
\sigma \propto \frac{1}{\sqrt{n}}
$$

这与标准的 **He/Kaiming Initialization** 是一致的（$1/\sqrt{\text{fan-in}}$）。

#### 推导 A2：学习率与更新 (Learning Rate & Updates)

这是 muP 与标准参数化（SP）分道扬镳的地方。
考虑 SGD 更新 $\Delta W_l = -\eta \nabla_{W_l} \ell$。
权重的更新量 $\Delta W_l$ 会导致激活值的变化 $\Delta h_l$。我们需要 $\Delta h_l$ 的量级为 $O(\sqrt{n})$（与激活值本身同级）。

推导过程涉及分析 $\Delta h_l = W_l \Delta h_{l-1} + \Delta W_l h_{l-1}$ 等项。
关键结论是，为了满足条件 A2，**学习率 $\eta$ 必须随着宽度的变化而缩放**。

* **对于 SGD**: 推导结果显示 $\eta \propto \frac{n_{out}}{n_{in}}$。对于 Transformer 这种宽度均匀的网络，这意味着学习率应为常数 $O(1)$。
* **对于 Adam**: 推导结果显示，由于 Adam 的自适应特性，学习率应缩放为：
  $$
  \eta_{\text{Adam}} \propto \frac{1}{n}
  $$

  即学习率应随宽度的增加而线性减小。

**注意**：在标准参数化（SP）中，我们通常对所有层使用全局常数学习率。而在 muP 中，不同类型的层（如 Embedding, Hidden, Readout）可能有不同的缩放规则。

`![muP 与标准参数化对比表](placeholder.png)`
`**[插入图片: lecture_11.pdf, 第8页, 表格, 描述: 详细对比了 Standard Parametrization (SP) 和 Maximal Update (muP) 在初始化方差、学习率缩放上的区别。关键点：muP 中 AdamW 学习率随宽度 (1/width) 缩放，初始化方差为 (1/width)。]**`

**[深入探讨: Maximal Update Parametrization (muP) 理论与推导](./Lecture11-muP-Theory.md)**

### 3.2 实证分析：muP 真的有效吗？

我们参考一篇名为 "A Large-Scale Exploration of $\mu$-Transfer" 的论文进行验证。

1. **有效性**：在宽度 Scaling 实验中（128 -> 2048），muP 确实使得最优学习率保持在一个非常稳定的区间（Base LR 约为 $2^{-6}$），而 SP 模型的学习率则需要不断调整，否则 Loss 会变得极差。
2. **muP 的鲁棒性（什么会打破 muP？）**：

   * **Robust (有效)**:
     * **非线性激活函数**: SwiGLU, Squared ReLU。
     * **Batch Size**: 增大或减小 Batch Size 不影响 muP 的 Scaling 规律。
     * **初始化变体**: Zero Query Init 等。
   * **Not Robust (失效)**:
     * **RMSNorm with Learnable Gains**: 如果 RMSNorm 包含可学习的增益参数（Gain），muP 会失效。**必须移除 Gain 或将其初始化行为特殊处理**。
     * **Exotic Optimizers**: 如 **Lion** 优化器（基于符号的梯度更新），muP 无法直接迁移。
     * **Strong Weight Decay**: 极强的权重衰减（如 0.1）会导致 muP 失效。

`![muP 鲁棒性消融实验](placeholder.png)`
`**[插入图片: lecture_11.pdf, 第51页, 表格, 描述: 展示 RMSNorm Gains 导致 muP 迁移失效的数据。表格显示带有可学习 Gains 的模型在不同宽度下最优学习率不再对齐（标红的叉）。]**`

---

## 4. 总结：Scaling in the Wild

在 2025 年的视角下，训练大模型的 Scaling 最佳实践可以总结为以下几点：

1. **超参数稳定性**：使用 **muP**（或类似的元参数化方法 Meta-P）是一个强有力的工具，它允许你在小模型上调试超参数，然后放心地扩展到大模型。主要关注点是初始化（$1/\sqrt{n}$）和每层学习率的缩放（Adam 下为 $1/n$）。
2. **数据 Scaling**：**WSD Scheduler** 是目前的行业首选。它不仅性能与 Cosine 相当，更重要的是它提供了“免费”的 Scaling Law 数据点收集能力（通过 Rewind），极大地降低了研究计算成本。
3. **Scaling Laws 的拟合**：
   * **IsoFlops 分析**仍然是确定模型大小与数据量权衡的黄金标准。
   * 不要迷信 Chinchilla 的 20:1 比例。最新的模型（Llama 3, MiniCPM）显示，在更高的数据比例（40:1 甚至 100+:1）下训练往往更加划算，特别是考虑到推理成本时。
4. **实战心态**：像 DeepSeek 那样，即使不使用 muP，也要通过网格搜索和严谨的曲线拟合来确定 Batch Size 和 Learning Rate 的 Scaling 趋势。不要盲目猜测大模型的超参数。
