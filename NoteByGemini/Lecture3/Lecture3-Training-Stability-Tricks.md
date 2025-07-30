### 专题笔记:训练稳定性技巧 (Training Stability Tricks)

#### 1. 问题背景

随着语言模型的规模(参数量、训练数据、训练时长)不断增大,**训练过程的稳定性**成为一个至关重要且极具挑战性的问题. 不稳定的训练通常表现为:
-   **损失尖峰(Loss Spikes)**:训练损失在平稳下降的过程中突然出现剧烈的、短暂的飙升. 
-   **梯度爆炸(Gradient Explosion)**:梯度范数(Gradient Norm)变得异常巨大,导致权重更新过大,模型状态被破坏. 
-   **训练崩溃**:在最坏的情况下,梯度或损失值会变成 `NaN` (Not a Number) 或 `Inf` (Infinity),导致整个训练任务失败. 

一个稳定的训练过程,其损失曲线应该平滑下降,梯度范数应保持在受控的、相对较低的水平. 

#### 2. 问题根源:Softmax 的不稳定性

在 **[Transformer 架构](./Lecture3-Transformer-Architecture.md)**中,大部分不稳定性问题都指向同一个“罪魁祸首”—— **Softmax 函数**. 
-   **位置**:Softmax 出现在两个关键位置:1) **最终的输出层**,用于生成词汇表概率;2) **每个注意力子层**,用于计算注意力权重. 
-   **原因**:Softmax 包含指数运算 `exp(x)`. 当输入 `x`(即 logits)的数值变得很大时,`exp(x)` 的结果会急剧增长,极易导致数值溢出和计算上的不稳定. 

#### 3. 关键稳定性技巧

为了“驯服”Softmax,研究者们开发了一系列技巧,这些技巧已成为现代大规模模型训练的标配. 

##### **A. Z-loss (应用于输出层 Softmax)**

-   **思想**:Softmax 的计算公式为 `P(x) = exp(logits) / Z`,其中 `Z` 是归一化因子(所有 `exp(logits)` 的总和). 当 `Z` 接近 1(即 `log(Z)` 接近 0)时,Softmax 的计算最稳定. 
-   **做法**:在主损失函数之外,增加一个**辅助损失项** `z_loss = α * log²(Z)`. 这个辅助损失会惩罚那些导致 `log(Z)` 偏离 0 的预测,从而鼓励模型学习产生更稳定的 logits. 
-   **应用**:由 PaLM 模型开创,并被 Baichuan 2、OLMo 2 等后续模型采纳. 

##### **B. QK-Norm (应用于注意力层 Softmax)**

-   **思想**:直接控制输入到 Softmax 的 logits 的大小. 既然 logits 是由 Query (Q) 和 Key (K) 的内积产生的,那么我们可以通过归一化 Q 和 K 来限制其内积的大小. 
-   **做法**:在计算 Q 和 K 的内积**之前**,对 Q 和 K 分别应用一次**[层归一化](./Lecture3-Layer-Normalization.md)**(通常是 LayerNorm 或 RMSNorm). 
-   **效果**:这是一种非常直接且有效的干预手段. 通过确保 Q 和 K 的范数受控,可以从根本上防止它们的内积(即 logits)变得过大,从而保证了注意力 Softmax 的稳定性. 
-   **应用**:这一技巧最初源于视觉和多模态模型,后被 Gemma 2、OLMo 2 等文本模型广泛采用,以稳定训练. 

##### **C. Logit Soft-capping**

-   **思想**:对计算出的 logits 设置一个“软上限”. 
-   **做法**:在将 logits 输入 Softmax 之前,通过一个 `tanh` 函数对其进行变换:
    `logits_capped = cap * tanh(logits / cap)`
    其中 `cap` 是一个预设的上限值(如 30.0). 当 `logits` 远大于 `cap` 时,`tanh` 的值会趋近于 1,从而将 `logits_capped` 限制在 `cap` 附近. 
-   **应用**:Gemma 2 等模型采用了此方法. 不过,一些研究表明,相比 QK-Norm,它可能对最终性能有轻微的负面影响. 

**结论**:层出不穷的稳定性技巧,尤其是围绕 LayerNorm 的各种创新应用(Pre-Norm, Double Norm, QK-Norm),体现了在追求更大模型规模的道路上,如何精细地控制和引导优化过程已成为一门核心的“炼丹术”. 