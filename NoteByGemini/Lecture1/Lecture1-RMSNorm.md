# 专题：RMSNorm (Root Mean Square Layer Normalization)
## 1. 核心思想
**RMSNorm (Root Mean Square Layer Normalization)** 是对标准的层归一化（Layer Normalization, LayerNorm）进行简化后的一种变体，由 Biao Zhang 和 Rico Sennrich 在 2019 年的论文《Root Mean Square Layer Normalization》中提出。
RMSNorm 的核心思想是：**在 LayerNorm 的基础上，移除了“重新居中”（re-centering）这一步（即减去均值的操作），只保留“重新缩放”（re-scaling）这一步。**
这个看似简单的改动，可以在保持与 LayerNorm 相当性能的同时，将计算速度提升 7% 到 64%（取决于模型和硬件）。
## 2. LayerNorm 与 RMSNorm 的对比
让我们回顾一下标准的 **LayerNorm** 的计算过程：
对于一个输入向量 `x`，LayerNorm 的计算如下：
1.  **计算均值:** `μ = mean(x)`
2.  **计算方差:** `σ^2 = variance(x)`
3.  **归一化:** `x_norm = (x - μ) / sqrt(σ^2 + ε)`
4.  **重新缩放和移位:** `y = γ * x_norm + β`
    *   `γ` (gamma) 和 `β` (beta) 是两个可学习的参数，分别代表缩放因子和移位因子。
**RMSNorm** 简化了这个过程：
1.  **计算均方根 (Root Mean Square):** `RMS(x) = sqrt(mean(x^2))`
2.  **归一化:** `x_norm = x / (RMS(x) + ε)`
3.  **重新缩放:** `y = g * x_norm`
    *   `g` (gain) 是一个可学习的缩放参数，功能上等同于 LayerNorm 中的 `γ`。
**关键区别：**
*   **移除了均值减法:** RMSNorm 不计算也不减去输入的均值。它假设均值可以被隐式地处理或不是必需的。
*   **移除了移位参数 `β`:** 由于没有进行居中操作，重新移位的参数 `β` 也变得没有必要。
*   **用 RMS 替代标准差:** 分母从标准差（`sqrt(variance)`）变成了均方根（`sqrt(mean of squares)`）。
## 3. 为什么 RMSNorm 有效且更快？
### 有效性
论文作者的假设是，LayerNorm 的成功主要归功于其**重新缩放的不变性（re-scaling invariance）**，而不是居中操作。也就是说，对输入向量 `x` 进行任意的缩放，LayerNorm 的输出（除了可学习参数 `γ` 和 `β` 的影响外）是基本不变的。RMSNorm 保留了这一关键特性，因此也能取得良好的效果。在许多 Transformer-based 的任务中，实验表明 RMSNorm 的性能与 LayerNorm 相当，有时甚至略好。
### 速度优势
RMSNorm 的速度提升来自于其计算上的简化：
*   **更少的计算量:** 它省去了计算均值和方差的步骤，直接计算均方根，减少了浮点数运算。
*   **更少的内存带宽:** 它只需要一个可学习的参数 `g`，而不是 `γ` 和 `β` 两个，减少了参数的存储和读取。
在 GPU 上，这些看似微小的计算节省，由于被执行了数十亿次，累积起来可以带来显著的总体训练和**[推理](./Lecture1-Inference.md)**速度提升。
## 4. 实践与影响
*   **成为现代 LLM 的标配:** 由于其效率和性能的完美结合，RMSNorm 已经取代了传统的 LayerNorm，成为许多最先进的**[语言模型](./Lecture1-Language-Models.md)**（如 LLaMA, PaLM, Qwen）中的标准归一化方法。
*   **Pre-Norm 结构:** 在现代 **[Transformer](./Lecture1-Transformer.md)** 架构中，RMSNorm 通常与 Pre-Norm 结构一起使用，即在每个子层（自注意力层和前馈网络层）的**输入端**进行归一化，而不是在输出端。这种“先归一化再计算”的 Pre-RMSNorm 结构被证明可以使训练过程更加稳定。
**结论:** RMSNorm 是 **[Transformer](./Lecture1-Transformer.md)** 架构演进过程中的又一个典型例子，它通过一个简单而深刻的洞察，实现了在不牺牲（甚至略微提升）性能的前提下，显著提高计算效率的目标。