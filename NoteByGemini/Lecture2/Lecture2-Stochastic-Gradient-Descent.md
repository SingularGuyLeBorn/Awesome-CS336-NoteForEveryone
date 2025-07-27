# 专题笔记: 随机梯度下降 (Stochastic Gradient Descent, SGD)

### 1. 核心概念

**随机梯度下降 (Stochastic Gradient Descent, SGD)** 是深度学习中最基本、也是最重要的**[优化器(Optimizer)](./Lecture2-Optimizers.md)**之一. 它是对经典**梯度下降(Gradient Descent, GD)**算法的一个改进,旨在解决 GD 在处理大规模数据集时的效率问题. 

### 2. 从梯度下降 (GD) 到 SGD

#### a. 批量梯度下降 (Batch Gradient Descent, GD)
*   **工作方式**: 在每次参数更新之前,GD 会**遍历整个训练数据集**,计算出损失函数在所有样本上的平均**[梯度](./Lecture2-Gradients.md)**. 然后,使用这个“精确”的平均梯度来更新一次参数. 
*   **优点**: 
    *   更新方向非常稳定,因为它是基于全体数据的梯度. 
    *   如果损失函数是凸函数,保证能收敛到全局最小值. 
*   **缺点**: 
    *   **计算成本极高**. 对于现代动辄数 TB 的数据集,每次更新都计算一遍整个数据集的梯度是完全不可行的. 
    *   **内存需求大**: 需要一次性将整个数据集的计算结果保存在内存中. 

#### b. 随机梯度下降 (Stochastic Gradient Descent, SGD)
*   **工作方式**: 为了解决 GD 的效率问题,SGD 采取了一种近似的策略. 在每次参数更新时,它**只随机抽取一个样本**来计算梯度,并立即用这个梯度来更新参数. 
*   **优点**: 
    *   **更新速度极快**: 每次更新的计算成本非常低. 
    *   **内存需求小**. 
*   **缺点**: 
    *   **更新方向非常不稳定(高方差)**. 由于单个样本的梯度可能与整体梯度方向差异很大,SGD 的更新路径会非常“嘈杂”和“随机”,像一个醉汉下山,摇摇晃晃. 
    *   这种噪音虽然有时能帮助跳出局部最优,但也使得收敛过程非常震荡,难以达到最优解. 

### 3. Mini-batch SGD: 两全其美的方案

在实践中,我们使用的“SGD”通常指的是**小批量随机梯度下降 (Mini-batch Stochastic Gradient Descent)**. 这是一种介于 GD 和纯 SGD 之间的折衷方案. 

*   **工作方式**: 在每次参数更新时,它会从训练数据中随机抽取一个**小批量(mini-batch)**的样本(例如,32、64 或 128 个样本). 然后,计算损失在这个 mini-batch 上的平均梯度,并用这个梯度来更新参数. 
*   **优点**: 
    1.  **兼顾了稳定性和效率**: 相比纯 SGD,mini-batch 的梯度更加稳定,降低了更新的方差; 相比 GD,计算成本大大降低. 
    2.  **充分利用硬件并行性**: 现代 GPU 是为并行计算设计的. 处理一个 mini-batch 的数据可以充分利用 GPU 的并行能力,效率远高于逐个处理样本. 
*   **缺点**: 
    *   引入了一个新的超参数: **批处理大小(batch size)**,需要仔细调整. 
    *   收敛路径仍然比 GD 震荡. 

**更新公式(Mini-batch SGD): **
`new_weight = old_weight - learning_rate * (1/batch_size) * Σ(gradient_for_each_sample_in_batch)`

### 4. SGD 的挑战与后续发展

尽管 Mini-batch SGD 是一个强大的基准,但它仍然存在一些挑战: 
*   **学习率选择敏感**: 学习率过大容易导致震荡不收敛,过小则收敛缓慢. 
*   **容易陷入局部最优或鞍点**. 
*   **在所有维度上使用相同的学习率**,对于那些不同参数梯度差异巨大的“病态”曲面,SGD 表现不佳. 

为了解决这些问题,研究人员提出了一系列更先进的优化器,它们都在 SGD 的基础上进行了改进: 
*   **[Momentum](./Lecture2-Momentum.md)**: 引入动量来加速收敛并抑制震荡. 
*   **AdaGrad, [RMSProp](./Lecture2-RMSProp.md)**: 为每个参数引入自适应的学习率. 
*   **[Adam / AdamW](./Lecture2-Adam-AdamW.md)**: 结合了 Momentum 和自适应学习率的优点,成为当前最主流的优化器之一. 

尽管如此,理解 SGD 仍然至关重要,因为它是所有这些高级优化算法的基石和出发点. 

---
**关联知识点**
*   [优化器 (Optimizers)](./Lecture2-Optimizers.md)
*   [梯度 (Gradients)](./Lecture2-Gradients.md)
*   [Momentum](./Lecture2-Momentum.md)
*   [Adam / AdamW](./Lecture2-Adam-AdamW.md)