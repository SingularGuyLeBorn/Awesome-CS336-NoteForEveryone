# 专题笔记: Adam / AdamW 优化器

### 1. 核心概念: Adam

**Adam (Adaptive Moment Estimation)** 是一种在深度学习中被广泛使用的**[优化器(Optimizer)](./Lecture2-Optimizers.md)**,由 Diederik P. Kingma 和 Jimmy Ba 在 2014 年提出. 它被设计为一种既高效又易于调参的优化算法,其核心思想是**结合了两种主流优化思想的优点**: 

1.  **动量 (Momentum)**: 引入一个“速度”向量来累积梯度的历史信息,帮助优化过程冲出平坦区域和狭窄的“峡谷”,加速收敛. 这对应于梯度的**一阶矩估计(the first moment)**. 
2.  **自适应学习率 (Adaptive Learning Rate)**: 为每个参数独立地计算和调整学习率. 对于梯度变化较大的参数,使用较小的学习率; 对于梯度变化较小的参数,使用较大的学习率. 这对应于梯度的**二阶矩估计(the second moment)**,类似于 **[RMSProp](./Lecture2-RMSProp.md)**. 

Adam 算法通过计算梯度的一阶矩(平均值)和二阶矩(未中心的方差)的指数移动平均,来动态地调整每个参数的学习率. 

### 2. Adam 的内存开销

由于 Adam 需要为模型中的每一个可训练参数存储两个额外的状态值: 
*   **m (the first moment)**: 梯度的一阶矩估计,与参数本身大小相同. 
*   **v (the second moment)**: 梯度的二阶矩估计,也与参数本身大小相同. 

因此,使用 Adam 优化器时,其状态占用的内存大约是参数本身的两倍. 对于一个使用 **[FP32](./Lecture2-FP32-FP16-BF16-FP8.md)** 的参数,Adam 的状态需要 `4 bytes (for m) + 4 bytes (for v) = 8 bytes`. 这就是课程中提到的,在使用Adam进行训练时,每个参数总共需要约 `4 (param) + 4 (grad) + 8 (optimizer state) = 16 bytes` 内存的原因. 

### 3. AdamW: 一个重要的改进

**AdamW (Adam with Decoupled Weight Decay)** 是对原始 Adam 算法关于**权重衰减 (Weight Decay)** 实现方式的一个重要修正. 

*   **权重衰减是什么？**
    权重衰减是一种正则化技术,用于防止模型过拟合. 其基本思想是在损失函数中增加一个惩罚项,该惩罚项与模型权重的平方和成正比. 在梯度下降中,这等效于在每次更新权重时,先让权重“衰减”一个小比例,然后再减去梯度. 
    `new_weight = old_weight - learning_rate * (gradient + weight_decay_factor * old_weight)`

*   **原始 Adam 的问题**: 
    在 L2 正则化(即权重衰减)的传统实现中,衰减项被包含在梯度中. 在 Adam 这类自适应学习率算法中,这个衰减项会受到梯度二阶矩 `v` 的归一化影响. 这意味着,对于梯度历史较小的权重,其有效的权重衰减效果会被放大; 对于梯度历史较大的权重,其衰减效果会被削弱. 这种耦合可能不是我们想要的行为,并且被证明在实践中效果较差. 

*   **AdamW 的解决方案**: 
    AdamW 采用了**解耦的权重衰减 (Decoupled Weight Decay)**. 它将权重衰减步骤从梯度更新中分离出来. 在计算完梯度并应用 Adam 的自适应更新规则之后,再直接从权重中减去一个与学习率相关的衰减量. 
    **步骤1 (Adam更新)**: `adam_update = learning_rate * m_hat / (sqrt(v_hat) + epsilon)`
    **步骤2 (解耦的权重衰减)**: `new_weight = old_weight - adam_update - (learning_rate * weight_decay_factor * old_weight)`

这种解耦的方式恢复了权重衰减在 **[SGD](./Lecture2-Stochastic-Gradient-Descent.md)** 中的原始行为,被证明在现代深度学习模型(特别是 **[Transformer](./Lecture2-Transformer.md)**)中更加稳定和有效. 

### 4. 结论与实践

*   **Adam** 曾是深度学习领域的默认首选优化器,因其快速收敛和对超参数不敏感的特性而广受欢迎. 
*   **AdamW** 现已取代 Adam,成为训练大型语言模型和其他现代神经网络架构的**事实标准**. 在 PyTorch 中,可以通过 `torch.optim.AdamW` 来使用它. 
*   在课程和实际项目中,当你需要选择一个通用的、性能强大的优化器时,**AdamW** 通常是你的不二之选. 

---
**关联知识点**
*   [优化器 (Optimizers)](./Lecture2-Optimizers.md)
*   [Momentum](./Lecture2-Momentum.md)
*   [RMSProp](./Lecture2-RMSProp.md)
*   [随机梯度下降 (SGD)](./Lecture2-Stochastic-Gradient-Descent.md)
*   [梯度 (Gradients)](./Lecture2-Gradients.md)