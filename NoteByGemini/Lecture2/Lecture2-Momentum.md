# 专题笔记: Momentum (动量)

### 1. 核心概念

**Momentum(动量)** 是一种用于加速**[随机梯度下降(SGD)](./Lecture2-Stochastic-Gradient-Descent.md)**收敛并抑制其震荡的优化技术. 它由 Polyak 在1964年提出,其核心思想源于对物理学中动量概念的模拟: 一个物体在运动时,其当前的速度不仅取决于当前受到的力,还受到其自身惯性的影响. 

在优化算法中: 
*   **物体**: 模型参数. 
*   **力**: 当前计算出的梯度(梯度的反方向). 
*   **速度**: 参数更新的向量. 
*   **惯性/动量**: 历史梯度的累积效应. 

Momentum 优化器维护一个“速度(velocity)”向量 `v`,它是过去所有梯度的一个**指数加权移动平均(exponentially weighted moving average)**. 在每次参数更新时,算法不是直接使用当前梯度,而是使用这个累积的速度向量. 

### 2. Momentum 的工作原理

**标准 SGD 的更新规则: **
`new_weight = old_weight - learning_rate * current_gradient`

**Momentum 的更新规则: **
1.  **计算速度向量 `v`**: 
    `v = beta * v + (1 - beta) * current_gradient`
    (在常见的 PyTorch 实现中,为了简化,`1-beta` 通常被并入学习率中,形式为: `v = beta * v + current_gradient`)

2.  **用速度向量更新权重**: 
    `new_weight = old_weight - learning_rate * v`

其中: 
*   **`v`** 是速度向量,与参数具有相同的维度,初始为0. 
*   **`beta`** 是动量系数,一个通常接近于1的超参数(例如 0.9). 它控制了历史梯度对当前更新方向的影响程度. `beta` 越大,历史梯度的影响就越大,更新方向越平滑. 

### 3. Momentum 的优势

Momentum 主要解决了标准 SGD 的两个问题: 

1.  **加速收敛**: 
    想象一下在一个狭长的“峡谷”地形中进行优化. SGD 会在峡谷的两壁之间来回震荡,同时缓慢地向谷底移动. 
    *   在**震荡方向**(梯度方向反复变换),动量项 `beta * v` 会将相反方向的梯度相互抵消,从而**抑制震荡**. 
    *   在**前进方向**(梯度方向基本一致),动量项会累积同向的梯度,使得速度 `v` 越来越大,从而**加速前进**. 
    最终,参数会更快、更稳定地向最优点移动. 

2.  **跳出局部最优**: 
    当参数陷入一个平坦的局部最优区域时,当前梯度 `current_gradient` 可能会变得非常小,导致 SGD 停止更新. 但如果此时速度向量 `v` 仍然携带了之前积累的“惯性”,它就可能帮助参数“冲出”这个局部最优,继续寻找更好的解. 

### 4. Nesterov 加速梯度 (Nesterov Accelerated Gradient, NAG)

Nesterov Momentum 是对标准 Momentum 的一个巧妙改进. 

*   **标准 Momentum**: 先计算当前位置的梯度,然后结合历史速度来决定下一步怎么走. 
*   **Nesterov Momentum**: 它更有“前瞻性”. 它首先**想象**一下,如果只按照历史速度 `beta * v` 会走到哪里(一个临时的未来位置). 然后,它在那个**未来的位置**计算梯度,并用这个“修正后”的梯度来更新真实的速度. 

这种“先走一步再看路”的方式,使得 Nesterov Momentum 在很多任务上,特别是对于凸函数,具有更好的收敛性质和更快的收敛速度. 在 PyTorch 的 `torch.optim.SGD` 中,可以通过设置 `nesterov=True` 来启用它. 

Momentum 是构建更高级优化器(如 **[Adam](./Lecture2-Adam-AdamW.md)**)的关键组成部分,理解其工作原理对于深入理解现代优化算法至关重要. 

---
**关联知识点**
*   [优化器 (Optimizers)](./Lecture2-Optimizers.md)
*   [随机梯度下降 (SGD)](./Lecture2-Stochastic-Gradient-Descent.md)
*   [Adam / AdamW](./Lecture2-Adam-AdamW.md)