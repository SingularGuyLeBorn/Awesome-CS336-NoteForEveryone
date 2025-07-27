# 专题笔记: RMSProp (Root Mean Square Propagation)

### 1. 核心概念

**RMSProp** 是一种**自适应学习率(Adaptive Learning Rate)**的**[优化器(Optimizer)](./Lecture2-Optimizers.md)**,由深度学习先驱 Geoffrey Hinton 在他的 Coursera 课程中提出. 它旨在解决 **[SGD](./Lecture2-Stochastic-Gradient-Descent.md)** 在处理非平稳目标(non-stationary objectives)或在不同参数上梯度差异巨大的“病态”优化曲面时的收敛问题. 

RMSProp 的核心思想是: **为每个参数独立地调整其学习率**. 具体来说,它会减小那些梯度持续较大的参数的学习率,同时增大学习率给那些梯度持续较小的参数. 

### 2. RMSProp 的工作原理

与 **[Momentum](./Lecture2-Momentum.md)** 维护梯度的一阶矩(均值)不同,RMSProp 维护的是**梯度平方的指数加权移动平均(exponentially weighted moving average of squared gradients)**. 我们称这个值为 `S`. 

**RMSProp 的更新规则: **

1.  **计算梯度平方的移动平均 `S`**: 
    `S = beta * S + (1 - beta) * (current_gradient)^2`
    其中 `(current_gradient)^2` 是对梯度向量进行逐元素平方. 

2.  **用 `S` 来调整学习率并更新权重**: 
    `new_weight = old_weight - (learning_rate / (sqrt(S) + epsilon)) * current_gradient`

其中: 
*   **`S`** 是梯度平方的累积量,与参数具有相同的维度. 
*   **`beta`** 是衰减率,一个类似于动量系数的超参数(例如 0.99). 
*   **`sqrt(S)`** 是梯度的均方根(Root Mean Square). 
*   **`epsilon`** 是一个非常小的数(例如 `1e-8`),用于防止分母为零,增加数值稳定性. 

### 3. RMSProp 的优势

通过将学习率除以 `sqrt(S)`,RMSProp 实现了自适应学习率: 

*   **对于梯度大的参数**: 如果某个参数的梯度 `current_gradient` 长期以来都很大,那么它的 `S` 也会很大. 这导致 `learning_rate / sqrt(S)` 这一项变小,从而**减小**了该参数的有效学习率. 这有助于在陡峭的方向上防止更新步长过大而产生震荡. 
*   **对于梯度小的参数**: 如果某个参数的梯度长期较小,它的 `S` 也会很小. 这会**增大**该参数的有效学习率,使其在平缓的方向上能以更快的速度前进. 

这种机制使得 RMSProp 在处理那些形状像狭长“峡谷”的优化地形时特别有效,因为它能在陡峭的维度上减速,在平缓的维度上加速. 

### 4. 与相关优化器的关系

*   **与 AdaGrad 的关系**: RMSProp 可以看作是 AdaGrad 优化器的一个改进. AdaGrad 同样使用梯度平方来调整学习率,但它累积的是从开始到现在的**所有**梯度平方,而不是一个移动平均. 这导致其分母会单调递增,学习率会持续下降,最终可能变得过小而使训练提前停止. RMSProp 通过使用指数移动平均,解决了这个问题,使得它更适合在非凸设定下(如神经网络)进行长期训练. 

*   **与 Adam 的关系**: **[Adam (Adaptive Moment Estimation)](./Lecture2-Adam-AdamW.md)** 优化器可以被看作是 **RMSProp 和 Momentum 的结合体**. Adam 不仅维护了梯度平方的移动平均(RMSProp 的部分),还维护了梯度本身的移动平均(Momentum 的部分). 这使得 Adam 既有自适应学习率的优点,又有动量加速的优点,因此在实践中通常表现得更加鲁棒和高效. 

RMSProp 是理解自适应学习率算法演进过程中的一个关键环节,它的核心思想被后续更先进的优化器所继承和发展. 

---
**关联知识点**
*   [优化器 (Optimizers)](./Lecture2-Optimizers.md)
*   [随机梯度下降 (SGD)](./Lecture2-Stochastic-Gradient-Descent.md)
*   [Momentum](./Lecture2-Momentum.md)
*   [Adam / AdamW](./Lecture2-Adam-AdamW.md)