# 优化器 (Optimizers) 的演进与实现

优化器是深度学习训练的核心,它根据损失函数计算出的梯度来指导模型参数如何更新. 选择和理解优化器对于模型的收敛速度和最终性能至关重要. 本笔记将梳理从基础的SGD到现代标配Adam的演进脉络,并分析它们的核心思想与内存开销.

## 1. 随机梯度下降 (SGD - Stochastic Gradient Descent)

这是最基础的优化器. 它的思想朴素而直接: 计算出一个小批量(mini-batch)数据的梯度,然后让参数沿着梯度的反方向移动一小步.

- **更新规则**:

  `parameter = parameter - learning_rate * gradient`

- **优点**: 实现简单,内存开销极小(不需要额外的状态存储).
- **缺点**:

  - **收敛慢**: 更新方向完全依赖当前批次的梯度,波动很大,导致收敛路径曲折.
  - **易陷于局部最优或鞍点**: 在梯度平坦的区域,更新会非常缓慢.


## 2. SGD with Momentum (动量)

为了解决SGD收敛路径曲折的问题,Momentum被引入. 它模拟了物理学中物体的惯性: 当前的更新方向不仅取决于当前的梯度,还受到上一步更新方向的影响.

- **核心思想**: 维护一个“速度”向量(一阶矩),它是过去梯度的指数移动平均. 参数更新由这个“速度”向量驱动.
- **更新规则**:

  `velocity = beta * velocity + gradient`

  `parameter = parameter - learning_rate * velocity`

  (`beta` 是动量系数,通常设为0.9)

- **优点**:

  - **加速收敛**: 在梯度方向一致的维度上,动量会累积,使得更新加速.
  - **抑制震荡**: 在梯度方向反复变化的维度上,动量会相互抵消,使得更新更平滑.

- **内存开销**: 需要为每个参数额外存储一个`velocity`状态. 成本: `1 * P * 4` 字节 (假设FP32).

## 3. AdaGrad (Adaptive Gradient)

Momentum解决了更新方向的问题,而AdaGrad则关注于更新步长(学习率)的问题. 它认为,不同参数应该有不同的学习率.

- **核心思想**: 对于更新频繁的参数,给予较小的学习率; 对于更新稀疏的参数,给予较大的学习率. 它通过累积每个参数梯度的**平方和**来实现这一点.
- **更新规则**:

  `grad_squared_sum = grad_squared_sum + gradient * gradient`

  `parameter = parameter - learning_rate * gradient / (sqrt(grad_squared_sum) + epsilon)`

- **优点**: 自动调整学习率,对稀疏特征(如词嵌入)非常有效.
- **缺点**: `grad_squared_sum`会单调递增,导致学习率在训练后期变得过小,模型可能提前停止学习.
- **内存开销**: 需要为每个参数额外存储一个`grad_squared_sum`状态. 成本: `1 * P * 4` 字节.

## 4. RMSProp (Root Mean Square Propagation)

RMSProp是AdaGrad的一个改进,旨在解决其学习率过早衰减的问题.

- **核心思想**: 将累积梯度平方和的方式,从简单的相加,改为**指数移动平均**. 这使得优化器能够“忘记”遥远的过去,只关注近期的梯度大小.
- **更新规则**:

  `grad_squared_avg = beta * grad_squared_avg + (1-beta) * gradient * gradient`

  `parameter = parameter - learning_rate * gradient / (sqrt(grad_squared_avg) + epsilon)`

- **优点**: 解决了AdaGrad学习率消失的问题,在非凸优化问题中表现更好.
- **内存开销**: 与AdaGrad相同,需要一个状态. 成本: `1 * P * 4` 字节.

## 5. Adam (Adaptive Moment Estimation) - 现代标配

Adam可以被看作是**Momentum和RMSProp的集大成者**. 它同时利用了梯度的一阶矩(动量)和二阶矩(自适应学习率).

- **核心思想**:

  1. 像Momentum一样,计算梯度的指数移动平均(一阶矩,`m`).
  2. 像RMSProp一样,计算梯度平方的指数移动平均(二阶矩,`v`).
  3. 用`m`来决定更新方向,用`v`来调整每个参数的学习率.

- **更新规则 (简化版)**:

  `m = beta1 * m + (1 - beta1) * gradient`

  `v = beta2 * v + (1 - beta2) * gradient * gradient`

  `parameter = parameter - learning_rate * m / (sqrt(v) + epsilon)`

  (Adam还包含一个“偏差修正”步骤,此处省略)

- **优点**: 结合了Momentum和RMSProp的优点,收敛速度快,对初始学习率不敏感,在绝大多数场景下都表现良好.
- **缺点**: 内存开销最大.
- **内存开销**: 需要为每个参数存储**两个状态**: 一阶矩`m`和二阶矩`v`. 成本: `2 * P * 4` 字节.

## 总结与对比

| 优化器       | 核心思想                  | 额外状态/参数        | 内存开销(FP32)                |
| :----------- | :------------------------ | :------------------- | :---------------------------- |
| **SGD**      | 沿梯度反方向更新          | 0                    | 0                             |
| **Momentum** | 累积速度,平滑更新         | 1 (velocity)         | `1 * 4 * P`                   |
| **AdaGrad**  | 累积梯度平方,自适应学习率 | 1 (grad_squared_sum) | `1 * 4 * P`                   |
| **RMSProp**  | 用移动平均改进AdaGrad     | 1 (grad_squared_avg) | `1 * 4 * P`                   |
| **Adam**     | **Momentum + RMSProp**    | **2** (m and v)      | **`2 * 4 * P`** **2 * 4 * P** |

这就是为什么在进行内存估算时,我们说Adam优化器会带来`8 * P`字节的额外开销. 在训练一个百亿参数的模型时,仅Adam的优化器状态就需要几十GB的GPU内存,这是一个必须被精确计算的重要成本.