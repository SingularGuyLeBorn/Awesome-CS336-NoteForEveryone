# 专题笔记: 优化器 (Optimizers)

### 1. 核心概念

在深度学习中,**优化器 (Optimizer)** 是一种算法或方法,它根据模型参数的损失函数梯度来调整这些参数(如权重和偏置),其最终目标是最小化损失函数. 训练神经网络的过程本质上就是一个迭代的优化过程. 

在 **[PyTorch](./Lecture2-PyTorch.md)** 中,所有优化器都位于 `torch.optim` 模块下. 它们接收一个包含模型参数的迭代器作为输入,并封装了更新这些参数的逻辑. 

### 2. 核心工作流程

一个典型的训练步骤中,优化器的使用流程如下: 

1.  **梯度清零 (`optimizer.zero_grad()`)**: 在计算新一轮的梯度之前,必须清除上一轮迭代中累积的梯度. 因为 PyTorch 的 `.backward()` 方法默认会累积梯度,而不是覆盖. 
2.  **计算梯度 (`loss.backward()`)**: 对损失张量调用 `.backward()`,PyTorch 的 **[Autograd](./Lecture2-Autograd.md)** 引擎会计算出损失函数关于所有可训练参数的**[梯度](./Lecture2-Gradients.md)**. 
3.  **更新参数 (`optimizer.step()`)**: 调用 `.step()` 方法,优化器会根据其内部定义的更新规则(例如,SGD的规则或Adam的规则)和已经计算好的梯度,来更新所有传入的参数. 

**代码示例: **
```python
import torch
import torch.nn as nn

# 假设有模型、数据和损失函数
model = nn.Linear(10, 2)
input_data = torch.randn(3, 10)
target = torch.randn(3, 2)
loss_fn = nn.MSELoss()

# 1. 实例化优化器,传入模型参数和学习率
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# --- 训练循环中的一步 ---
# 2. 清除旧梯度
optimizer.zero_grad()

# 3. 前向传播计算损失
output = model(input_data)
loss = loss_fn(output, target)

# 4. 反向传播计算梯度
loss.backward()

# 5. 调用优化器更新参数
optimizer.step()
# -------------------------

print("参数已更新！")
```

### 3. 主流优化器及其演进

#### a. [随机梯度下降 (SGD)](./Lecture2-Stochastic-Gradient-Descent.md)
*   **核心思想**: 最基本的优化算法. 每次更新都沿着当前批次数据计算出的梯度的反方向移动一小步. 
*   **公式**: `new_weight = old_weight - learning_rate * gradient`
*   **优缺点**: 简单、内存开销小. 但容易陷入局部最优,且在某些方向上收敛缓慢(例如在“峡谷”地形中震荡). 

#### b. [SGD with Momentum](./Lecture2-Momentum.md)
*   **核心思想**: 为了解决 SGD 的震荡问题,引入了**动量(Momentum)**. 它模拟物理学中的动量概念,维护一个梯度的“滚动平均值”(速度向量). 更新时,不仅考虑当前梯度,还考虑历史梯度的累积方向. 
*   **公式**: 
    *   `velocity = momentum * velocity + gradient`
    *   `new_weight = old_weight - learning_rate * velocity`
*   **优缺点**: 能够加速在正确方向上的收敛,并抑制震荡,从而更快地穿过“峡谷”. 

#### c. [RMSProp](./Lecture2-RMSProp.md)
*   **核心思想**: 引入自适应学习率. 它为每个参数独立地维护一个学习率,其思想是: 对于梯度较大的参数,减小其学习率; 对于梯度较小的参数,增大学习率. 这是通过维护一个梯度平方的滚动平均值来实现的. 
*   **优缺点**: 解决了不同参数梯度差异巨大的问题,在处理非平稳目标时表现良好. 

#### d. [Adam / AdamW](./Lecture2-Adam-AdamW.md)
*   **核心思想**: **Adam (Adaptive Moment Estimation)** 可以看作是 **Momentum** 和 **RMSProp** 的结合体. 它同时维护了梯度的一阶矩(动量,类似Momentum)和二阶矩(梯度平方,类似RMSProp)的指数加权移动平均. 
*   **内存开销**: 由于需要存储一阶和二阶矩,Adam 的内存开销是 SGD 的三倍(对于FP32参数,SGD约需4字节/参数,Adam约需12字节/参数,加上参数本身共16字节). 
*   **AdamW**: 是 Adam 的一个改进版本,它在权重衰减(Weight Decay)的实现方式上有所不同. 原始 Adam 的权重衰减与梯度更新耦合,可能导致效果不佳. AdamW 将权重衰减解耦,直接在更新权重时减去一个小量,这被证明在现代深度学习模型中通常更有效. **AdamW 是目前训练大型模型(如 Transformer)的首选优化器. **

### 4. 优化器的状态

像 Adam 和 Momentum 这样的优化器是有“状态”的,它们需要在多次迭代之间存储额外的信息(如动量向量、梯度平方均值). 这些状态与模型参数一样重要,在进行**[模型检查点](./Lecture2-Checkpointing.md)**时,**必须同时保存优化器的状态字典 (`optimizer.state_dict()`)**,否则在恢复训练时,优化器会从零开始,丢失掉累积的动量等信息,可能导致训练曲线出现剧烈波动. 

---
**关联知识点**
*   [梯度 (Gradients)](./Lecture2-Gradients.md)
*   [Autograd](./Lecture2-Autograd.md)
*   [随机梯度下降 (SGD)](./Lecture2-Stochastic-Gradient-Descent.md)
*   [Momentum](./Lecture2-Momentum.md)
*   [RMSProp](./Lecture2-RMSProp.md)
*   [Adam / AdamW](./Lecture2-Adam-AdamW.md)
*   [模型检查点 (Checkpointing)](./Lecture2-Checkpointing.md)