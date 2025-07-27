# 专题笔记: 梯度 (Gradients)

### 1. 核心概念

在深度学习和优化的语境中,**梯度(Gradient)** 是一个向量,它指向函数值增长最快的方向. 更具体地说,对于一个多变量函数(如一个神经网络的损失函数 L,其变量是所有模型参数 w1, w2, ...),梯度 ∇L 是由该函数对每个变量的偏导数组成的向量: 

`∇L = (∂L/∂w1, ∂L/∂w2, ...)`

在训练神经网络时,我们的目标是**最小化**损失函数. 因此,我们会让参数沿着**梯度的反方向**进行更新,因为梯度的反方向是函数值下降最快的方向. 这个过程就是**梯度下降(Gradient Descent)**. 

### 2. 梯度在深度学习中的角色

梯度是连接**损失函数**和**模型参数更新**的桥梁,是学习过程的核心. 

1.  **计算损失**: 首先,模型对一批输入数据进行**前向传播**,得到预测结果. 然后,将预测结果与真实标签进行比较,通过一个损失函数(如交叉熵或均方误差)计算出一个标量值——损失(Loss). 这个损失值衡量了模型当前预测的好坏程度. 

2.  **计算梯度**: 接下来,通过**[反向传播(Backpropagation)](./Lecture2-Backpropagation.md)**算法,计算损失函数关于模型中每一个可训练参数的梯度. 这个梯度值 `∂L/∂w` 直观地表示了: **如果我将参数 `w` 增加一个微小的量,损失函数 `L` 大约会增加多少**. 
    *   如果梯度为正,意味着增加参数会增加损失,所以我们应该减小参数. 
    *   如果梯度为负,意味着增加参数会减小损失,所以我们应该继续增加参数. 

3.  **更新参数**: **[优化器(Optimizer)](./Lecture2-Optimizers.md)**(如 [SGD](./Lecture2-Stochastic-Gradient-Descent.md) 或 [Adam](./Lecture2-Adam-AdamW.md))使用计算出的梯度来更新参数. 最简单的更新规则是: 
    `new_parameter = old_parameter - learning_rate * gradient`

这个“计算损失 -> 计算梯度 -> 更新参数”的循环不断重复,直到模型收敛. 

### 3. PyTorch 中的梯度

在 **[PyTorch](./Lecture2-PyTorch.md)** 中,梯度的处理由 **[Autograd](./Lecture2-Autograd.md)** 引擎无缝完成. 

*   **开启梯度追踪**: 对于需要计算梯度的**[张量](./Lecture2-Tensors.md)**(通常是模型的参数),需要将其 `requires_grad` 属性设置为 `True`. `nn.Parameter` 默认就是 `requires_grad=True`. 
*   **梯度累积**: 当在一个张量上调用 `.backward()` 后,计算出的梯度会**累积(add up)**到对应参数的 `.grad` 属性中. 这就是为什么在每个训练迭代开始时,我们都需要调用 `optimizer.zero_grad()` 来清除上一轮的梯度. 
*   **动态计算图**: PyTorch 的一个强大之处在于其动态计算图. 这意味着反向传播的路径是在每次前向传播时动态定义的,允许在模型中使用任意的 Python 控制流(如 `if` 语句和 `for` 循环). 

**代码示例: **
```python
import torch

# 创建一个需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)

# 定义一个简单的函数
y = x**2 + 3*x

# 对 y 执行反向传播
y.backward()

# 检查 x 的梯度
# y 对 x 的导数是 2x + 3
# 当 x=2 时,梯度是 2*2 + 3 = 7
print(x.grad) # 输出: tensor(7.)
```

### 4. 梯度消失与梯度爆炸

在深度网络中,梯度是通过链式法则从后向前传播的. 如果网络中许多层的梯度值都小于1,那么梯度在反向传播过程中会指数级衰减,导致靠近输入层的参数几乎无法得到更新,这就是**梯度消失(Vanishing Gradients)**. 反之,如果梯度值都大于1,梯度会指数级增长,导致数值溢出,这就是**梯度爆炸(Exploding Gradients)**. 

这些问题可以通过以下方法缓解: 
*   **ReLU** 及其变体激活函数. 
*   **精心设计的[参数初始化](./Lecture2-Parameter-Initialization.md)**(如 Xavier, He). 
*   **Batch Normalization** 或 **Layer Normalization**. 
*   **残差连接(Residual Connections)**(如 ResNet). 
*   **梯度裁剪(Gradient Clipping)**: 在更新前,将梯度的范数限制在一个阈值内,防止梯度爆炸. 

---
**关联知识点**
*   [反向传播 (Backpropagation)](./Lecture2-Backpropagation.md)
*   [Autograd](./Lecture2-Autograd.md)
*   [优化器 (Optimizers)](./Lecture2-Optimizers.md)
*   [张量 (Tensors)](./Lecture2-Tensors.md)