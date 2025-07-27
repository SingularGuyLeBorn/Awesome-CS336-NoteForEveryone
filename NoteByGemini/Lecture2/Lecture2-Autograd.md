# 专题笔记: Autograd

### 1. 核心概念

**Autograd** 是 **[PyTorch](./Lecture2-PyTorch.md)** 中实现**自动微分(Automatic Differentiation)**功能的核心引擎.  它的存在使得用户无需手动计算复杂的导数,就能轻松地为任意复杂的神经网络模型计算**[梯度](./Lecture2-Gradients.md)**.  这也是 PyTorch 如此受欢迎的关键原因之一. 

Autograd 的核心是**基于磁带的自动微分(tape-based automatic differentiation)**. 它通过在**前向传播**过程中记录所有执行的操作,构建出一个动态的**有向无环图(Directed Acyclic Graph, DAG)**,这个图通常被称为**计算图(Computation Graph)**. 

### 2. 计算图 (Computation Graph)

*   **节点 (Nodes)**: 计算图中的节点代表了**[张量(Tensors)](./Lecture2-Tensors.md)**或**操作(Functions)**. 
*   **边 (Edges)**: 边代表了数据流和依赖关系. 
*   **叶子节点 (Leaf Nodes)**: 通常是由用户直接创建的张量,例如模型的输入数据、模型的权重和偏置. 
*   **根节点 (Root Nodes)**: 是计算流程的最终输出,通常是损失函数(一个标量). 

这个图是在每次前向传播时**动态构建**的. 这意味着你可以在模型中使用任何标准的 Python 控制流(如 `if` 语句、`for` 循环),Autograd 都能正确地记录操作并构建相应的图. 这与一些早期框架(如 TensorFlow 1.x)的静态图形成了鲜明对比,极大地提升了灵活性和易用性. 

### 3. 工作原理

Autograd 的工作流程与**[反向传播](./Lecture2-Backpropagation.md)**算法紧密相连: 

1.  **前向传播与记录**: 
    *   当你对设置了 `requires_grad=True` 的张量执行任何操作时,Autograd 会将这个操作记录下来,并创建一个 `Function` 对象来代表这个操作. 
    *   这个 `Function` 对象会连接输入张量和输出张量,形成计算图的一部分. 每个非叶子节点的张量都有一个 `.grad_fn` 属性,它指向创建该张量的 `Function` 对象,这就是进入反向图的入口点. 

2.  **反向传播与梯度计算**: 
    *   当你对最终的输出(通常是损失 `loss`)调用 `.backward()` 方法时,Autograd 开始工作. 
    *   它会从根节点(`loss`)出发,沿着计算图**反向**追溯. 
    *   在每个 `Function` 节点,它会调用相应的反向计算函数,根据链式法则计算出输入张量的梯度. 
    *   这些梯度会被不断地向后传递和累积. 
    *   最终,当追溯到叶子节点时,计算出的梯度会**累积**到这些叶子张量的 `.grad` 属性中. 

**代码示例: **
```python
import torch

# 1. 创建叶子节点张量,并设置 requires_grad=True
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# 2. 进行一系列操作,Autograd 在后台构建计算图
Q = 3*a**3 - b**2
# Q 是一个中间节点,它的 .grad_fn 是 <SubBackward0>
# a**3 的 .grad_fn 是 <PowBackward0>
# ...

# 3. 假设 Q 是我们的“损失”,但由于它不是标量,我们需要先聚合
# .backward() 只能对标量调用
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad) # 等价于 L = Q.sum(); L.backward()

# 4. 检查叶子节点的梯度
# dQ/da = 9*a^2 = 9 * [4, 9] = [36, 81]
print(a.grad) # 输出: tensor([36., 81.])

# dQ/db = -2*b = -2 * [6, 4] = [-12, -8]
print(b.grad) # 输出: tensor([-12.,  -8.])
```

### 4. 控制 Autograd 的行为

*   **`torch.no_grad()`**: 这是一个上下文管理器,在它的作用域内,PyTorch **不会**追踪操作,也不会构建计算图. 这在模型**推理(inference)**时非常有用,因为它可以显著减少内存消耗并加速计算,避免不必要的梯度计算. 
*   **`torch.inference_mode()`**: 从 PyTorch 1.9 开始引入,是 `no_grad()` 的一个更优化的版本,用于推理时性能更好. 推荐在推理时使用. 
*   **`.detach()`**: 创建一个与原始张量共享数据 `storage` 但**不**在计算图中的新张量. 这在你希望某个操作不被追踪梯度时很有用. 

### 5. 结论

Autograd 是 PyTorch 框架的魔法核心. 它通过动态计算图和自动化的反向传播,将开发者从繁琐的手动梯度计算中解放出来,使得快速迭代和实现复杂的模型架构成为可能. 理解 Autograd 的基本工作原理,有助于编写更高效、更清晰的 PyTorch 代码,并能在出现问题时更好地进行调试. 

---
**关联知识点**
*   [PyTorch](./Lecture2-PyTorch.md)
*   [张量 (Tensors)](./Lecture2-Tensors.md)
*   [梯度 (Gradients)](./Lecture2-Gradients.md)
*   [反向传播 (Backpropagation)](./Lecture2-Backpropagation.md)