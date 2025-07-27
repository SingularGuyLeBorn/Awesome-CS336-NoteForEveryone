# 专题笔记: PyTorch

### 1. 核心概念

PyTorch 是一个由 Facebook 的 AI 研究实验室(FAIR)开发的开源机器学习库,现已成为学术界和工业界进行深度学习研究和开发的主流框架之一. 其核心魅力在于**简洁性、灵活性和强大的GPU加速能力**. 

PyTorch 的设计哲学是 "Python-first",它与 Python 编程语言紧密集成,使得代码直观易懂,调试方便,感觉就像在写普通的 Python 程序. 

### 2. 两大核心特性

PyTorch 的强大功能主要建立在两个核心组件之上: 

*   **张量 (Tensors)**: PyTorch 的 `torch.Tensor` 是一种类似于 [NumPy](./Lecture2-NumPy.md) 的n维数组,但它有一个关键的超能力: 可以在 GPU 上进行计算以实现大规模加速. 所有的数据,无论是输入、模型参数还是梯度,都以张量的形式存在. 

*   **自动微分 (Automatic Differentiation)**: PyTorch 内置了名为 **[Autograd](./Lecture2-Autograd.md)** 的自动微分引擎. 当你使用张量进行计算时,Autograd 会自动构建一个动态计算图,记录下所有操作. 这使得计算任意复杂模型的梯度变得异常简单,只需在损失张量上调用 `.backward()` 方法,PyTorch 就会自动执行**[反向传播](./Lecture2-Backpropagation.md)**,并计算出所有需要梯度的参数的梯度值. 

### 3. 构建模型: `nn.Module`

在 PyTorch 中,所有神经网络模型都应该继承自 `torch.nn.Module` 类. 这是一个功能强大的基类,它提供了许多便利的功能: 

*   **参数管理**: `nn.Module` 会自动追踪所有被定义为 `nn.Parameter` 的属性. `nn.Parameter` 是 `Tensor` 的一个特殊子类,当它被赋值给 `nn.Module` 的属性时,它会自动被添加到模型的参数列表中. 
*   **模型状态管理**: 可以轻松地通过 `.train()` 和 `.eval()` 方法切换模型的工作模式(例如,这会影响 Dropout 和 BatchNorm 等层的行为). 
*   **设备迁移**: 通过调用 `.to(device)` 或 `.cuda()`,可以轻松地将整个模型(包括所有参数和缓冲区)移动到指定的计算设备(如 GPU)上. 
*   **状态保存与加载**: 使用 `.state_dict()` 和 `.load_state_dict()` 可以方便地保存和加载模型的状态,这对于**[模型检查点](./Lecture2-Checkpointing.md)**至关重要. 

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        # 定义模型的层
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, output_dim)

        # self.layer1.weight 和 self.layer1.bias 会被自动注册为模型参数

    def forward(self, x):
        # 定义前向传播逻辑
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 使用模型
model = SimpleModel(input_dim=784, output_dim=10)
print(model)

# 打印所有可训练参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)
```

### 4. 迈向现代化: `torch.compile`

从 PyTorch 2.0 开始,引入了一个革命性的功能: **[`torch.compile`](./Lecture2-torch-compile.md)**. 它通过即时编译(JIT)技术,将 Python 代码转换为更高效的优化内核,旨在大幅提升代码执行速度,同时只需极少的代码改动. `torch.compile` 整合了 TorchDynamo、AOTAutograd 和 TorchInductor 等多个后端技术,能够智能地捕获 PyTorch 代码的计算图并进行深度优化,有效减少了 Python 解释器的开销和 GPU 的读写次数. 

---
**关联知识点**
*   [张量 (Tensors)](./Lecture2-Tensors.md)
*   [Autograd](./Lecture2-Autograd.md)
*   [反向传播 (Backpropagation)](./Lecture2-Backpropagation.md)
*   [nn.Module](./Lecture2-nn-Module.md)
*   [torch.compile](./Lecture2-torch-compile.md)
*   [NumPy](./Lecture2-NumPy.md)