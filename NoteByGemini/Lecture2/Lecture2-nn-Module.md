# 专题笔记: nn.Module

### 1. 核心概念

在 **[PyTorch](./Lecture2-PyTorch.md)** 中,`torch.nn.Module` 是所有神经网络层和模型的基本构建块(building block). 它是一个功能强大的基类,任何你想要自定义的模型或层都应该继承自它. 

你可以将 `nn.Module` 想象成一个可包含其他 `Module` 的容器. 这些模块以树状结构组织起来. 一个完整的模型是一个大的 `nn.Module`,它包含了代表各层的较小的 `nn.Module`(如 `nn.Linear`, `nn.Conv2d`),而这些层本身也是 `nn.Module`. 

### 2. `nn.Module` 的核心功能

继承自 `nn.Module` 会为你的类自动提供许多至关重要的功能: 

1.  **参数追踪 (Parameter Tracking)**: 
    *   当你将一个 `torch.nn.Parameter` 类的实例赋值给 `nn.Module` 的一个属性时,这个参数会自动被注册到该模块的参数列表中. `nn.Parameter` 是 **[张量](./Lecture2-Tensors.md)** 的一个特殊子类,其 `requires_grad` 默认为 `True`. 
    *   通过调用 `model.parameters()` 或 `model.named_parameters()`,你可以轻松地获取到模型及其所有子模块中所有已注册的参数. 这对于将参数传递给**[优化器](./Lecture2-Optimizers.md)**至关重要. 

2.  **状态管理 (State Management)**: 
    *   `model.train()`: 将模型及其所有子模块设置为**训练模式**. 这会激活像 Dropout 和 BatchNorm 这样的层. 
    *   `model.eval()`: 将模型及其所有子模块设置为**评估/推理模式**. 这会关闭 Dropout,并使用 BatchNorm 在训练时计算出的运行均值和方差. 在进行验证或测试时,必须调用此方法. 

3.  **子模块管理 (Submodule Management)**: 
    *   当你在一个 `nn.Module` 的 `__init__` 方法中将另一个 `nn.Module` 赋值给一个属性时,这个子模块会被自动注册. 
    *   所有对父模块的操作(如 `.cuda()`, `.to()`, `.state_dict()`)都会递归地应用到所有子模块上. 

4.  **状态字典 (State Dictionary)**: 
    *   `model.state_dict()`: 返回一个字典,其中包含了模型所有可学习参数(权重和偏置)以及持久化缓冲区(persistent buffers,如 BatchNorm 的运行统计量)的状态. 这是实现**[模型检查点](./Lecture2-Checkpointing.md)**的核心. 
    *   `model.load_state_dict()`: 从一个状态字典中加载模型的状态. 

### 3. 如何构建一个自定义模型

要使用 `nn.Module` 构建一个自定义模型,你需要遵循以下两个步骤: 

1.  **在 `__init__` 方法中定义层**: 
    *   继承 `nn.Module`. 
    *   在构造函数 `__init__` 中,首先必须调用 `super().__init__()`. 
    *   然后,实例化你模型中需要用到的所有层(它们本身也是 `nn.Module`),并将它们作为类的属性. 

2.  **在 `forward` 方法中定义计算逻辑**: 
    *   实现 `forward(self, input)` 方法. 这个方法接收输入张量,并定义了数据如何流经你在 `__init__` 中定义的各个层,最终返回模型的输出. 
    *   当你使用 `model(input_data)` 来调用模型时,PyTorch 内部会自动调用你所定义的 `forward` 方法. 你不应该直接调用 `model.forward(input_data)`. 

**代码示例: 一个简单的多层感知机 (MLP)**
```python
import torch
import torch.nn as nn

class MyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # 步骤 1: 定义层
        super(MyMLP, self).__init__() # 必须先调用父类的构造函数

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # self.fc1 和 self.fc2 中的权重和偏置被自动注册

    def forward(self, x):
        # 步骤 2: 定义前向传播逻辑
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 实例化模型
model = MyMLP(input_size=784, hidden_size=500, num_classes=10)
print(model)

# 打印所有可训练参数
print("\nTrainable Parameters:")
for name, param in model.named_parameters():
    print(f"Layer: {name}, Size: {param.size()}")

# 将模型移动到 GPU (如果可用)
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    print(f"\nModel moved to {device}")
```

通过使用 `nn.Module`,我们可以构建出结构清晰、易于管理和调试的复杂模型,同时利用 PyTorch 框架提供的所有强大功能. 

---
**关联知识点**
*   [PyTorch](./Lecture2-PyTorch.md)
*   [张量 (Tensors)](./Lecture2-Tensors.md)
*   [优化器 (Optimizers)](./Lecture2-Optimizers.md)
*   [模型检查点 (Checkpointing)](./Lecture2-Checkpointing.md)