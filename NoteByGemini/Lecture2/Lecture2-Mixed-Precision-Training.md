# 专题笔记: 混合精度训练 (Mixed Precision Training)

### 1. 核心概念

**混合精度训练(Mixed Precision Training)** 是一种在深度学习训练中同时使用多种数值精度(主要是 **[FP32](./Lecture2-FP32-FP16-BF16-FP8.md)** 和 **[BF16/FP16](./Lecture2-FP32-FP16-BF16-FP8.md)**)的技术,旨在**在不牺牲模型准确性和稳定性的前提下,显著提升训练速度并减少内存占用**. 

它由 NVIDIA 在 2017 年提出,现已成为训练大型模型的标准实践. 其核心思想是利用现代 GPU(如 Tensor Cores)对低精度计算的硬件加速能力. 

### 2. 工作原理

混合精度训练并非简单地将所有数据和计算都切换到低精度,因为它需要精巧地处理数值稳定性问题. 一个标准的混合精度训练流程(尤其是在 PyTorch 的 `torch.cuda.amp` 中实现的)通常包含以下三个关键部分: 

1.  **维护 FP32 的主权重 (Master Weights)**
    *   模型参数的“主副本”始终以高精度(FP32)的形式存储在内存中. 这是为了确保在多次迭代中进行梯度更新时,能够精确地累积微小的梯度变化,避免舍入误差导致的训练失败. 

2.  **使用低精度进行计算 (Forward/Backward Pass)**
    *   在每次**前向传播**和**[反向传播](./Lecture2-Backpropagation.md)**的计算过程中,FP32 的主权重会被**临时转换(cast)**为一个低精度(如 BF16 或 FP16)的副本. 
    *   所有的计算密集型操作,如矩阵乘法和卷积,都在这个低精度的副本上进行. 这充分利用了 GPU Tensor Cores 的加速能力. 
    *   计算出的**[梯度](./Lecture2-Gradients.md)**也是低精度的. 

3.  **使用损失缩放 (Loss Scaling) 防止梯度下溢 (Underflow)**
    *   **问题**: 在使用 FP16 时,由于其数值范围很小,许多微小的梯度值可能会在转换过程中被舍入为零(梯度下溢),这会导致模型的部分参数停止学习. 
    *   **解决方案**: 在计算损失后、反向传播开始前,将损失值乘以一个很大的**缩放因子(scale factor)**,例如 2^16. 根据链式法则,这将使得所有梯度也同比例地放大. 
    *   这样,原本会下溢的微小梯度被放大到 FP16 可以表示的范围内,从而保留了它们的数值信息. 
    *   在优化器更新权重之前,会将缩放后的梯度再转换回 FP32,并**除以相同的缩放因子**,将其恢复到原始大小,然后才用于更新 FP32 的主权重. 
    *   **对于 BF16**: 由于 **[BF16](./Lecture2-FP32-FP16-BF16-FP8.md)** 拥有与 FP32 相同的动态范围,梯度下溢的问题大大缓解,因此**通常不需要(或可以禁用)损失缩放**. 这是 BF16 相对于 FP16 的一个巨大优势. 

### 3. PyTorch 中的实现

PyTorch 通过 `torch.cuda.amp` (Automatic Mixed Precision) 模块极大地简化了混合精度训练的实现. 开发者不再需要手动进行类型转换和损失缩放. 

**代码示例: **
```python
import torch

# 1. 导入 autocast 上下文管理器和 GradScaler
from torch.cuda.amp import autocast, GradScaler

# 模型、优化器、数据等...
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler() # 2. 创建一个 GradScaler 实例

for data, target in dataloader:
    optimizer.zero_grad()

    # 3. 使用 autocast 上下文管理器
    # autocast 会自动为其中的操作选择合适的精度(如 BF16 或 FP16)
    with autocast(dtype=torch.bfloat16): # 推荐使用 bfloat16
        output = model(data)
        loss = loss_fn(output, target)

    # 4. 使用 scaler.scale() 来缩放损失
    scaler.scale(loss).backward()

    # 5. 使用 scaler.step() 来 unscale 梯度并更新权重
    # 如果梯度没有发生 inf/nan,scaler.step 会调用 optimizer.step()
    scaler.step(optimizer)

    # 6. 更新缩放因子
    scaler.update()
```

### 4. 收益总结

*   **速度提升**: 通常能带来 1.5x 到 3x 的训练速度提升,具体取决于模型和硬件. 
*   **内存减少**: 显著降低模型激活值和梯度的内存占用,允许使用更大的模型或批处理大小. 
*   **保持准确性**: 通过维护 FP32 主权重和损失缩放等机制,混合精度训练能够在获得上述好处的同时,达到与纯 FP32 训练相当的收敛精度. 

---
**关联知识点**
*   [FP32 / FP16 / BF16 / FP8](./Lecture2-FP32-FP16-BF16-FP8.md)
*   [梯度 (Gradients)](./Lecture2-Gradients.md)
*   [优化器 (Optimizers)](./Lecture2-Optimizers.md)
*   [反向传播 (Backpropagation)](./Lecture2-Backpropagation.md)