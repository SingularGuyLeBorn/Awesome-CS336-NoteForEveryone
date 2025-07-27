# 专题笔记: 模型检查点 (Checkpointing)

### 1. 核心概念

**模型检查点 (Checkpointing)** 是在长时间的深度学习训练过程中,定期将训练状态的完整快照保存到持久化存储(如硬盘)中的关键实践. 

训练大型语言模型(LLM)可能需要数周甚至数月的时间. 在这个过程中,硬件故障、软件崩溃、断电等意外中断几乎是不可避免的. 如果没有检查点机制,任何一次中断都将意味着所有已经完成的训练付诸东流,必须从头开始,这将造成巨大的时间和金钱损失. 

检查点机制确保了训练可以从最近一次保存的状态无缝恢复,继续进行. 

### 2. 一个完整的检查点应该包含什么？

仅仅保存模型的权重(parameters)是不够的. 一个功能完备的检查点,需要能够让训练恢复到和中断前**完全一样**的状态,这通常需要包含以下信息: 

1.  **模型的状态字典 (`model.state_dict()`)**
    *   这是最重要的部分,包含了模型所有可学习的参数(权重和偏置). 

2.  **优化器的状态字典 (`optimizer.state_dict()`)**
    *   这同样至关重要. 像 **[AdamW](./Lecture2-Adam-AdamW.md)** 这样的现代**[优化器](./Lecture2-Optimizers.md)**是有状态的,它们内部维护着梯度的一阶矩(动量)和二阶矩的移动平均值. 如果不保存和恢复这些状态,优化器会从零开始,丢失掉历史梯度信息,这会导致训练恢复后出现剧烈的性能波动,并可能影响最终的收敛效果. 

3.  **训练元数据 (Metadata)**
    *   **迭代步数或周期数 (Iteration/Epoch)**: 记录当前训练进行到了哪一步,以便从正确的数据批次开始. 
    *   **学习率调度器状态 (`scheduler.state_dict()`)**: 如果使用了学习率衰减策略,也需要保存其状态. 
    *   **损失值 (Loss)**: 保存截至当前的最佳验证集损失或训练损失,有助于监控和决策. 
    *   **随机数生成器状态 (RNG State)**: 为了实现完全可复现的训练,需要保存 PyTorch、NumPy 和 Python 内置 `random` 模块的随机数生成器状态. 这可以确保数据加载的顺序、dropout 模式等在恢复后保持一致. 
    *   **混合精度缩放器状态 (`scaler.state_dict()`)**: 如果在使用**[混合精度训练](./Lecture2-Mixed-Precision-Training.md)**,也需要保存 `GradScaler` 的状态. 

### 3. PyTorch 中的实现

在 **[PyTorch](./Lecture2-PyTorch.md)** 中,保存和加载检查点通常通过以下步骤完成: 

**保存检查点: **
```python
import torch

# 假设 model, optimizer, epoch, loss 已经定义
# ...

# 1. 组织需要保存的信息到一个字典中
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    # 可选: 'scheduler_state_dict': scheduler.state_dict(),
    # 可选: 'scaler_state_dict': scaler.state_dict(),
}

# 2. 定义保存路径并保存
PATH = f"checkpoint_epoch_{epoch}.pt"
torch.save(checkpoint, PATH)

print(f"Checkpoint saved to {PATH}")
```

**加载检查点: **
```python
# 实例化模型和优化器
model = MyModel(*args, **kwargs)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# 1. 加载检查点文件
PATH = "checkpoint_epoch_10.pt"
checkpoint = torch.load(PATH)

# 2. 恢复状态
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"Checkpoint loaded. Resuming from epoch {epoch+1}.")

# 3. 将模型设置为训练模式
model.train()
# 或者评估模式
# model.eval()
```

### 4. 最佳实践

*   **定期保存**: 根据训练时长和稳定性,设置合理的保存频率. 例如,每几千次迭代或每小时保存一次. 
*   **保存多个检查点**: 不要只覆盖同一个检查点文件. 可以保存最近的 N 个检查点,或者在验证集性能提升时才保存“最佳”检查点. 这可以防止因保存过程中断而导致检查点文件损坏. 
*   **原子化写入**: 一个更健壮的做法是,先将检查点写入一个临时文件,写入成功后再将其重命名为最终文件名. 这可以防止在写入过程中断电导致文件不完整. 
*   **分布式训练**: 在多 GPU 或多节点训练中,通常只在主进程(rank 0)上执行保存操作,以避免文件写入冲突. 

检查点是任何严肃的深度学习项目的生命线,正确和健壮地实现检查点机制是保障项目顺利进行的基础. 

---
**关联知识点**
*   [优化器 (Optimizers)](./Lecture2-Optimizers.md)
*   [Adam / AdamW](./Lecture2-Adam-AdamW.md)
*   [混合精度训练 (Mixed Precision Training)](./Lecture2-Mixed-Precision-Training.md)
*   [PyTorch](./Lecture2-PyTorch.md)