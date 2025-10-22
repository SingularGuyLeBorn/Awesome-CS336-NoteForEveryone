# 代码实现深度解析: `data_parallelism_main` (DDP 实现)

### 1. 核心功能与目标 (Core Function & Goal)
该函数在一个简单的 MLP 模型上实现了基础的 **[数据并行 (DDP)](./Lecture8-Data-Parallelism.md)** 训练循环。其核心目标是展示如何在标准 SGD 流程中插入梯度同步步骤，使得在不同数据分片上训练的多个模型副本能够保持参数一致。

### 2. 参数解析 (Parameters)
*   `rank` (int): 当前进程的 ID。
*   `world_size` (int): 总进程数（GPU 数）。
*   `data` (torch.Tensor): 完整的输入数据集。
*   `num_layers` (int): MLP 的层数。
*   `num_steps` (int): 训练步数。

### 3. 核心逻辑 (Core Logic)

该实现运行在 **[SPMD](./Lecture8-SPMD.md)** 模式下。

```python
def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)
    
    # --- 1. 数据分片 (Data Sharding) ---
    batch_size = data.size(0)
    num_dim = data.size(1)
    # 计算每个 rank 应处理的批次大小
    local_batch_size = int_divide(batch_size, world_size)
    # 计算当前 rank 数据的起始和结束索引
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    # 获取本地数据切片并移动到对应的 GPU
    # 在实际应用中，每个 rank 通常使用 DistributedSampler 直接从磁盘读取属于自己的数据
    data = data[start_index:end_index].to(get_device(rank))
    
    # --- 2. 模型初始化 (Model Replication) ---
    # 每个 rank 初始化一组相同的参数（依赖相同的随机种子）
    # 参数维度是完整的 (num_dim x num_dim)
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    # 每个 rank 拥有独立的优化器实例
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    # --- 3. 训练循环 ---
    for step in range(num_steps):
        # -- 前向传播 --
        x = data
        for param in params:
            x = x @ param # 矩阵乘法
            x = F.gelu(x) # 激活函数
        # 计算由本地数据产生的局部损失
        loss = x.square().mean()
        
        # -- 反向传播 --
        # 计算本地梯度
        loss.backward()
        
        # --- 4. 梯度同步 (Gradient Synchronization) [DDP 的核心] ---
        # 在优化器更新参数之前，必须同步所有 rank 的梯度。
        for param in params:
            # 使用 All-Reduce 操作计算梯度的平均值。
            # 原位修改 param.grad。操作后，所有 rank 的 param.grad 变为全局平均梯度。
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
        
        # -- 参数更新 --
        # 使用同步后的全局平均梯度更新参数。
        # 由于初始参数相同，梯度相同，更新后的参数依然保持一致。
        optimizer.step()
        
        # 打印信息：注意不同 rank 的 loss 不同，但 params 摘要应相同
        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = ...", flush=True)
    
    cleanup()
```

### 4. 与理论的连接 (Connection to Theory)
*   代码清晰地实现了 **[数据并行](./Lecture8-Data-Parallelism.md)** 的逻辑：数据被切分 (`data[...]`)，模型被复制（全尺寸 `params`）。
*   核心在于利用 **[All-Reduce](./Lecture8-Collective-Operations.md#all-reduce)** 原语实现梯度的平均化。这是保证数学上等价于大批次单机训练的关键步骤。
*   体现了 DDP 的通信开销发生在反向传播之后、参数更新之前，且传输的是梯度数据。