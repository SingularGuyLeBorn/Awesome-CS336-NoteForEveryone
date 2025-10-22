# 代码实现深度解析: `tensor_parallelism_main` (TP 前向传播实现)

### 1. 核心功能与目标 (Core Function & Goal)
该函数演示了 **[张量并行 (TP)](./Lecture8-Tensor-Parallelism.md)** 在 MLP 层的**前向传播**过程。它展示了如何对权重矩阵进行列切分，在本地进行部分计算，然后通过通信聚合完整的激活值，以便传递给下一层。

### 2. 参数解析 (Parameters)
*   `rank`, `world_size`: 分布式进程信息。
*   `data`: 输入数据（在此示例中，输入数据被复制到了所有 Rank，实际中可能结合数据并行）。
*   `num_layers`: MLP 层数。

### 3. 核心逻辑 (Core Logic)

这里演示的是 Megatron-LM 风格中 MLP 第一层（$X \cdot W$）的**列并行（Column Parallelism）**及其后续处理。

```python
def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)
    data = data.to(get_device(rank)) # 假设输入数据已存在于每个设备
    batch_size = data.size(0)
    num_dim = data.size(1)
    
    # --- 1. 模型切分 (Model Sharding) ---
    # 计算每个 rank 负责的隐藏维度大小 (列切分)
    local_num_dim = int_divide(num_dim, world_size)
    
    # 初始化分片后的参数。
    # 完整矩阵形状: [num_dim, num_dim]
    # 本地切片形状: [num_dim, local_num_dim]
    # 每个 rank 持有完整权重矩阵的一部分列。
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]
    
    # --- 2. 前向传播 (Forward Pass) ---
    x = data # 输入形状: [batch_size, num_dim]
    for i in range(num_layers):
        # -- 本地计算 --
        # [batch_size, num_dim] @ [num_dim, local_num_dim] -> [batch_size, local_num_dim]
        # 得到的是部分激活值（Partial Activations）
        x = x @ params[i]
        x = F.gelu(x)
        
        # --- 3. 激活值聚合 (Communication) [TP 的核心] ---
        # 为了进行下一层的计算（其输入需要是完整的 num_dim 宽度），
        # 必须收集所有 rank 计算出的部分激活值并将它们沿列方向拼接。
        
        # 预分配缓冲区列表，用于接收来自所有 rank 的数据
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]
        
        # 执行 All-Gather：将当前的局部 x 发送给所有人，并接收所有人的 x 存入 activations 列表。
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)
        
        # 拼接：将列表中的分片沿维度 1 拼接，恢复为 [batch_size, num_dim]
        x = torch.cat(activations, dim=1)
        
        # 此时，x 恢复为完整维度，可以作为下一层的输入。
        # 注意：在实际 Megatron-LM 的 MLP 中，第二层通常使用行并行，
        # 配合 All-Reduce 而非 All-Gather，这里仅为演示原理。
        
    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations ...", flush=True)
    # 反向传播逻辑更为复杂，涉及梯度的切分与聚合，此处略去。
    cleanup()
```

### 4. 与理论的连接 (Connection to Theory)
*   精确实现了 **[张量并行](./Lecture8-Tensor-Parallelism.md)** 中关于权重矩阵切分的描述（此处为列切分）。
*   展示了 TP 必须在每一层（for 循环内部）进行通信。
*   使用的通信原语是 **[All-Gather](./Lecture8-Collective-Operations.md)**，用于在所有设备上重建完整的激活张量。
*   由于每一层都要传输 `batch_size * num_dim` 级别的数据，这解释了为什么 TP 需要极高的 **[硬件通信带宽](./Lecture8-Hardware-Hierarchy.md)**（如 NVLink）。