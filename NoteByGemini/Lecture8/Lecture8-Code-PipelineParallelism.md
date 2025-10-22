# 代码实现深度解析: `pipeline_parallelism_main` (PP 前向传播实现)

### 1. 核心功能与目标 (Core Function & Goal)
该函数演示了 **[流水线并行 (PP)](./Lecture8-Pipeline-Parallelism.md)** 的基本机制。它将模型层分配给不同的 Rank，引入 **[微批次 (Micro-batches)](./Lecture8-Micro-batches.md)** 概念，并使用点对点通信原语在 Rank 之间传递中间激活值，模拟了数据在流水线中的流动。

### 2. 参数解析 (Parameters)
*   `world_size`: 流水线的阶段（Stage）数。
*   `num_layers`: 模型总层数。
*   `num_micro_batches`: 将大批次切分为多少个微批次。

### 3. 核心逻辑 (Core Logic)

这是一个简化的实现，展示了数据如何按顺序流经不同的 Rank。

```python
def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)
    data = data.to(get_device(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)
    
    # --- 1. 模型分层 (Layer Sharding) ---
    # 计算每个 rank 负责的层数
    local_num_layers = int_divide(num_layers, world_size)
    # 初始化属于当前 rank 的那部分层的参数
    # 例如: Rank 0 拥有 Layers 0-1, Rank 1 拥有 Layers 2-3
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]
    
    # --- 2. 微批次处理 (Micro-batching) ---
    # 计算微批次大小
    micro_batch_size = int_divide(batch_size, num_micro_batches)
    
    if rank == 0:
        # 流水线的源头：将输入数据切分为微批次列表
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # 后续阶段：只分配用于接收激活值的缓冲区，不持有原始数据
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]
        
    # --- 3. 流水线执行循环 ---
    # 依次处理每个微批次。此处为简化的同步实现，实际应异步重叠通信与计算。
    for i, x in enumerate(micro_batches):
        # -- 接收输入 (Receive) --
        if rank > 0:
            # 如果不是第一个 rank，从上一个 rank 接收激活值
            # dist.recv 是点对点通信
            dist.recv(tensor=x, src=rank - 1)
            # 对于 rank 0，x 已经是数据切片了
            
        # -- 本地计算 (Compute) --
        # 数据流经分配给当前 rank 的所有层
        for param in local_params:
            x = x @ param
            x = F.gelu(x)
            
        # -- 发送输出 (Send) --
        if rank < world_size - 1:
            # 如果不是最后一个 rank，将计算结果发送给下一个 rank
            print(f"[pipeline_parallelism] Rank {rank}: sending ... to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)
        else:
            # 最后一个 rank 持有最终输出，此处可计算 Loss 并开始反向传播
            pass
            
    # 注意：此实现未包含反向传播和复杂的流水线调度（如 1F1B），
    # 也没有展示异步通信(`isend`, `irecv`)带来的计算通信重叠。
    cleanup()
```

### 4. 与理论的连接 (Connection to Theory)
*   实现了 **[流水线并行](./Lecture8-Pipeline-Parallelism.md)** 的模型按层切分策略。
*   使用了 **[微批次](./Lecture8-Micro-batches.md)** (`data.chunk`)，这是缓解 **[流水线气泡](./Lecture8-Pipeline-Parallelism.md#流水线气泡-pipeline-bubbles)** 的核心技术。虽然代码是同步执行的，但循环结构为异步调度打下了基础。
*   展示了 PP 依赖于点对点通信 (`dist.send`, `dist.recv`) 在相邻 Stage 间传递数据边界。