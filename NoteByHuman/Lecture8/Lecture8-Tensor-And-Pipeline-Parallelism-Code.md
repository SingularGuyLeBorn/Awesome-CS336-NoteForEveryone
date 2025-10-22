# 精英笔记：张量与流水线并行代码实现

当单个模型的参数量或中间激活值大到无法放入单张GPU显存时，就需要模型并行。本笔记将并列解析两种核心的模型并行策略——张量并行和流水线并行——的代码实现，并对比它们的运作机制与通信模式。

## 1. 张量并行 (Tensor Parallelism)

张量并行的核心思想是将模型中的大型权重矩阵（如Transformer中的`W_q, W_k, W_v`或MLP层）沿其宽度（或高度）维度进行切分，每个GPU只持有一部分权重。计算时，每个GPU用完整的输入数据和自己的权重分片进行局部计算，然后通过集体通信操作（通常是`All-Gather`）来还原完整的输出结果，以供下一层使用。

### 代码解析: `tensor_parallelism_main` (前向传播)

```python
def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)
    data = data.to(get_device(rank))

    # 1. 维度与参数分片
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_num_dim = int_divide(num_dim, world_size) # 将隐藏维度切分
    # 每个rank的权重矩阵列数只有 1/world_size
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    # 2. 训练循环 (前向传播)
    x = data
    for i in range(num_layers):
        # 2a. 局部矩阵乘法
        # (batch, num_dim) @ (num_dim, local_num_dim) -> (batch, local_num_dim)
        x_local = x @ params[i]
        x_local = F.gelu(x_local)

        # 2b. 全局通信 (All-Gather)
        # 准备一个列表来接收所有rank的数据
        activations = [torch.empty_like(x_local) for _ in range(world_size)]
        dist.all_gather(tensor_list=activations, tensor=x_local, async_op=False)

        # 2c. 拼接还原完整激活
        # 将4个 (batch, local_num_dim) 的张量沿维度1拼接
        x = torch.cat(activations, dim=1) # 结果变回 (batch, num_dim)
        
    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)
    cleanup()
```

### 张量流动分析
以 `world_size=4`, `batch_size=128`, `num_dim=1024` 为例，追踪激活张量 `x` 在一层计算中的变化：

1.  **层输入**: `x` 的形状为 `[128, 1024]`。此张量在所有4个rank上都是完整且相同的。
2.  **局部计算**: `params[i]` 的形状为 `[1024, 256]`。
    *   `x_local = x @ params[i]` 的结果形状为 `[128, 256]`。
    *   此时，`rank 0` 拥有完整1024维激活的前256维，`rank 1` 拥有256-511维，以此类推。
3.  **`All-Gather`通信**:
    *   `dist.all_gather` 将所有4个rank上的 `[128, 256]` 张量收集起来。
    *   操作完成后，`activations` 列表在**每个rank**上都包含了4个 `[128, 256]` 的张量。
4.  **拼接还原**:
    *   `x = torch.cat(activations, dim=1)` 将这4个张量沿第二个维度（`dim=1`）拼接。
    *   `x` 的形状恢复为 `[128, 1024]`，并且在所有rank上都相同。这个恢复后的完整张量可以作为下一层的输入。

**关键点**: 张量并行在**每一层**的计算之后都必须伴随一次 `All-Gather` 通信，通信频率非常高。这要求极高的设备间互联带宽（如NVLink），否则通信开销将成为主要瓶颈。

---

## 2. 流水线并行 (Pipeline Parallelism)

流水线并行的核心思想是模型的不同层（按深度）分配到不同的GPU上。数据像在流水线上一样，依次通过各个GPU。GPU 0计算完第0-7层后，将结果传递给GPU 1，GPU 1再计算第8-15层，以此类推。为了提高设备利用率，通常将大批次数据切分为多个**微批次(micro-batches)**，使得后一个GPU可以提前开始处理前一个GPU完成的第一个微批次，从而实现部分计算的重叠。

### 代码解析: `pipeline_parallelism_main` (前向传播)

```python
def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)
    data = data.to(get_device(rank))
    
    # 1. 层分配
    local_num_layers = int_divide(num_layers, world_size)
    # 每个rank只初始化自己负责的层
    local_params = [get_init_params(data.size(1), data.size(1), rank) for i in range(local_num_layers)]

    # 2. 微批次切分
    micro_batch_size = int_divide(data.size(0), num_micro_batches)
    if rank == 0:
        # 第一个rank直接切分输入数据
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # 其他rank准备空张量来接收激活
        micro_batches = [torch.empty(micro_batch_size, data.size(1), device=get_device(rank)) for _ in range(num_micro_batches)]

    # 3. 流水线式前向传播
    for x in micro_batches:
        # 3a. 从上一个rank接收激活值 (点对点通信)
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # 3b. 在本地层上计算
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # 3c. 发送激活值到下一个rank (点对点通信)
        if rank + 1 < world_size:
            dist.send(tensor=x, dst=rank + 1)

    cleanup()
```

### 张量流动分析
以 `world_size=2`, `num_micro_batches=4` 为例，追踪第一个微批次`micro_batches[0]`的流动：

1.  **Rank 0**:
    *   输入: `micro_batches[0]` 是原始数据的一个切片。
    *   计算: `micro_batches[0]` 依次通过 `rank 0` 负责的所有`local_params`。
    *   通信: 计算完成后，通过 `dist.send(tensor=micro_batches[0], dst=1)` 将结果发送给 `rank 1`。

2.  **Rank 1**:
    *   通信: 主循环开始，执行 `dist.recv(tensor=micro_batches[0], src=0)`，这是一个阻塞操作，会一直等待直到从`rank 0`接收到数据，并将数据填充到预先分配的空张量`micro_batches[0]`中。
    *   计算: 接收到数据后，`rank 1` 开始用它自己的`local_params`进行计算。
    *   通信: 由于是最后一个rank，它不再发送数据。计算结果就是该微批次的最终输出。

**关键点**:
*   流水线并行使用**点对点通信** (`send`/`recv`)，而非集体通信。
*   通信发生在模型的**阶段（stage）之间**，而不是每一层之后。通信频率远低于张量并行。
*   主要挑战是**流水线气泡**：在流水线的启动和排空阶段，部分GPU会处于空闲状态。使用更多的微批次可以减小气泡的相对大小，但也会增加调度的开销。

## 3. 对比总结

| 特性 | 张量并行 (Tensor Parallelism) | 流水线并行 (Pipeline Parallelism) |
| :--- | :--- | :--- |
| **切分维度** | 模型宽度 (Hidden Dimension) | 模型深度 (Layers) |
| **通信模式** | 集体通信 (`All-Gather`, `All-Reduce`) | 点对点通信 (`Send`, `Recv`) |
| **通信频率** | **非常高** (每层计算后) | **较低** (每个Stage计算后) |
| **硬件要求** | 极高的节点内带宽 (e.g., NVLink) | 对节点间带宽要求相对较低 |
| **主要挑战** | 通信开销成为瓶颈 | 流水线气泡导致设备利用率下降 |
| **适用场景** | 单个算子（如大矩阵乘法）过大 | 模型层数非常深 |

在实践中，这两种策略经常被结合使用。例如，在一个节点内部的8张GPU之间使用张量并行，而在节点之间使用流水线并行，以充分利用不同层级硬件的通信能力。