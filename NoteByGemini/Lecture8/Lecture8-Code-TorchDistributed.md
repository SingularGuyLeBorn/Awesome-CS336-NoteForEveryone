# 代码实现深度解析: `torch.distributed` 原语与基准测试

### 1. 核心功能与目标 (Core Function & Goal)
本部分代码展示了如何使用 PyTorch 的 `torch.distributed` 模块执行核心的 **[集合通信操作](./Lecture8-Collective-Operations.md)**（All-Reduce, Reduce-Scatter, All-Gather），并通过基准测试函数测量这些操作在实际硬件（假定为基于 **[NCCL](./Lecture8-NCCL.md)** 的 GPU 环境）上的有效带宽。这是构建高级分布式训练策略的基础。

### 2. 关键函数解析
该文件包含多个演示和测试函数，核心在于对 `torch.distributed` (别名 `dist`) API 的调用。所有函数都运行在 **[SPMD](./Lecture8-SPMD.md)** 模式下。

*   `collective_operations_main(rank, world_size)`: 演示通信原语的功能逻辑。
*   `all_reduce(rank, world_size, num_elements)`: 对 All-Reduce 进行基准测试。
*   `reduce_scatter(rank, world_size, num_elements)`: 对 Reduce-Scatter 进行基准测试。

### 3. 核心逻辑 (Core Logic)

#### 3.1 初始化与同步
在进行任何通信前，必须初始化进程组。`dist.barrier()` 用于同步所有进程，常用于基准计时前确保所有进程处于同一起跑线。

```python
# 位于 setup 函数中
# 初始化进程组，使用 NCCL 后端（针对 GPU）
dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 在操作前后使用 barrier 进行同步
dist.barrier()
```

#### 3.2 集合操作演示 (`collective_operations_main`)

此函数直观展示了数据在操作前后的变化。

```python
def collective_operations_main(rank: int, world_size: int):
    setup(rank, world_size)
    
    # --- 演示 All-Reduce ---
    dist.barrier()
    # 创建一个 tensor，其值依赖于 rank。例如 world_size=4:
    # Rank 0: [0, 1, 2, 3]
    # Rank 1: [1, 2, 3, 4] ...
    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank
    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)
    
    # 执行 All-Reduce (求和)。这是一个原位(in-place)操作。
    # 操作后，所有 rank 上的 tensor 变为所有初始 tensor 之和。
    # 结果应为: [0+1+2+3, 1+2+3+4, ...] = [6, 10, 14, 18]
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)

    # --- 演示 Reduce-Scatter ---
    dist.barrier()
    # 输入张量大小为 world_size。
    input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank
    # 输出张量大小为 1 (输入大小 / world_size)。
    output = torch.empty(1, device=get_device(rank))
    
    # Reduce-Scatter: 先对 input 求和，然后将结果切分并散发到 output。
    # Rank 0 的 output 接收 sum(input)[0], Rank 1 接收 sum(input)[1]...
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    
    # --- 演示 All-Gather ---
    dist.barrier()
    input = output  # 将上一步 Reduce-Scatter 的输出作为这一步的输入
    output = torch.empty(world_size, device=get_device(rank)) # 分配空间接收汇总后的数据
    
    # All-Gather: 收集所有 rank 的 input，拼接到 output 中。
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    # 此时，output 的值应该恢复为之前 All-Reduce 的结果，
    # 证明了 All-Reduce = Reduce-Scatter + All-Gather。
    
    cleanup()
```

#### 3.3 带宽基准测试 (`all_reduce` 示例)

基准测试的关键在于准确计时和计算传输的数据量。需要注意 GPU 操作的异步性，必须使用 `torch.cuda.synchronize()` 确保内核执行完毕。

```python
def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)
    # 创建大张量 (例如 100M 元素)
    tensor = torch.randn(num_elements, device=get_device(rank))
    
    # 热身 (Warmup)：确保通信器初始化，缓存加载等
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize() # 等待 CUDA 内核完成
        dist.barrier()           # 等待所有进程完成热身

    # 开始计时
    start_time = time.time()
    # 执行实际操作
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    # 停止计时前必须同步
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        dist.barrier()
    end_time = time.time()
    duration = end_time - start_time

    # --- 带宽计算 ---
    # 单个张量的字节数
    size_bytes = tensor.element_size() * tensor.numel()
    # All-Reduce 的通信量计算：基于 Ring 算法，每个节点发送和接收的数据量约为 2 * size_bytes * (N-1)/N。
    # 这里的简化计算：总线上的传输量视为 2 * size_bytes (发送输入，接收规约后的输出，针对N很大时的近似)
    # 讲义代码中的计算方式：
    sent_bytes = size_bytes * 2 * (world_size - 1) 
    # 这种计算方式可能特定于某种特定的带宽定义，通常 Ring All-Reduce 每个 rank 的通信量是 2 * size * (n-1)/n
    
    # 总时间视作所有进程时间的总和 (这是一个视角问题，也可以计算算法带宽)
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    
    if rank == 0:
        print(f"All-reduce effective bandwidth: {bandwidth / 1024**3:.2f} GB/s")
    cleanup()
```
*注：代码中对 `Reduce-Scatter` 的带宽计算没有乘以 2，这符合理论，因为 Reduce-Scatter 的通信量约为 All-Reduce 的一半。*

### 4. 与理论的连接 (Connection to Theory)
*   此代码直接实现了 **[集合通信操作](./Lecture8-Collective-Operations.md)** 理论笔记中描述的原语。
*   `dist.init_process_group("nccl", ...)` 明确使用了 **[NCCL](./Lecture8-NCCL.md)** 作为后端，利用 **[硬件通信层级](./Lecture8-Hardware-Hierarchy.md)** 中的 NVLink/PCIe 进行通信。
*   代码结构遵循 **[SPMD](./Lecture8-SPMD.md)** 范式，所有 Rank 运行相同代码但处理各自的数据。
*   通过代码演示了逻辑等式：**All-Reduce = Reduce-Scatter + All-Gather**。