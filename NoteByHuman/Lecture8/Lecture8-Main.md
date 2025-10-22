# CS336 第8讲：手撕大模型并行训练

上周我们探讨了单个GPU内部的并行计算，本周我们将视野扩展到跨越多GPU和多节点的并行训练。

### 核心挑战：数据传输瓶颈

首先，我们脑海中应该有这样一幅硬件层级图景：我们有多个计算节点（Node），每个节点内有多张GPU（通常是8张）。每张GPU内部又有多个流式多处理器（Streaming Multiprocessors, SMs）来执行实际的计算任务。

![GPU节点概览图](placeholder.png)
**[插入图片: lecture_08.py, image call "images/gpu-node-overview.png", 描述: 展示了从节点到GPU再到SM的硬件层级结构，以及L1缓存、高带宽内存HBM和GPU间连接。]**

计算发生在SM中的算术逻辑单元（ALU）上，而计算所需的输入数据和产生的输出结果通常存储在远离计算单元的内存中。无论是单GPU还是多GPU场景，一个共同的主题始终是如何编排计算以避免数据传输瓶颈。

**广义的存储与通信层级 (从最小/最快到最大/最慢):**
1.  **单节点，单GPU内部**: L1缓存/共享内存 (极快，但容量极小)
2.  **单节点，单GPU内部**: 高带宽内存 (HBM)
3.  **单节点，多GPU之间**: NVLink (直接连接GPU，绕过CPU)
4.  **多节点，多GPU之间**: NVSwitch (跨节点直接连接GPU，绕过以太网)

上周我们学习了通过融合（Fusion）和切块（Tiling）等技术，在L1缓存/共享内存这个层面进行优化，以减少对HBM的访问。本周，我们将聚焦于跨GPU和节点的通信，通过复制（Replication）和分片（Sharding）模型、参数及优化器状态来解决问题。

本讲的目标是将上节课的理论概念在代码中具体化，从而更深入地理解其工作原理。

## 第一部分：分布式通信与计算的基石

### **集体通信操作 (Collective Operations)**

分布式编程的核心是**集体通信操作**。这些是用于在多个设备（或进程）之间协调数据交换的标准化原语。它们诞生于上世纪80年代的并行计算领域，相比于开发者自己管理复杂的点对点通信，集体操作提供了更高层、更高效的抽象。

**基本术语:**
*   **World Size**: 参与分布式任务的设备总数。例如，world size为4意味着有4个GPU参与。
*   **Rank**: 每个设备的唯一标识符。如果world size为4，则设备的rank通常为0, 1, 2, 3。

**常见的集体操作：**

*   **Broadcast (广播)**: 将rank 0上的一个张量复制并发送给所有其他ranks。
    ![Broadcast示意图](placeholder.png)
*   **Scatter (分散)**: 将rank 0上的一个张量切分成多块，并将每一块分别发送给一个rank。每个rank接收到的数据是不同的。
    ![Scatter示意图](placeholder.png)
*   **Gather (收集)**: Scatter的逆操作。将所有rank上的张量收集到一个目标rank上，并沿指定维度拼接起来。
    ![Gather示意图](placeholder.png)
*   **Reduce (规约)**: 与Gather类似，也是将所有rank的数据汇集到一个rank，但不是拼接，而是对它们执行一个规约操作（如求和、求最大/最小值）。
    ![Reduce示意图](placeholder.png)
*   **All-gather (全局收集)**: 与Gather类似，但目标不是单个rank，而是所有rank。操作完成后，每个rank都拥有了所有其他rank数据的完整拼接副本。
    ![All-gather示意图](placeholder.png)
*   **Reduce-scatter (规约分散)**: 这是一个复合操作。首先像Reduce一样，对所有rank的数据进行规约（如求和），然后像Scatter一样，将规约后的结果切分并分散给所有rank。
    ![Reduce-scatter示意图](placeholder.png)
*   **All-reduce (全局规约)**: 同样是复合操作，相当于先执行Reduce操作，再执行Broadcast操作。最终，所有rank都拥有了对全体数据进行规约后的相同结果。
    ![All-reduce示意图](placeholder.png)

一个重要的恒等式是：**All-reduce = Reduce-scatter + All-gather**。

**[深入探讨: 集体通信操作 (Collective Operations)](./Lecture8-Collective-Operations.md)**

### 分布式通信的软硬件实现

#### 硬件层面

*   **传统架构 (家用/早期)**:
    *   同一节点内的GPU通过PCIe总线与CPU通信，GPU间通信需要CPU中转。
    *   不同节点间的通信依赖于慢速的以太网。
    这种架构开销巨大，因为数据传输路径长且效率低下。

*   **现代数据中心架构**:
    *   **NVLink**: 在节点内部直接连接GPU，绕过CPU，极大提升了GPU间的通信带宽。
    *   **NVSwitch**: 在节点之间直接连接GPU集群，绕过以太网，实现了跨节点的高速互联。
    以H100 GPU为例，它拥有18个NVLink 4.0链路，总带宽高达900GB/s，远超PCIe和以太网。当然，这仍低于HBM的内存带宽（约3.9 TB/s）。

#### 软件层面

*   **NVIDIA Collective Communication Library (NCCL)**:
    NVIDIA提供的底层通信库，它将上层应用的集体操作（如`all_reduce`）翻译成底层的、针对硬件拓扑优化的网络包。NCCL会自动检测硬件连接（NVLink, PCIe等），优化GPU间的通信路径，并启动CUDA核心来发送和接收数据。

*   **PyTorch Distributed (`torch.distributed`)**:
    PyTorch为分布式训练提供的上层库，它封装了NCCL等后端。
    *   **简洁的接口**: 开发者可以在Python代码中轻松调用`dist.all_gather()`等函数。
    *   **多后端支持**: 支持`nccl` (用于GPU) 和 `gloo` (用于CPU) 等不同后端，使得代码可以在没有GPU的环境中（如笔记本电脑上）进行逻辑调试。
    *   **高层封装**: 还提供了如`FullyShardedDataParallel (FSDP)`等更高级的并行策略封装（本课程为深入原理，不直接使用）。

### PyTorch集体操作代码实例

下面的代码 (`collective_operations_main`) 展示了如何在PyTorch中实际使用这些操作。该函数会被`spawn`工具函数在4个独立的进程中并行执行，每个进程代表一个rank。

```python
def collective_operations_main(rank: int, world_size: int):
    """此函数在每个进程中异步运行 (rank = 0, ..., world_size - 1)"""
    setup(rank, world_size)  # 初始化分布式环境

    # --- All-reduce 示例 ---
    dist.barrier()  # 同步点，等待所有进程到达这里
    # 每个rank创建一个不同的张量
    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank
    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)
    # 执行全局规约（求和），结果会原地更新到tensor
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)

    # --- Reduce-scatter 示例 ---
    dist.barrier()
    input_tensor = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank
    output_tensor = torch.empty(1, device=get_device(rank))
    print(f"Rank {rank} [before reduce-scatter]: input = {input_tensor}, output = {output_tensor}", flush=True)
    # 执行规约分散，每个rank得到总和向量的一个分量
    dist.reduce_scatter_tensor(output=output_tensor, input=input_tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} [after reduce-scatter]: input = {input_tensor}, output = {output_tensor}", flush=True)

    # --- All-gather 示例 ---
    dist.barrier()
    # 将reduce-scatter的输出作为all-gather的输入
    input_tensor_ag = output_tensor
    output_tensor_ag = torch.empty(world_size, device=get_device(rank))
    print(f"Rank {rank} [before all-gather]: input = {input_tensor_ag}, output = {output_tensor_ag}", flush=True)
    # 执行全局收集，每个rank都得到完整的向量
    dist.all_gather_into_tensor(output_tensor=output_tensor_ag, input_tensor=input_tensor_ag)
    print(f"Rank {rank} [after all-gather]: input = {input_tensor_ag}, output = {output_tensor_ag}", flush=True)

    cleanup() # 清理分布式环境
```

从输出可以看到，先执行`reduce_scatter`再执行`all_gather`，得到的结果与直接执行`all_reduce`完全相同，验证了 **All-reduce = Reduce-scatter + All-gather** 的关系。

### **基准测试 (Benchmarking)**

理论带宽和实际性能总有差距，因此对通信操作进行基准测试非常重要。我们通过运行`all_reduce`和`reduce_scatter`来测量单节点内4张GPU间的有效带宽。

*   **`all_reduce`测试**: 测量结果显示有效带宽约为 **277 GB/s**。
*   **`reduce_scatter`测试**: 测量结果显示有效带宽约为 **70 GB/s**。

`all_reduce`的带宽远高于`reduce_scatter`，这可能是因为NVIDIA硬件中有针对`all_reduce`的专门优化，例如SHARP技术，它可以在网络交换机内部完成部分规约计算，效率更高。这也再次说明了，底层实现细节会对最终性能产生巨大影响。

## 第二部分：分布式训练策略

我们将通过在深度MLP上实现三种主要的并行策略，来理解它们的核心思想。虽然MLP结构简单，但它们是Transformer中的计算瓶颈，因此这些实现具有很好的代表性。

### **数据并行 (Data Parallelism)**

![数据并行示意图](placeholder.png)
**[插入图片: lecture_08.py, image call "images/data-parallelism.png", 描述: 模型被完整复制到每个设备上，而数据（批次维度）被切分，每个设备处理一部分数据。]**

**分片策略**: 将数据（沿着batch维度）切分，每个rank获得一小批（mini-batch）数据。模型参数在所有rank上完全复制。

**`data_parallelism_main` 函数核心逻辑**:
1.  **数据分片**: 根据rank索引，从全局数据`data`中获取属于自己的`local_batch_size`大小的数据切片。
2.  **模型和优化器**: 每个rank独立创建完整的模型参数`params`和优化器`optimizer`。
3.  **训练循环**:
    *   **前向传播**: 在本地数据上正常计算损失`loss`。
    *   **反向传播**: 正常计算梯度。
    *   **梯度同步**: 这是与单机训练唯一的区别。对每个参数的梯度`param.grad`，调用`dist.all_reduce`操作求平均值。
    *   **参数更新**: 使用同步后的梯度更新本地的模型参数。

```python
# 数据并行核心代码片段
def data_parallelism_main(rank: int, world_size: int, ...):
    # ... 省略设置和数据分片代码 ...
    params = [...]  # 每个rank有完整的模型副本
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    
    for step in range(num_steps):
        # ... 前向传播和计算loss ...
        loss.backward()
        
        # 核心步骤：同步所有worker的梯度
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)
            
        optimizer.step()
    # ...
```

**关键点**:
*   由于每个rank处理的数据不同，计算出的`loss`也是不同的。
*   通过在梯度上执行`all_reduce`，所有rank获得了完全相同的平均梯度。
*   由于初始参数相同，且每次都使用相同的梯度进行更新，因此所有rank上的模型参数在整个训练过程中始终保持一致。

**[深入探讨: 数据并行 DDP 代码实现](./Lecture8-Data-Parallelism-Code.md)**

### **张量并行 (Tensor Parallelism)**

![张量并行示意图](placeholder.png)
**[插入图片: lecture_08.py, image call "images/tensor-parallelism.png", 描述: 数据在所有设备上是完整的，而模型的每一层（宽度/隐藏层维度）被切分到不同设备上。]**

**分片策略**: 数据在所有rank上是完整的，但模型的每一层（例如MLP的权重矩阵）都沿着隐藏层维度（宽度）被切分。每个rank只拥有模型参数的一部分。

**`tensor_parallelism_main` 函数核心逻辑 (仅前向传播)**:
1.  **参数分片**: 创建MLP参数时，每个权重矩阵的形状是 `[num_dim, local_num_dim]`，其中`local_num_dim = num_dim / world_size`。
2.  **逐层计算与通信**:
    *   输入`x`与本地的参数分片`params[i]`进行矩阵乘法，得到一个局部的激活值`x`，其形状为 `[batch_size, local_num_dim]`。
    *   此时，每个rank只拥有完整激活值的一部分。为了进行下一层的计算（需要完整的激活值），必须进行通信。
    *   调用`dist.all_gather`，将所有rank上的局部激活值收集起来，并在每个rank上拼接成完整的激活值张量，形状恢复为`[batch_size, num_dim]`。
    *   将拼接后的完整激活值作为下一层的输入，重复此过程。

```python
# 张量并行核心代码片段
def tensor_parallelism_main(rank: int, world_size: int, ...):
    # ...
    # 每个rank只拥有模型参数的一个切片
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]
    
    x = data
    for i in range(num_layers):
        # 1. 局部计算
        x = x @ params[i]
        x = F.gelu(x)  # x 的形状是 [batch_size, local_num_dim]
        
        # 2. 通信以获取完整激活
        activations = [...] # 准备接收数据的列表
        dist.all_gather(tensor_list=activations, tensor=x)
        
        # 3. 拼接
        x = torch.cat(activations, dim=1) # x 的形状恢复为 [batch_size, num_dim]
    # ...
```

**关键点**: 张量并行在每一层内部都需要进行`all_gather`通信，通信量巨大，因此对GPU间的互联带宽（如NVLink）要求非常高。

### **流水线并行 (Pipeline Parallelism)**

![流水线并行示意图](placeholder.png)
**[插入图片: lecture_08.py, image call "images/pipeline-parallelism.png", 描述: 数据在所有设备上是完整的，而模型的不同层（深度）被分配到不同设备上。]**

**分片策略**: 数据在所有rank上是完整的，但模型的层（沿着深度方向）被切分。例如，rank 0拥有模型的第1-2层，rank 1拥有第3-4层。

为了减少设备空闲（即“流水线气泡”），通常会将一个大的batch切分成多个更小的**微批次 (micro-batches)**。

**`pipeline_parallelism_main` 函数核心逻辑 (仅前向传播)**:
1.  **层分配**: 每个rank只创建和存储分配给自己的那部分层`local_params`。
2.  **微批次处理**:
    *   将输入数据`data`切分成多个`micro_batches`。
    *   对于每个微批次`x`：
        *   如果当前rank不是第一个（`rank > 0`），则通过`dist.recv`从上一个rank接收其计算好的激活值。
        *   在接收到的激活值上，执行分配给当前rank的层的计算。
        *   如果当前rank不是最后一个（`rank < world_size - 1`），则通过`dist.send`将计算结果发送给下一个rank。

```python
# 流水线并行核心代码片段
def pipeline_parallelism_main(rank: int, world_size: int, ...):
    # ...
    # 每个rank只拥有模型的一部分层
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # 将大batch切分为微批次
    micro_batches = ...

    for x in micro_batches:
        # 从上一个rank接收数据
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)
        
        # 在本地层上计算
        for param in local_params:
            x = x @ param
            x = F.gelu(x)
            
        # 发送给下一个rank
        if rank + 1 < world_size:
            dist.send(tensor=x, dst=rank + 1)
    # ...
```
**关键点**:
*   流水线并行使用点对点通信（`send`/`recv`）而非集体通信。
*   这个简单的实现没有处理通信和计算的重叠，会导致显著的流水线气泡。实际的框架（如GPipe, PipeDream）有更复杂的调度策略来最大化设备利用率。
*   反向传播的实现更为复杂，需要精心设计的交错前向和反向计算步骤。

**[深入探讨: 张量与流水线并行代码实现](./Lecture8-Tensor-And-Pipeline-Parallelism-Code.md)**

## 总结与展望

### 缺失的部分
我们手撕的实现是教学性质的，省略了许多生产级框架的复杂性：
*   **更通用的模型支持**: 未处理Attention等复杂结构。
*   **通信与计算重叠**: 我们的实现大多是同步的，而异步操作和精心调度是性能优化的关键。
*   **复杂的代码簿记**: 生产级框架如`Megatron-LM`或PyTorch的`FSDP`需要大量代码来管理参数、梯度和优化器状态的分片与收集。

### JAX/TPU生态系统
值得一提的是，Google的JAX/TPU生态系统提供了一种更高级、更具声明性的并行化方式。开发者只需定义模型和分片策略（例如，哪个维度的数据或参数应该被切分），JAX编译器会自动处理底层的通信原语和数据移动，大大简化了并行编程的复杂性。

### 核心思想回顾
*   **并行化的维度**: 我们可以沿多个维度切分计算任务：数据批次（数据并行）、模型宽度（张量并行）、模型深度（流水线并行），甚至序列长度。
*   **计算、存储与通信的权衡**: 这是一个永恒的主题。我们可以重新计算以节省内存（激活检查点），或者将数据存储在本地内存、甚至其他GPU的内存中，但这会带来不同程度的通信开销。
*   **硬件发展与模型规模**: 尽管硬件在飞速发展，但我们对更大模型的需求也在同步增长。这意味着模型规模将永远挑战硬件能力的极限，因此，理解并掌握并行计算的原理和技术将始终至关重要。