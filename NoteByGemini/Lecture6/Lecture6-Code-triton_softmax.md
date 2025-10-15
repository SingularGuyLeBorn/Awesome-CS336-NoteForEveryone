### 模板D: 代码实现深度解析

#### 1. 核心功能与目标 (Core Function & Goal)
`triton_softmax` 函数及其 Kernel `triton_softmax_kernel` 的目标是使用 **[Triton](./Lecture6-Triton.md)** 来实现一个高性能的、完全融合的 **[Softmax](./Lecture6-Softmax.md)** 算子. 这个实现旨在解决 `manual_softmax` 中因多次内存往返而导致的严重性能问题. 它通过一种“一行一块” (one row per block) 的巧妙设计, 将包含归约操作的复杂计算流程完全限制在 SM 的高速片上内存中执行, 是**[算子融合](./Lecture6-Kernel-Fusion.md)**在非逐元素操作上的典范应用. 

#### 2. 参数解析 (Parameters)
*   `triton_softmax` (Python Host Function):
    *   `x` (torch.Tensor): 输入的二维张量(矩阵). 

*   `triton_softmax_kernel` (Triton JIT Kernel):
    *   `x_ptr`, `y_ptr`: 指向输入和输出张量内存地址的指针. 
    *   `x_row_stride`, `y_row_stride`: 输入和输出张量在行方向上的步长(即移动到下一行需要跳过的元素数量). 这对于处理非连续的张量视图是必要的. 
    *   `num_cols` (int): 矩阵的列数. 
    *   `BLOCK_SIZE` (tl.constexpr): 一个编译时常量, 其值通常是大于等于 `num_cols` 的最小 2 的幂. 它定义了内部处理块的大小. 

#### 3. 核心逻辑 (Core Logic)
```python
import torch
import triton
import triton.language as tl

# 主机端代码 (Host Code)
def triton_softmax(x: torch.Tensor):
    y = torch.empty_like(x)
    M, N = x.shape # M: 行数, N: 列数
    
    # 关键设计:一个块处理一整行
    # 每个块处理的元素数, 向上取整到最近的 2 的幂, 以获得更好的性能
    block_size = triton.next_power_of_2(N)
    
    # 启动 M 个块, 每个块负责一行
    num_blocks = M
    
    triton_softmax_kernel[(M,)](
        x, y,
        x.stride(0), y.stride(0), # 传递行步长
        N, # 传递列数
        BLOCK_SIZE=block_size
    )
    return y

# Triton Kernel 代码 (Device Code)
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    # 1. 确定当前块负责的行
    # 每个程序实例(块)的 ID 对应于行索引
    row_idx = tl.program_id(0)
    
    # 2. 计算要加载的列的偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 计算当前行在内存中的起始地址
    x_start_ptr = x_ptr + row_idx * x_row_stride
    # 计算当前行所有列的内存地址
    x_ptrs = x_start_ptr + col_offsets
    
    # 3. 加载一整行数据
    # 使用掩码来处理列数不等于 BLOCK_SIZE 的情况, 边界外的值加载为负无穷
    mask = col_offsets < num_cols
    x_row = tl.load(x_ptrs, mask=mask, other=float("-inf"))
    
    # 4. 在片上 (on-chip) 完成所有计算
    # a. 数值稳定:减去最大值
    # tl.max 在整个向量 x_row 上进行归约
    row_max = tl.max(x_row, axis=0) 
    x_row = x_row - row_max
    
    # b. 计算指数
    numerator = tl.exp(x_row)
    
    # c. 计算归一化分母
    # tl.sum 在整个向量 numerator 上进行归约
    denominator = tl.sum(numerator, axis=0)
    
    # d. 归一化
    y_row = numerator / denominator
    
    # 5. 写回一整行结果
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=mask)
```

#### 4. 与理论的连接 (Connection to Theory)
*   **最大化算子融合**:此 Kernel 将 `max`(归约)、`sub`(逐元素)、`exp`(逐元素)、`sum`(归约)、`div`(逐元素)五个逻辑步骤**完全融合**. 数据从全局内存被加载一次到寄存器中, 所有中间结果(如 `row_max`, `numerator`, `denominator`)都在高速的片上内存中产生和消耗, 从未写回全局内存. 
*   **片上归约 (On-chip Reduction)**:Triton 的强大之处在于 `tl.max` 和 `tl.sum` 等归约操作可以直接作用于在寄存器中的数据块 `x_row`. 编译器会自动生成高效的并行归约代码(通常使用**[共享内存](./Lecture6-Shared-Memory.md)**或 warp-level primitives), 将归约操作的延迟隐藏在计算中. 
*   **提升[算术强度](./Lecture6-Arithmetic-Intensity.md)**:通过将多次全局内存遍历(在 `manual_softmax` 中约 8MN 的数据移动)压缩为一次读(MN)和一次写(MN), 该 Kernel 极大地提高了有效算术强度, 将一个典型的**[内存密集型](./Lecture6-Memory-vs-Compute-Bound.md)**任务转变得更接近**[计算密集型](./Lecture6-Memory-vs-Compute-Bound.md)**. 
*   **结构化并行**: "一行一块" 的设计是一种优雅的并行化策略. 它将问题分解为 `M` 个完全独立的子问题(处理每一行), 完美地匹配了 **[GPU 执行模型](./Lecture6-GPU-Execution-Model.md)** 中线程块之间相互独立的特性, 实现了简单而高效的扩展. 