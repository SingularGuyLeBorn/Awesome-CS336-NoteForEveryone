### 1. 核心功能与目标

本笔记将解析如何使用Triton实现一个高性能的**Softmax**核函数. 与逐元素的GeLU不同, Softmax是一个**行级 (row-wise)** 操作, 并且包含**归约 (reduction)** 步骤 (求最大值, 求和). 这使得它成为展示Triton处理更复杂并行模式的绝佳案例.

我们的目标是设计一个融合的Softmax核函数, 它通过一次遍历输入行的内存, 就在GPU的高速内存 (寄存器/共享内存) 中完成所有计算 (减去最大值, exponentiate, 求和, 归一化), 最后将结果一次性写回, 从而将多次全局内存往返压缩为一次读和一次写, 大幅提升性能.

### 2. 代码块与逐行注释

#### 2.1 设计思路: 一行一块 (One Row per Block)

处理Softmax最高效的并行策略是**将输入矩阵的每一行分配给一个独立的线程块 (Thread Block)**.
- **数据局部性**: 由于一个线程块被调度到单个SM上执行, 这意味着计算一行Softmax所需的所有数据都可以在该SM内部处理.
- **高效归约**: 求一行中的最大值和总和, 变成了一个块内所有线程的协同归约操作, 这可以利用SM的高速共享内存来完成, 避免了昂贵的全局内存同步.

#### 2.2 Python 封装函数 (Wrapper)

封装函数负责根据上述设计思路来配置和启动Triton核函数.

```python
import triton
import triton.language as tl

def triton_softmax(x: torch.Tensor):
    # 分配输出张量
    y = torch.empty_like(x)

    # 确定执行网格 (Grid)
    M, N = x.shape  # M: 行数, N: 列数
    
    # 1. 设置网格维度: 我们需要M个独立的程序实例, 每个处理一行
    # 所以网格是一维的, 大小为M
    num_blocks = M
    
    # 2. 设置块大小: 每个块需要能容纳一行的所有元素
    # Triton性能最佳时, 块大小通常是2的幂次
    # 我们找到不小于N的最小的2的幂次作为块大小
    block_size = triton.next_power_of_2(N)

    # 启动核函数
    triton_softmax_kernel[(num_blocks,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), # 传递行步长, 用于2D寻址
        y_row_stride=y.stride(0),
        num_cols=N,               # 传递实际的列数, 用于掩码
        BLOCK_SIZE=block_size     # 编译时常量, 定义块的大小
    )
    return y
```

#### 2.3 Triton 核函数 (Kernel)

这个核函数实现了单个线程块处理单行数据的全部逻辑.

```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    # 1. 获取当前处理的是哪一行
    # 每个程序实例(块)处理一行, 所以行索引就是程序ID
    row_idx = tl.program_id(0)

    # 2. 计算当前行数据的内存地址
    # 首先定位到该行的起始地址
    x_start_ptr = x_ptr + row_idx * x_row_stride
    # 然后创建列的偏移量向量 [0, 1, ..., BLOCK_SIZE-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # 得到这一行所有元素的完整指针向量
    x_ptrs = x_start_ptr + col_offsets
    
    # 3. 创建掩码, 防止读取超出实际列数的内存
    mask = col_offsets < num_cols

    # 4. 加载一整行数据到块内的高速内存(寄存器)
    # 对于掩码外的无效位置, 加载负无穷, 这样它们不会影响max的计算
    x_row = tl.load(x_ptrs, mask=mask, other=float("-inf"))

    # 5. 在块内执行Softmax的计算步骤 (全部是向量化操作)
    # 5.1 减去最大值 (归约操作): tl.max在块内高效计算向量的最大值
    row_max = tl.max(x_row, axis=0)
    x_row = x_row - row_max
    
    # 5.2 计算指数
    numerator = tl.exp(x_row)
    
    # 5.3 计算分母 (归约操作): tl.sum在块内高效计算向量的和
    denominator = tl.sum(numerator, axis=0)
    
    # 5.4 归一化
    y_row = numerator / denominator

    # 6. 将计算好的一整行结果写回全局内存
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=mask)
```

### 3. 张量流动分析

- **输入**: `x_ptr` 指向一个形状为 `(M, N)` 的二维张量.
- **核函数启动**: 网格维度为 `(M,)`, 意味着启动了 `M` 个并行的核函数实例.
- **单个核函数实例内部**:
    - `row_idx`: 一个标量, 如 `0, 1, ..., M-1`.
    - `col_offsets`: 形状为 `(BLOCK_SIZE,)` 的索引张量.
    - `x_row`: 从全局内存加载到寄存器的一行数据, 形状为 `(BLOCK_SIZE,)`.
    - `row_max`, `denominator`: 都是通过归约操作 (`tl.max`, `tl.sum`) 从 `x_row` 计算得到的**标量**.
    - `y_row`: 最终计算得到的输出行, 形状为 `(BLOCK_SIZE,)`.
- **输出**: `y_row` 被写回到输出张量 `y` 对应的行中.

### 4. 与理论的连接

- **算术强度与内存瓶颈**: 朴素的Softmax实现 (如用多个PyTorch操作) 是典型的**内存密集型 (Memory-Bound)** 操作. 它需要多次完整地读写整个张量 (一次为了找max, 一次为了减max, 一次为了exp, 一次为了sum, 一次为了除).
- **核函数融合的威力**: 这个Triton核函数是**核函数融合**思想的完美体现. 它将所有这些步骤融合到了一个单一的内核中. 数据从全局内存**只被读取一次**到SM的高速寄存器中, 所有中间计算 (`max`, `-`, `exp`, `sum`, `/`) 都在寄存器上完成, 最后结果**只被写回一次**.
- **并行归约**: `tl.max` 和 `tl.sum` 是Triton中实现**并行归约**的关键. Triton编译器会将这些高级API调用转换成高效的底层指令, 利用块内线程的协作和共享内存来实现快速的聚合计算, 这是实现高性能归约操作的标准模式.

通过这种方式, 我们将一个原本算术强度极低的操作, 通过最大化数据复用和最小化全局内存通信, 极大地提升了其有效算术强度和最终性能.

这个为Softmax量身定制的"一行一块"策略, 是在掌握了Triton基础之上的进阶应用. 关于Triton的块级编程范式, 向量化加载 (`tl.load`) 和掩码 (`mask`) 等基础概念, 读者可以先参考 [Triton GeLU笔记](./Lecture6-Code-Triton-GeLU) 以建立更扎实的基础.