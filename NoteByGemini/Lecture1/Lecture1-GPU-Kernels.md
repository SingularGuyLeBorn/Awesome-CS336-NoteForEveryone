# 专题: GPU 核函数 (Kernels)
## 1. 核心定义
**GPU 核函数 (Kernel)** 是指一段在 GPU 上执行的、专门编写的并行计算程序. 当我们从 CPU(主机 Host)调用一个操作(如矩阵乘法)在 GPU(设备 Device)上执行时,CPU 实际上是启动(launch)了一个或多个核函数. 
GPU 拥有数千个微小的计算核心,能够以大规模并行的方式执行相同的指令. 核函数正是为这种并行架构设计的程序,它定义了**单个线程(thread)**需要执行的操作. GPU 会同时启动成千上万个线程,每个线程都执行相同的核函数代码,但处理的数据是不同的. 
## 2. 为何需要自定义核函数？
像 PyTorch、TensorFlow 这样的深度学习框架已经为我们提供了大量预编译好的、高度优化的核函数(如矩阵乘法 `torch.matmul`、卷积 `torch.nn.Conv2d`). 但在某些情况下,我们需要编写自己的核函数: 
1.  **性能优化 (Operator Fusion):**
    *   **问题:** 在 **[Transformer](./Lecture1-Transformer.md)** 等模型中,经常会有一系列连续的、逐元素(element-wise)的操作,例如 `y = scale * (x + bias)`. 如果按顺序调用 PyTorch 的操作,会发生多次核函数启动和多次全局内存(HBM)读写: 读 `x` -> 写 `x+bias` -> 读 `x+bias` -> 写 `y`. 
    *   **解决方案:** 编写一个**融合核函数 (Fused Kernel)**. 这个核函数一次性将 `x` 和 `bias` 从全局内存读入到 GPU 核心旁边的快速缓存(SRAM)中,在缓存中完成所有计算 `scale * (x + bias)`,然后将最终结果 `y` 一次性写回全局内存. 这大大减少了与缓慢的全局内存的交互次数,从而显著提升性能. **[FlashAttention](./Lecture1-FlashAttention.md)** 就是一个极致的融合核函数示例. 
2.  **实现新颖的操作:**
    *   当研究人员提出一种新的、现有框架不支持的算法或操作时(例如一种新的注意力机制或激活函数),就需要自己编写核函数来实现它. 
## 3. GPU 内存层次与性能瓶颈
理解 GPU 的内存层次对于编写高效的核函数至关重要. 可以将其类比为工厂和仓库: 
*   **全局内存 (Global Memory / HBM):** 位于 GPU 芯片之外,容量大(几十 GB),但访问速度慢. 相当于一个**巨大的远方仓库**. 
*   **L2 缓存 (L2 Cache):** 位于 GPU 芯片上,被所有计算单元共享. 容量中等(几十 MB),速度比全局内存快. 相当于**工厂的中央货运站**. 
*   **共享内存 (Shared Memory / SRAM):** 位于每个流式多处理器(Streaming Multiprocessor, SM)内部,只被该 SM 内的线程块(Thread Block)共享. 容量小(几十到一百多 KB),但速度极快,几乎和寄存器一样快. 相当于**车间旁边的小型工具架**. 
*   **寄存器 (Registers):** 位于每个计算核心内部,是最小、最快的存储单元. 相当于**工人手中的工具**. 
**性能瓶颈:** 大部分计算操作都是**内存带宽受限 (Memory-bound)** 而非**计算受限 (Compute-bound)**. 这意味着,性能的瓶颈往往在于**从全局内存中搬运数据到计算核心的速度**,而不是计算核心本身的处理速度. 
高效核函数设计的核心目标就是: **最大化计算/内存访问比率 (Compute-to-Memory-Access Ratio)**,即尽量减少与全局内存的交互,让数据在快速的片上缓存(SRAM)中被充分利用. 
## 4. Triton: 编写核函数的现代语言
传统上,编写 GPU 核函数需要使用 CUDA C++,它非常强大但学习曲线陡峭,需要手动管理复杂的内存分配和线程同步. 
**[Triton](./Lecture1-Triton.md)** 是由 OpenAI 开发的一种基于 Python 的领域特定语言(DSL),旨在简化高性能 GPU 核函数的编写. 
*   **优点:**
    *   **Pythonic 语法:** 开发者可以在 Python 环境中编写核函数,代码更简洁、易读. 
    *   **自动优化:** Triton 的编译器会自动处理许多底层的优化细节,如内存合并(memory coalescing)、指令调度和线程块分配,让开发者可以更专注于算法逻辑. 
    *   **性能媲美 CUDA:** 在许多情况下,Triton 生成的代码性能可以与专家手写的 CUDA C++ 代码相媲美. 
**示例(一个简单的加法核函数): **
```python
import torch
import triton
import triton.language as tl
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```
这个 Triton 核函数展示了如何以并行块(block)的方式加载数据、进行计算,并将结果写回,同时屏蔽了大量复杂的底层细节. 