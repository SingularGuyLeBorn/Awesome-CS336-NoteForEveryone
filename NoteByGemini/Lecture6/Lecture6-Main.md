# 第 6 讲：手写高性能算子 (Kernel)

### 前言

欢迎来到第六讲。在本讲中，我们将深入底层，学习如何为 GPU 编写高性能代码。这不仅是 Assignment 2 的核心部分——你需要进行大量的性能分析，并为 Flash Attention 2 编写自己的 Triton 算子——更是优化语言模型中标准组件性能的关键技能。我们将从 GPU 的基础知识回顾开始，逐步深入**[性能评测](./Lecture6-Benchmarking.md)**与**[性能分析](./Lecture6-Profiling.md)**，并最终亲手用 **[CUDA](./Lecture6-CUDA.md)** 和 **[Triton](./Lecture6-Triton.md)** 编写我们自己的算子 (Kernel)。我们甚至会探索 PyTorch 的 JIT 编译器 **[`torch.compile`](./Lecture6-torch.compile.md)**，看看它如何为我们自动优化代码。整个过程，我们会一直深挖到 **[PTX](./Lecture6-PTX.md)**，即接近机器码的层面，去真正理解 GPU 在底层究竟在做什么。

### 1. GPU 基础回顾

让我们快速回顾一下 GPU 的工作原理。像 A100 或 H100 这样的现代 GPU，其核心是大量的**[流式多处理器 (Streaming Multiprocessors, SM)](./Lecture6-GPU-Architecture.md)**。每个 SM 内部都包含海量的计算单元，用于执行 FP32 等类型的计算。

#### 1.1. GPU 架构与内存层次

GPU 的**[架构](./Lecture6-GPU-Architecture.md)**包含一个清晰的内存层次结构：
*   **DRAM (全局内存)**：容量巨大，但速度较慢。
*   **缓存 (Caches)**：如 L1 和 L2 缓存，速度快得多。
*   **寄存器文件 (Register File)**：这是每个线程可以访问的超高速内存，也是我们编写高性能代码时需要重点利用的资源。

#### 1.2. 执行模型

GPU 的**[执行模型](./Lecture6-GPU-Execution-Model.md)**是分层的：
*   **线程 (Thread)**：执行计算的基本单位。例如，对一个向量进行操作时，每个线程可能负责处理向量中的一个或几个元素。
*   **线程块 (Thread Block)**：一组线程的集合，会被调度到**单个 SM**上执行。这是我们思考和编程（尤其是在 Triton 中）的原子单位。线程块内的线程可以通过**[共享内存](./Lecture6-Shared-Memory.md)**高速通信和同步，这对于矩阵乘法等需要数据交换的复杂操作至关重要。跨线程块的通信则非常昂贵。
*   **网格 (Grid)**：一次算子调用所启动的所有线程块的集合。
*   **线程束 (Warp)**：SM 中执行的最小调度单位，通常是 32 个线程的集合。Warp 的存在减少了控制逻辑的开销，使得 GPU 能以极高的并行度执行计算，这也是 GPU 相较于 CPU 的核心优势之一。

#### 1.3. 算术强度

在性能优化中，**[算术强度](./Lecture6-Arithmetic-Intensity.md)**是一个至关重要的概念。它定义为浮点运算次数 (FLOPs) 与内存访问字节数 (Bytes) 的比值。我们的目标是尽可能提高算术强度，因为现代 GPU 的计算能力增长速度远超内存带宽的增长速度。
*   **[计算密集型 (Compute-Bound)](./Lecture6-Memory-vs-Compute-Bound.md)**：算术强度高，性能受限于计算单元的速度。这是我们追求的理想状态。
*   **[内存密集型 (Memory-Bound)](./Lecture6-Memory-vs-Compute-Bound.md)**：算术强度低，性能受限于内存带宽。大多数操作（除了优化良好的矩阵乘法）都属于此类。本讲的核心任务之一就是通过各种技巧（如**[算子融合](./Lecture6-Kernel-Fusion.md)**）来缓解内存瓶颈。

### 2. 性能评测与分析

如果你想编写高性能代码，最重要的一条原则是：**永远要评测和分析你的代码**。理论分析和硬件规格表固然重要，但代码的实际性能受到库版本、具体硬件和工作负载等多种因素的影响。因此，没有比直接测量更能反映真实情况的方法了。

我们将以一个简单的 **[MLP 模型](./Lecture6-Code-MLP.md)** 作为贯穿本节的示例，对其进行前向和反向传播的性能评估。

#### 2.1. 性能评测 (Benchmarking)

**[性能评测](./Lecture6-Benchmarking.md)**指的是测量一段代码端到端的执行时间（墙上时钟时间）。虽然它不能告诉你时间具体花费在哪里，但对于比较不同实现的优劣和理解性能如何随输入规模扩展非常有帮助。

我们定义了一个辅助函数 **[`benchmark`](./Lecture6-Code-benchmark.md)** 来完成这项任务。在进行评测时，必须注意两个关键点：
1.  **预热 (Warm-up)**：首次运行代码时，会涉及到底层编译、数据传输初始化等一次性开销。为了测量稳态性能，必须进行几次预热迭代。
2.  **[CPU-GPU 同步](./Lecture6-CPU-GPU-Synchronization.md)**：CPU 和 GPU 是异步执行的。CPU 提交一个任务后会继续执行，而不会等待 GPU 完成。如果在测量时间时不进行同步，你可能只测量到了 CPU 提交任务的时间，而不是 GPU 实际执行的时间。因此，在计时开始和结束时调用 `torch.cuda.synchronize()` 至关重要。

通过对矩阵乘法和我们的 MLP 模型进行基准测试，我们可以观察到，执行时间与矩阵尺寸、层数、步数等因素之间存在着预期的（超线性或线性）关系。

#### 2.2. 性能分析 (Profiling)

与评测不同，**[性能分析](./Lecture6-Profiling.md)**旨在深入代码内部，找出时间的具体分布。这不仅能帮你定位瓶颈，更能让你理解 PyTorch 底层调用了哪些 CUDA 算子，从而建立对程序执行过程的深刻直觉。

PyTorch 内置了一个强大的 Profiler。通过我们的辅助函数 **[`profile`](./Lecture6-Code-profile.md)**，我们可以看到简单的加法或矩阵乘法操作在底层是如何被分派到具体的 CUDA 算子的。例如，对于不同尺寸的矩阵乘法，PyTorch 会智能地调用不同的底层实现（如 NVIDIA 的 `cuBLAS` 或 `CUTLASS` 库中的特定算子）。

对于更复杂的模型，如我们的 MLP，PyTorch Profiler 的表格输出可能不够直观。这时，我们需要借助更专业的工具，如 **[NVIDIA Nsight Systems](./Lecture6-NVIDIA-Nsight-Systems.md)**。Nsight Systems 能够以时间轴的形式，极其清晰地展示 CPU 和 GPU 的活动，包括：
*   **CPU-GPU 交互**：你可以清楚地看到 CPU 线程如何提前运行，将 CUDA 算子推送到一个队列中，而 GPU 则在后面异步地执行这些任务。
*   **同步点的影响**：如果在代码中加入一个 `print(loss)` 语句，你会发现它会在时间轴上引入一个同步点。CPU 必须等待 GPU 完成损失计算才能打印，这会打断 CPU 的提前执行流，在某些极端情况下甚至可能造成 CPU 瓶颈。
*   **算子粒度分析**：你可以放大到任意时间段，查看每个 CUDA 算子的执行时长、启动来源等详细信息。

通过这些工具，我们能真正理解 Python 作为一种高性能语言的胶水层角色：正是因为 CPU-GPU 的异步解耦，CPU 才不会成为瓶颈，使得我们能用高级语言充分利用 GPU 的计算能力。

### 3. 核心优化技术：算子融合 (Kernel Fusion)

**[算子融合](./Lecture6-Kernel-Fusion.md)**是 GPU 编程中最核心的优化思想之一。我们可以用一个生动的比喻来理解它：
*   **DRAM (全局内存)** 就像一个**仓库**。
*   **SM 内部的高速内存 (SRAM/共享内存)** 就像一个**工厂**。

每次执行一个独立的算子（如一次乘法、一次加法），都相当于把数据从“仓库”运到“工厂”，加工后再运回“仓库”。这个运输过程（内存读写）非常耗时。如果我们把多个连续的操作“融合”成一个算子，数据就可以在“工厂”内部流转，只需一次入库和一次出库，从而大大减少了与慢速 DRAM 的交互，显著提升性能。

#### GeLU 案例研究

我们将以 **[GeLU](./Lecture6-GeLU.md)** 激活函数为例，展示算子融合的威力。
1.  **PyTorch 内置实现**：`torch.nn.functional.gelu`。这是一个高度优化的、融合后的算子。
2.  **[手动实现](./Lecture6-Code-manual_gelu.md)**：我们用 PyTorch 的基本操作（乘法、加法、`tanh` 等）手动复现 GeLU 的计算公式。

通过**[性能评测](./Lecture6-Benchmarking.md)**，我们发现，融合后的 PyTorch 版本比手动实现的版本快了将近 8 倍！**[性能分析](./Lecture6-Profiling.md)**结果也证实了我们的猜想：手动版本调用了多个独立的 CUDA 算子，而 PyTorch 版本只调用了一个融合后的 `gelu` 算子。

我们的目标，就是通过手写算子，来接近甚至超越 PyTorch 内置实现的性能。

### 4. 手写高性能算子

现在，让我们亲手打开 GPU 编程的黑盒，用不同的语言和工具编写自己的高性能算子。

#### 4.1. 使用 CUDA C++

**[CUDA](./Lecture6-CUDA.md)** 是 NVIDIA 提供的、基于 C++ 的 GPU 编程平台。它允许我们编写在 GPU 上成千上万个线程上并行执行的 `kernel` 函数。

在 **[`create_cuda_gelu`](./Lecture6-Code-create_cuda_gelu.md)** 的实现中，我们定义了一个 `gelu_kernel`。其核心逻辑是：
*   **计算全局索引**：每个线程通过其 `blockIdx`（线程块索引）和 `threadIdx`（线程内索引）计算出它在整个输入张量中应该负责处理的全局索引 `i`。
*   **边界检查**：确保索引 `i` 没有超出张量的范围。
*   **执行计算**：从输入指针读取 `in[i]`，执行 GeLU 计算，并将结果写入输出指针 `out[i]`。

通过 PyTorch 的 `load_inline` 工具，我们可以方便地在 Python 中编译和调用这段 CUDA 代码。评测结果显示，我们的 CUDA 实现比手动 PyTorch 版本快得多（1.8ms vs 8.1ms），已经非常接近官方的融合算子性能（1.1ms）。这证明了我们成功地实现了**[算子融合](./Lecture6-Kernel-Fusion.md)**。

#### 4.2. 使用 Triton

虽然 CUDA 功能强大，但其 C++ 语法和手动的内存管理较为繁琐。**[Triton](./Lecture6-Triton.md)** 是 OpenAI 开发的一种领域特定语言，它允许我们用类似 Python 的语法来编写高性能 GPU 算子，同时自动处理许多底层的优化细节（如内存合并、共享内存管理等）。

Triton 的编程模型从**线程级别**提升到了**线程块级别**。在 **[`triton_gelu`](./Lecture6-Code-triton_gelu.md)** 的实现中，我们定义的 `triton_gelu_kernel` 的工作方式有所不同：
*   **块级编程**：每个程序实例 (program instance) 负责处理一个数据块 (block)。
*   **向量化操作**：我们通过 `tl.arange` 创建一个偏移量向量 `offsets`，然后使用 `tl.load` 和 `tl.store` 一次性读写整个数据块。所有的计算（如 GeLU 公式）也都作用于这些数据块向量。
*   **掩码 (Masking)**：使用掩码来处理边界情况，确保不会读写到张量范围之外的内存。

Triton 代码不仅更易读、易写、易调试，而且其编译器生成的 **[PTX](./Lecture6-PTX.md)** (GPU 的汇编语言) 代码通常是高度优化的。评测结果显示，我们的 Triton 实现性能与手写的 CUDA C++ 版本相当，但开发体验却好得多。

#### 4.3. 使用 `torch.compile`

最后一种方法，也是最简单的方法，就是利用 PyTorch 2.0 引入的 **[`torch.compile`](./Lecture6-torch.compile.md)**。这是一个 JIT (Just-In-Time) 编译器，它可以接收普通的 Python 函数（比如我们之前写的**[手动 GeLU 实现](./Lecture6-Code-manual_gelu.md)**），并自动将其编译成优化的、融合后的算子。

评测结果令人印象深刻：`torch.compile` 生成的代码性能甚至比我们手写的 CUDA/Triton 版本还要好一些。深入分析发现，`torch.compile` 在底层也是将 Python 代码编译成了 Triton 算子，但它的编译器可能进行了比我们手动编写更精细的优化。

**结论**：现代 JIT 编译器非常强大。对于大多数常见的**[算子融合](./Lecture6-Kernel-Fusion.md)**场景，`torch.compile` 是首选。只有在处理像 Flash Attention 这样具有非常规内存访问模式的复杂新架构时，才需要我们亲自动手编写 Triton 算子。

### 5. 进阶：实现带归约操作的 Softmax

前面的 GeLU 是一个逐元素 (element-wise) 操作，相对简单。现在我们来挑战一个更复杂的操作：**[Softmax](./Lecture6-Softmax.md)**。Softmax 需要对矩阵的每一行进行归约操作（求最大值、求和），这涉及到了行内的数据依赖。

我们的 **[`triton_softmax`](./Lecture6-Code-triton_softmax.md)** 实现采用了一种简洁而高效的策略：
*   **一行一块 (One row per block)**：我们将网格 (grid) 设置为矩阵的行数，让每个 Triton 程序实例（即每个线程块）独立负责处理一整行数据。
*   **片上计算 (On-chip computation)**：
    1.  将一整行数据通过 `tl.load` 加载到 SM 的高速缓存（寄存器）中。
    2.  在片上完成所有计算：`tl.max` 求最大值，减去最大值以保证数值稳定性，`tl.exp` 计算指数，`tl.sum` 求和。
    3.  计算最终的归一化结果。
    4.  通过 `tl.store` 将结果一次性写回全局内存。

这种方法将多次与慢速 DRAM 的读写操作（在 **[手动 Softmax 实现](./Lecture6-Code-manual_softmax.md)** 中清晰可见）融合成了一次读和一次写，完美体现了**[算子融合](./Lecture6-Kernel-Fusion.md)**的思想，性能也远超非融合版本，与 PyTorch 官方实现和 `torch.compile` 版本相媲美。

### 总结与拓展阅读

本讲我们深入探讨了 GPU 性能优化的世界。核心要点包括：

*   **性能鸿沟**：编程模型（PyTorch, Triton, PTX）与底层硬件之间存在差距，这是性能问题的根源。
*   **工具为王**：通过**[性能评测](./Lecture6-Benchmarking.md)**理解扩展性，通过**[性能分析](./Lecture6-Profiling.md)**（特别是 **[Nsight Systems](./Lecture6-NVIDIA-Nsight-Systems.md)**）洞察内部机制。
*   **五种实现方式**：我们看到了实现一个函数的五种方式——手动 PyTorch、官方 PyTorch、`torch.compile`、CUDA 和 Triton，并比较了它们的性能。
*   **核心原则**：组织计算以最小化内存读写。
*   **关键思想**：通过**[算子融合](./Lecture6-Kernel-Fusion.md)**减少内存往返；对于更复杂的操作（如矩阵乘法），则需要利用**[共享内存](./Lecture6-Shared-Memory.md)**进行**[分块 (Tiling)](./Lecture6-Matrix-Multiplication-Tiling.md)**。

随着自动编译器（Triton, `torch.compile`）的不断进步，手动编写底层算子的需求可能会减少，但理解这些底层原理将永远是构建高效深度学习系统的基石。

#### 拓展阅读

为了最大限度地吸收本讲内容，我们建议遵循以下学习路径：

1.  **理论先行：** 首先阅读 **[GPU 架构](./Lecture6-GPU-Architecture.md)** 和 **[GPU 执行模型](./Lecture6-GPU-Execution-Model.md)** 笔记，建立对底层硬件和编程模型的基本认识。
2.  **性能基础：** 接着学习 **[性能评测](./Lecture6-Benchmarking.md)** 和 **[性能分析](./Lecture6-Profiling.md)**，并对照 **[benchmark 函数](./Lecture6-Code-benchmark.md)** 和 **[profile 函数](./Lecture6-Code-profile.md)** 的代码实现，掌握衡量与诊断代码性能的关键工具。
3.  **核心优化思想：** 理解 **[算子融合](./Lecture6-Kernel-Fusion.md)** 的重要性。这是本讲所有优化技巧的核心驱动力。
4.  **实践对比 (GeLU)：**
    *   从 **[手动实现 GeLU](./Lecture6-Code-manual_gelu.md)** 开始，理解其性能瓶颈。
    *   深入 **[CUDA C++ 实现 GeLU](./Lecture6-Code-create_cuda_gelu.md)**，体会底层编程的精细控制。
    *   学习 **[Triton 实现 GeLU](./Lecture6-Code-triton_gelu.md)**，感受 Pythonic GPU 编程的便利与高效。
    *   最后，了解 **[`torch.compile`](./Lecture6-torch.compile.md)** 如何自动化这一过程。
5.  **进阶实践 (Softmax):** 将所学应用于更复杂的归约操作，学习 **[Triton 实现 Softmax](./Lecture6-Code-triton_softmax.md)**。