### 模板A: 核心概念

#### 1. 这是什么？(What is it?)
**GPU 架构 (GPU Architecture)** 指的是图形处理单元 (Graphics Processing Unit) 的内部设计和组织方式。虽然最初为图形渲染而设计，但其大规模并行处理能力使其成为深度学习等计算密集型任务的理想硬件。现代 GPU 架构，如 NVIDIA 的 Ampere (A100) 或 Hopper (H100)，其核心是围绕着大规模并行计算和高效的内存访问来构建的。

#### 2. 为什么重要？(Why is it important?)
理解 GPU 架构是编写高性能代码的基础。不了解硬件的工作方式，就如同在不了解引擎的情况下试图调校一辆赛车。具体来说，它能帮助我们：
*   **解释性能瓶颈**：理解为什么某些操作是**[内存密集型](./Lecture6-Memory-vs-Compute-Bound.md)**而另一些是**[计算密集型](./Lecture6-Memory-vs-Compute-Bound.md)**。
*   **指导优化策略**：架构的特性直接决定了**[算子融合](./Lecture6-Kernel-Fusion.md)**、**[分块 (Tiling)](./Lecture6-Matrix-Multiplication-Tiling.md)** 等优化技术为何有效。
*   **合理利用资源**：知道如何有效利用不同层级的内存（寄存器、**[共享内存](./Lecture6-Shared-Memory.md)**、缓存、DRAM）是性能调优的关键。

#### 3. 它是如何工作的？(How does it work?)
一个现代 GPU 的核心组件包括：

*   **流式多处理器 (Streaming Multiprocessor, SM)**：
    *   这是 GPU 的“大脑”和基本的计算调度单位。一个 GPU 包含数十到上百个 SM（例如 A100 有 108 个）。
    *   所有的**[线程块 (Thread Blocks)](./Lecture6-GPU-Execution-Model.md)** 都会被调度到某个特定的 SM 上执行。
    *   每个 SM 内部包含大量的计算核心（如 CUDA Cores for FP32, Tensor Cores for mixed-precision matrix math）、调度器、以及高速的本地内存。

*   **内存层次结构 (Memory Hierarchy)**：
    *   **寄存器 (Registers)**：位于 SM 内部，是每个线程私有的、最快的内存。用于存储临时变量。Triton 编译器会大量使用寄存器来减少对更慢内存的访问。
    *   **L1 缓存 / 共享内存 (L1 Cache / Shared Memory)**：同样位于 SM 内部，速度极快（接近寄存器）。L1 缓存对线程是透明的，而**[共享内存](./Lecture6-Shared-Memory.md)**则可以由同一线程块内的所有线程显式地编程访问。这是实现线程间高效协作的关键。
    *   **L2 缓存 (L2 Cache)**：被所有 SM 共享，容量比 L1 大，但速度稍慢。作为 DRAM 的一个中间层，用于减少对主内存的访问延迟。
    *   **高带宽内存 (High Bandwidth Memory, HBM) / DRAM**：也称为全局内存 (Global Memory)，是 GPU 的主内存。容量巨大（如 A100 有 40GB 或 80GB），但访问延迟最高、速度最慢。几乎所有的性能优化都是为了尽可能减少对 DRAM 的读写次数。

#### 4. 关键要点 (Key Takeaways)
*   GPU 的本质是**大规模并行**，其架构围绕着成百上千个计算核心（分布在 SM 中）并行工作而设计。
*   内存访问是性能的主要瓶颈。**数据局部性**至关重要——让数据尽可能地靠近计算单元（即保留在寄存器和共享内存中）。
*   **SM 是调度的核心**。我们的编程模型（特别是线程块的设计）应该以 SM 的资源（如共享内存大小、最大线程数）为中心进行考量。