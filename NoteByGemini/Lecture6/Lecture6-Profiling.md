### 模板C: 方法/流程

#### 1. 目标 (Objective)
**性能分析 (Profiling)** 是一种比**[性能评测 (Benchmarking)](./Lecture6-Benchmarking.md)**更精细的性能诊断技术。它的目标不是测量总时间，而是**分解总时间**，找出代码中具体是哪些函数或操作消耗了最多的时间，即定位**性能瓶颈**。通过性能分析，我们可以回答“我的代码为什么慢？”以及“时间都去哪儿了？”这类问题。

更深层次地，性能分析帮助我们理解程序的实际执行流程，揭示 PyTorch 高级 API 背后的底层 CUDA 算子调用，为我们提供了优化代码所需的关键洞察。

#### 2. 核心步骤与最佳实践 (Steps & Best Practices)
与评测类似，一个有效的性能分析流程也需要遵循特定的步骤，这些都在课程的 **[`profile` 函数](./Lecture6-Code-profile.md)** 中有所体现：

1.  **隔离分析目标**：将被分析的代码段封装在一个独立的函数中，以便 Profiler 能够清晰地捕获其执行过程。
2.  **预热 (Warm-up)**：同样需要预热，以避免将一次性的初始化开销计入分析结果，确保分析的是稳态性能。
3.  **启用 Profiler 上下文**：
    *   使用 `torch.profiler.profile` 上下文管理器来包裹要分析的代码。
    *   指定 `activities`，通常包括 `ProfilerActivity.CPU` 和 `ProfilerActivity.CUDA`，以同时捕获 CPU 和 GPU 上的活动。
    *   对于复杂的代码，开启 `with_stack=True` 可以记录调用栈信息，这对于生成火焰图等可视化结果至关重要。
4.  **执行与同步**：在 Profiler 上下文内运行代码，并在结束后调用 `torch.cuda.synchronize()` 确保所有 GPU 活动都被完整记录下来。
5.  **结果分析与可视化**：
    *   **表格视图**：使用 `prof.key_averages().table(...)` 可以生成一个按时间消耗排序的函数/算子列表。这是最直接的瓶颈定位方法。
    *   **时间轴/轨迹视图**：更高级的工具如 **[NVIDIA Nsight Systems](./Lecture6-NVIDIA-Nsight-Systems.md)** 或 PyTorch Profiler TensorBoard 插件可以将事件呈现在时间轴上，直观地展示 CPU 和 GPU 的并行与依赖关系。
    *   **火焰图 (Flame Graph)**：当开启调用栈记录时，可以生成火焰图，这是一种强大的可视化工具，用于分析函数调用的层次结构和时间占比。

#### 3. 使用的工具 (Tools)
*   **PyTorch Profiler (`torch.profiler`)**：
    *   PyTorch 内置的官方性能分析工具，易于使用，功能强大。
    *   可以直接在 Python 脚本中使用，输出清晰的表格报告。
    *   支持导出 Chrome Trace 文件，可以在 `chrome://tracing` 中进行可视化分析。
    *   可以与 TensorBoard 集成，提供更丰富的交互式可视化界面。

*   **NVIDIA Nsight Systems (`nsys`)**：
    *   NVIDIA 官方提供的专业级系统性能分析工具，功能最为强大。
    *   它能提供关于整个系统（CPU、GPU、驱动、库）活动的极其详尽的时间轴视图。
    *   特别适合分析复杂的 **[CPU-GPU 交互](./Lecture6-CPU-GPU-Synchronization.md)**、延迟问题和硬件利用率。
    *   在 PyTorch 代码中，可以通过 `torch.cuda.nvtx.range` 来添加自定义的标记，这些标记会在 Nsight Systems 的时间轴上显示，极大地提高了分析的可读性。课程中的 **[`lecture_06_mlp.py`](./Lecture6-Code-MLP.md)** 就使用了 NVTX 标记。