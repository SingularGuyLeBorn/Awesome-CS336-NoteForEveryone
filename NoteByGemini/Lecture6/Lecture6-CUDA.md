### 模板B: 特定术语/技术

#### 1. 定义 (Definition)
**CUDA (Compute Unified Device Architecture)** 是由 NVIDIA 创建的一个并行计算平台和编程模型. 它允许软件开发者和科学家使用 NVIDIA GPU 进行通用目的处理(即 GPGPU, General-Purpose computing on Graphics Processing Units). CUDA 平台可以通过其软件开发工具包 (SDK) 进行访问, 该工具包的核心是基于 C/C++ 语言的扩展, 并提供了对 Fortran、Python 等多种语言的接口支持. 

从本质上讲, CUDA 让我们能够编写一种特殊的函数, 称为 **Kernel**, 这段代码可以在 GPU 上的成千上万个线程中并行执行. 

#### 2. 关键特性与用途 (Key Features & Usage)
*   **并行编程模型**:CUDA 提供了强大的**[GPU 执行模型](./Lecture6-GPU-Execution-Model.md)**, 通过 Grid-Block-Thread 的层次结构来组织并行任务, 使得开发者能够系统地管理大规模并行. 
*   **底层硬件访问**:它提供了对 **[GPU 架构](./Lecture6-GPU-Architecture.md)** 特性的细粒度控制, 包括对**[共享内存](./Lecture6-Shared-Memory.md)**的显式管理、线程同步等. 这使得极致的性能优化成为可能. 
*   **丰富的生态系统**:NVIDIA 围绕 CUDA 构建了庞大的生态系统, 包括 cuDNN(用于深度神经网络)、cuBLAS(用于基本线性代数)、cuFFT(用于快速傅里叶变换)等高度优化的库, 这些库是 PyTorch 等深度学习框架的性能基石. 
*   **语言扩展**:
    *   **`__global__`**:函数限定符, 用于声明一个函数是 Kernel, 即可以从 CPU 调用并在 GPU 上执行. 
    *   **`__device__`** / **`__host__`**:函数限定符, 分别表示函数在 GPU 上调用/执行, 或在 CPU 上调用/执行. 
    *   **Kernel 启动语法**:`kernel_name<<<num_blocks, threads_per_block>>>(args...);` 这种特殊的语法用于从 CPU 启动 GPU Kernel, 并指定所需的线程块和线程数量. 
    *   **内置变量**:如 `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, 这些变量让 Kernel 内的每个线程能够知道自己的身份和位置, 从而处理不同的数据. 

#### 3. 案例分析 (Case Study in this Lecture)
在本讲座中, 我们通过 **[`create_cuda_gelu`](./Lecture6-Code-create_cuda_gelu.md)** 函数演示了如何使用 CUDA C++ 编写一个高性能的 GeLU Kernel. 

*   **实现细节**:
    *   `gelu_kernel` 函数被声明为 `__global__`. 
    *   在 Kernel 内部, 通过 `int i = blockIdx.x * blockDim.x + threadIdx.x;` 计算出每个线程负责的全局索引. 
    *   通过 `if (i < num_elements)` 进行边界检查, 防止内存越界访问. 
    *   在边界内, 执行 **[GeLU](./Lecture6-GeLU.md)** 的完整计算, 并将结果写入输出数组. 
*   **与 PyTorch 集成**:我们使用 `torch.utils.cpp_extension.load_inline` 工具, 在 Python 脚本中动态编译 CUDA 源码, 并将其绑定为一个可调用的 Python 函数. 这极大地简化了在 PyTorch 中使用自定义 CUDA Kernel 的流程. 
*   **性能表现**:这个手写的 CUDA Kernel 成功地将多个操作**[融合](./Lecture6-Kernel-Fusion.md)**在一起, 性能远超朴素的 PyTorch 实现, 证明了直接使用 CUDA 进行底层编程在追求极致性能时的价值. 然而, 与 **[Triton](./Lecture6-Triton.md)** 相比, 其开发复杂度和代码量都更高. 