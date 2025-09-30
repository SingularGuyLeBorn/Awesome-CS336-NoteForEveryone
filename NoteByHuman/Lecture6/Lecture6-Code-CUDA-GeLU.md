### 1. 核心功能与目标

本笔记将深入解析如何使用底层的**CUDA C++** API来手写一个**GeLU**激活函数的融合核函数. 目标是绕过PyTorch的抽象层, 直接通过C++代码定义一个在GPU上并行执行的函数, 以实现将多个数学运算融合在一起, 从而最小化全局内存访问, 提升性能.

这代表了最经典, 最底层的GPU编程范式. 理解它有助于我们明白Triton等现代工具在底层为我们自动化了哪些工作.

### 2. 代码块与逐行注释

完整的实现包含两个部分: 一个是在CPU端调用的**C++封装函数 (wrapper)**, 另一个是真正在GPU上成千上万个线程中执行的**CUDA核函数 (kernel)**.

#### 2.1 C++ 封装函数 (Host Code)

这段代码运行在CPU上, 它的职责是:
1.  接收PyTorch张量作为输入.
2.  进行必要的检查 (如张量是否在GPU上, 内存是否连续).
3.  设置GPU执行配置 (计算需要多少个线程块).
4.  启动 (launch) CUDA核函数.

```cpp
// gelu.cu 文件中的一部分
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

// 前置声明CUDA核函数
void gelu_kernel(const float* in, float* out, int num_elements);

// 这是我们从Python中调用的C++函数
torch::Tensor gelu(torch::Tensor x) {
    // 1. 检查输入张量是否是CUDA张量
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    // 2. 检查内存是否连续, 这对指针计算至关重要
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    // 3. 创建一个与输入形状和类型相同的输出张量
    // 使用empty_like避免了不必要的内存初始化开销
    auto y = torch::empty_like(x);

    // 4. 计算执行配置 (Execution Configuration)
    const int num_elements = x.numel();
    const int block_size = 1024; // 每个线程块包含1024个线程 (一个常用值)
    // 向上取整, 确保所有元素都被覆盖
    const int num_blocks = (num_elements + block_size - 1) / block_size;

    // 5. 启动核函数!
    // <<<num_blocks, block_size>>> 是CUDA的特殊语法, 用于定义执行网格
    gelu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), // 指向输入数据的底层指针
        y.data_ptr<float>(), // 指向输出数据的底层指针
        num_elements
    );

    // 检查CUDA调用是否出错
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return y;
}
```

#### 2.2 CUDA 核函数 (Device Code)

这段代码以`__global__`关键字标记, 意味着它将被编译为在GPU设备上运行的代码. 当我们启动它时, `num_blocks * block_size` 个线程会同时执行这份代码. 每个线程必须根据自己独特的ID来计算它应该处理哪个数据.

```cpp
// gelu.cu 文件中的另一部分
#include <cmath> // for tanh

// __global__ 关键字表示这是一个CUDA核函数
__global__ void gelu_kernel(const float* in, float* out, int num_elements) {
    // 1. 计算当前线程的全局唯一索引 (Global Index)
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. 边界检查 (Boundary Check)
    if (i < num_elements) {
        // 3. 执行融合计算
        float x = in[i];
        const float c1 = 0.79788456;
        const float c2 = 0.044715;
        float x_cubed = x * x * x;
        float inner = c1 * (x + c2 * x_cubed);
        float tanh_val = tanhf(inner); // 使用CUDA提供的快速tanhf函数
        out[i] = 0.5 * x * (1.0 + tanh_val);
    }
}
```

#### 2.3 可视化理解: 全局线程索引

对于初学者而言, 公式 `int i = blockIdx.x * blockDim.x + threadIdx.x;` 可能是理解CUDA编程的第一个难点. 下图可以帮助我们直观地理解这个计算过程:

![CUDA线程索引计算示意图](https://developer.download.nvidia.com/assets/cuda/files/reduction.png)
> 图 1: 该图展示了一个一维的线程网格. `blockDim.x` 是每个块的宽度 (线程数). `blockIdx.x` 是块的ID. 一个线程的全局索引 `i` 是其所在块的起始索引 (`blockIdx.x * blockDim.x`) 加上它在块内的偏移量 (`threadIdx.x`).

### 3. 张量流动分析

对于逐元素操作, 张量形状保持不变.
- **输入**: `x` (PyTorch Tensor), 形状为 `(N)`. 在C++层, 我们通过 `x.data_ptr<float>()` 获得一个指向其连续内存块头部的`float*`指针.
- **输出**: `y` (PyTorch Tensor), 形状为 `(N)`. 我们同样传递一个指向其内存的`float*`指针给核函数.
- **核函数内部**: 每个线程处理一个索引 `i`, 执行 `out[i] = f(in[i])` 的计算.

### 4. 与理论的连接

这个实现是**GPU并行执行模型**最直接的体现.
- **网格 (Grid)**, **线程块 (Block)**, **线程 (Thread)** 的层级结构通过`<<<num_blocks, block_size>>>`语法和`blockIdx`, `blockDim`, `threadIdx`内置变量得以实现.
- **SIMD (单指令多数据)**: 成千上万个线程同时执行`gelu_kernel`这份相同的代码, 但由于每个线程计算出的索引`i`不同, 它们操作的是全局内存中的不同数据.
- **核函数融合**: GeLU的所有数学运算 (`*`, `+`, `tanhf`) 都被包含在`if (i < num_elements)`这个单次读写的代码块内. 这种在GPU寄存器中完成所有中间计算的方式, 是性能提升的关键.

这种手动的、线程级的底层控制虽然强大, 但也十分繁琐. 它的复杂性也凸显了[Triton版本](./Lecture6-Code-Triton-GeLU)中块级编程范式的简洁与高效.

### 5. PTX汇编剖析 (选读)

当NVCC编译器编译我们的CUDA C++核函数时, 它会生成一种名为PTX (Parallel Thread Execution) 的中间汇编代码. 分析它可以让我们一窥GPU的底层操作.

```ptx
// PTX for gelu_kernel (simplified)
.visible .entry _Z11gelu_kernelPKfPfi(
    .param .u64 _Z11gelu_kernelPKfPfi_param_0, // in*
    .param .u64 _Z11gelu_kernelPKfPfi_param_1, // out*
    .param .u32 _Z11gelu_kernelPKfPfi_param_2  // num_elements
)
{
    // --- 寄存器声明 ---
    .reg .b32 %r<9>; .reg .b64 %rd<4>; .reg .f32 %f<15>;

    // --- 入口与参数加载 ---
    ld.param.u64    %rd1, [_Z11gelu_kernelPKfPfi_param_0]; // 加载in指针到%rd1
    ld.param.u64    %rd2, [_Z11gelu_kernelPKfPfi_param_1]; // 加载out指针到%rd2
    ld.param.u32    %r1,  [_Z11gelu_kernelPKfPfi_param_2]; // 加载num_elements到%r1

    // --- 索引计算: i = blockIdx.x * blockDim.x + threadIdx.x ---
    mov.u32         %r2, %ctaid.x;      // %r2 = blockIdx.x
    mov.u32         %r3, %ntid.x;       // %r3 = blockDim.x
    mov.u32         %r4, %tid.x;        // %r4 = threadIdx.x
    mad.lo.s32      %r5, %r2, %r3, %r4; // %r5 = %r2 * %r3 + %r4 (fused multiply-add)

    // --- 边界检查: if (i < num_elements) ---
    setp.ge.s32     %p1, %r5, %r1;      // if %r5 >= %r1, set predicate %p1 to true
    @%p1 bra        LBB0_2;             // if %p1 is true, branch to end

    // --- 核心计算 ---
    // 加载数据: float x = in[i];
    cvta.to.global.u64 %rd3, %rd1;       // 指针转换
    mul.wide.s32    %rd4, %r5, 4;       // 计算字节偏移量 (i * 4 bytes)
    add.s64         %rd5, %rd3, %rd4;   // 计算最终地址 in + i
    ld.global.f32   %f1, [%rd5];        // 从全局内存加载x到浮点寄存器%f1

    // GeLU 数学运算 (会编译成一系列 fma 指令)
    // ... e.g., fma.rn.f32 %f2, %f1, %f1, 0; // x*x
    // ... (many math instructions)
    // call           __nv_tanhf;

    // --- 数据写回: out[i] = ... ---
    cvta.to.global.u64 %rd6, %rd2;       // 指针转换
    mul.wide.s32    %rd7, %r5, 4;       // 字节偏移
    add.s64         %rd8, %rd6, %rd7;   // 计算最终地址 out + i
    st.global.f32   [%rd8], %f14;       // 将最终结果%f14存回全局内存

LBB0_2:
    ret;
}
```

- **索引计算**: CUDA C++中的 `blockIdx.x * blockDim.x + threadIdx.x` 被高效地编译成了一条 `mad.lo.s32` (低32位乘加) 指令.
- **内存操作**: 我们可以清晰地看到 `ld.global.f32` (从全局内存加载) 和 `st.global.f32` (存储到全局内存) 指令, 它们分别对应 `in[i]` 和 `out[i]` 操作. 每一个这样的指令都代表着一次与慢速DRAM的交互.