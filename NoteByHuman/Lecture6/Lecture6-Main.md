# 第六讲:手写高性能算子

> 居然还是Tatsunori Hashimoto这位老师,他一开始不像Percy一样用py脚本用传统的PDF我还不太习惯,他自己一开始也说比较传统:(？)然后我现在听了三节课终于习惯了他又改了！艹 damn. 不过我实际上更喜欢用python代码当课件的形式,因为这意味着我不需要费劲从ppt里粘图片,就很舒服
>
> 不过说实话,就这个课程的前半段来说,做笔记其实并没有那么必要,运行程序比看笔记要更好

大家好. 在本次课程中, 我们将深入探讨如何为GPU编写高性能代码. 第二次作业的部分内容就会要求大家进行大量的性能分析(profiling), 甚至为Flash Attention V2编写自己的Triton核函数(kernel), 致力于实现极致的性能. 因此, 本次课程的目标就是为构建语言模型中的标准组件编写一些高性能代码. 

课程计划如下:

1. **GPU基础回顾**: 快速回顾GPU的关键组件, 为理解后续内容打下基础. 
2. **基准测试(Benchmarking)与性能分析(Profiling)**: 学习衡量与诊断代码性能的基本功, 这对完成作业和日常工作都至关重要. 
3. **编写核函数**: 我们将亲手实践, 分别使用传统的**CUDA C++**和现代的**Triton**编写核函数. 
4. **JIT编译器**: 最后, 我们会采取一种简单但效果很好的方法, 即使用PyTorch现有的即时(Just-In-Time)编译器`torch.compile`来自动为我们进行优化. 

在整个过程中, 我们将对所有这些方法进行比较, 并进行深入剖析, 甚至会探究到**PTX**——一种非常接近机器码的GPU汇编语言, 以理解当我们编写这些代码时, GPU在底层到底做了什么. 

### 1. GPU核心概念回顾

好的, 现在我们来回顾一下GPU的工作原理. 无论是A100还是H100, 其核心都由大量的**流式多处理器(Streaming Multiprocessors, SMs)** 构成. 

#### 1.1 硬件架构

- **计算单元**: 每个SM内部包含众多可以执行计算的单元, 如INT32和FP32核心. 一个SM会启动大量的线程来执行计算. 
- **内存层级**:
  - **DRAM (全局内存)**: 容量巨大(A100可达80GB), 但速度较慢. 
  - **L1/L2缓存**: 容量小得多, 但速度快很多. L1缓存是每个SM独享的. 
  - **寄存器文件(Register File)**: 这是**每个线程**可以访问的**最快**的内存, 容量极小. 今天在编写高性能GPU代码时, 我们会大量使用这些寄存器. 

![GPU硬件架构图](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg)

> 图 1: GPU的硬件架构示意图, 展示了SMs以及DRAM, 缓存, 寄存器等内存层级. 

#### 1.2 执行模型

GPU的并行计算是通过一个层级化的结构来组织的:

- **线程(Thread)**: 执行最基础的计算任务. 例如, 如果你要对一个向量进行操作, 你编写的代码会让每个线程进入并处理该向量的一个或几个元素. 
- **线程块(Thread Block)**: 由一组线程构成. **一个线程块会被调度到单个SM上执行**. 这是我们思考和编写Triton代码时的原子单元. 
- **网格(Grid)**: 由所有线程块共同组成, 对应整个计算任务. 

**为什么要有线程块这个概念呢?** 为什么不直接在全局上下文中使用线程? 关键在于**通信和共享内存(Shared Memory)**. 同一个线程块内的线程可以通过SM内部的一块高速共享内存进行快速的数据交换和同步, 其速度堪比L1缓存. 当你需要执行像矩阵乘法这样需要在线程之间传递信息的操作时, 块内通信是非常快的. 相反, 跨线程块的通信则非常昂贵, 需要通过慢速的全局内存进行. 因此, 任何需要共享的数据, 都应该尽量保持在同一个线程块内. 

#### 1.3 Warp(线程束)与Wave

对于性能而言, **Warp**是一个重要组成部分. 当我们实际运行程序时, 线程会被硬件自动分组, 每32个连续的线程组成一个Warp. 同一个Warp中的32个线程会以“步调一致”的方式(SIMD)被一次性执行. 

这种设计的优势在于**减少了控制逻辑的开销**. 硬件不需要为每个线程都配备一个独立的控制单元, 只需要为每32个线程的Warp配备一个即可. 这也是GPU能将更多芯片面积用于计算单元, 而非像CPU那样用于复杂控制逻辑和分支预测的原因之一. 

#### 1.4 算术强度 (Arithmetic Intensity)

这也许是本次回顾中最重要的概念之一. **算术强度**衡量了计算量与内存访问量的比例:

$$
\text{Arithmetic Intensity} = \frac{\text{浮点运算次数 (FLOPs)}}{\text{内存访问字节数 (Bytes)}}
$$

我们希望保持高的算术强度, 因为计算性能的扩展速度远快于内存性能的扩展速度. 这导致很多时候计算任务最终会变成**内存密集型(Memory-Bound)**, 即计算单元因为数据供应不足而闲置, 无法充分发挥其计算能力. 

一个普遍的规律是: **精心实现的矩阵乘法是计算密集型的, 而其他绝大多数操作(如逐元素加法, 激活函数)都是内存密集型的**. 我们将尝试巧妙地减少内存受限的操作数量, 或减轻内存受限的严重程度. 

### 2. 基准测试与性能分析: 优化的前提

如果要记住一个高层次的要点, 那就是: **如果你想编写高性能代码, 就必须对你的代码进行基准测试和性能分析**. 这听起来显而易见, 但我见过很多情况, 有人凭感觉说“我认为瓶颈在这里”, 然后花三小时去优化, 结果发现那根本不是瓶颈. 这是一种时间的错配. 使用专业的性能分析器, 你就能精确地看到瓶颈在哪里, 机器到底在做什么. 

#### 2.1 理论的极限与实践的必要性

理论是有极限的. 你可以很好地对系统层面进行推理, 但硬件架构在某种程度上是很难纯粹通过理论来预测的. 你可以思考屋顶线模型(roofline model)等理论, 但你的矩阵乘法到底有多快? 真实性能可能取决于你使用的库版本、具体的硬件型号、甚至是一些你并不完全了解的微码细节. 所以, **最终在开发这些东西时, 你必须进行端到端的基准测试**. 

#### 2.2 我们的实验对象: 一个简单的MLP

为了贯穿整个课程, 我们将使用一个非常简单的多层感知机(MLP)作为实验对象. 它的定义和运行方式如下:

```python
import torch
import torch.nn as nn
from typing import Callable

# 假设 get_device() 函数已经定义, 用于获取CUDA设备
# from torch_util import get_device

class MLP(nn.Module):
    """一个简单的MLP: linear -> GeLU -> linear -> GeLU -> ..."""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x

def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    """
    设置并返回一个执行MLP前向和后向传播的可调用对象.
    这个函数封装了模型和数据的创建过程.
    """
    # 定义一个模型 (使用随机权重)
    model = MLP(dim, num_layers).to(get_device())
    # 定义一个输入 (随机)
    x = torch.randn(batch_size, dim, device=get_device())
  
    def run():
        # 运行模型 `num_steps` 次 (注意: 为了简化, 没有优化器更新步骤)
        for step in range(num_steps):
            # 前向传播
            y = model(x).mean()
            # 后向传播
            y.backward()
    return run
```

#### 2.3 基准测试 (Benchmarking)

基准测试就是测量执行操作的墙上时钟时间(wall-clock time). 它有一些微妙之处, 如果不注意, 很容易掉入陷阱. 我们的目标是比较不同实现的优劣, 以及理解性能如何随输入规模扩展. 

我们将使用下面这个`benchmark`函数, 它包含了两个至关重要的实践:

```python
import time
from typing import Callable
import torch

def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    """
    对一个函数进行基准测试, 运行 num_trials 次并返回平均时间.
  
    Args:
        description: 对测试内容的描述.
        run: 要进行基准测试的可调用对象.
        num_warmups: 预热运行次数, 避免测量首次执行的编译等开销.
        num_trials: 实际测量次数, 取平均值以减少波动.
    """
    # 1. 预热 (Warm-up): 首次运行时可能涉及编译、缓存加载等一次性开销.
    # 我们关心的是稳定状态下的性能, 因此先运行几次.
    for _ in range(num_warmups):
        run()

    # 2. 同步 (Synchronization): 确保在开始计时前, GPU已完成所有先前任务.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for trial in range(num_trials):  # 多次运行以捕捉方差
        start_time = time.time()
        run()  # 实际执行计算
        # 3. 再次同步: 确保run()中的所有GPU任务都完成后再记录结束时间.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # 转换为毫秒

    mean_time = sum(times) / len(times)
    print(f"{description}: {mean_time:.3f} ms")
    return mean_time
```

请记住其中两个重要的部分: 

1. **始终进行预热 (Warm-up)**: 当你第一次运行PyTorch代码时, 底层会发生很多一次性的初始化工作, 比如编译机器码、将指令发送到GPU等. 如果你不进行预热, 你测量的就是包含了这些启动开销的速度, 而非我们真正关心的稳态运行速度. 
2. **确保调用`torch.cuda.synchronize()`**: 这是最容易出错的地方. CPU和GPU是两个**异步执行**的单元. 当CPU上的Python代码执行一个GPU操作时, 它只是把这个任务**提交(dispatch)**到GPU的任务队列中, 然后CPU会**立即继续**执行下一行代码, 而**不会等待**GPU完成. 

   如果你不调用`synchronize()`, `end_time`就会在GPU还在埋头苦干时被记录下来. 这会导致你测量到的时间无限接近于零, 仿佛你的大型矩阵乘法瞬间就完成了, 这绝对是不可能的. `torch.cuda.synchronize()`会强制CPU暂停, 直到GPU执行完队列中所有已提交的任务, 这样我们才能测量到完整的、真实的执行时间. 

#### 2.4 性能分析 (Profiling)

基准测试告诉我们代码**慢不慢**, 而性能分析则告诉我们**慢在哪里**. 它不仅能展示时间花费在哪个函数上, 更能让你深入了解PyTorch底层到底调用了哪些具体的CUDA核函数, 从而对硬件的实际执行过程有一个更清晰的直观认识. 

```python
import torch
import torch.profiler
from torch.profiler import ProfilerActivity
from typing import Callable

def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    """
    一个用于性能分析的工具函数. 

    参数:
    - description (str): 本次性能分析的描述,会用于输出文件的命名. 
    - run (Callable): 需要被分析的函数. 
    - num_warmups (int): 在正式分析前,预热运行的次数. 默认为1. 
    - with_stack (bool): 是否采集堆栈信息,用于生成火焰图等可视化结果. 默认为False. 
    """
    # 预热(Warmup)阶段: 先运行几次以确保 CUDA 初始化等一次性开销不影响分析结果
    for _ in range(num_warmups):
        run()
  
    # 如果使用了 CUDA,则同步以确保所有预热的 CUDA kernel 都已执行完毕
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 使用 torch.profiler.profile 作为上下文管理器来启动性能分析
    with torch.profiler.profile(
            # activities: 指定要分析的活动类型,这里同时分析 CPU 和 CUDA
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # with_stack: 决定是否记录算子的堆栈信息,方便追溯算子来源
            with_stack=with_stack,
            # experimental_config: 实验性配置,这里为了导出堆栈信息需要开启
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        # 正式运行需要被分析的代码
        run()
        # 再次同步,确保 `run()` 中的所有 CUDA kernel 都在 profiler 退出前完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 打印性能分析结果表格
    # key_averages() 会聚合所有算子的数据
    # sort_by="cuda_time_total" 表示按 CUDA 总耗时降序排列
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
  
    # 以下两行被注释掉了,可能是用于在特定环境(如Jupyter)中直接显示结果
    # text(f"## {description}")
    # text(table, verbatim=True)

    # 如果 with_stack 为 True,则导出堆栈信息用于生成火焰图
    if with_stack:
        # 定义输出堆栈信息的文件路径
        text_path = f"var/stacks_{description}.txt"
        # prof.export_stacks() 可以将采集到的堆栈信息导出为文本文件
        # 这个文件可以被 flameprof.py 等工具用来生成 SVG 格式的火焰图
        prof.export_stacks(text_path, "self_cuda_time_total")

    # 返回分析结果的表格对象
    return table

add_profile = profile("add", run_operation2(dim=2048, operation=add_function))
```

通过Profiler, 我们能看到一个简单的加法操作背后, PyTorch会调用`aten::add` C++接口, 最终分发到名为`vectorized_elementwise_kernel`的CUDA核函数. 我们甚至能看到`cudaLaunchKernel`(启动核函数的CPU开销)和`cudaDeviceSynchronize`(同步等待的开销)所占用的时间. 对于更复杂的模型, 我们需要更强大的可视化工具, 如**NVIDIA Nsight Systems**, 它可以揭示CPU与GPU之间复杂的异步交互. 

- [剖析CPU与GPU的异步执行模型](./Lecture6-CPU-GPU-Asynchronicity.md)

### 3. 核函数融合 (Kernel Fusion): 性能优化的黄金法则

**核函数融合**是GPU编程中最重要的优化思想之一. 我们可以用一个形象的比喻来理解它: DRAM(全局内存)就像一个**仓库**, 容量大但取货慢; SRAM(共享内存/缓存)就像SM内部的**工厂车间**, 空间小但加工快. 

如果我们执行一系列操作, 比如`y = a * b + c`, 未经优化的朴素实现会为每个操作都进行一次“从仓库取货 -> 在工厂加工 -> 送回仓库”的完整流程. 这在慢速的仓库和快速的工厂之间造成了大量不必要的数据往返, 成为性能瓶颈. 

![核函数融合示意图](https://horace.io/img/perf_intro/operator_fusion.png)

> 图 2: 核函数融合将多个独立的读-算-写操作合并为一次读-算-写, 大幅减少了对慢速DRAM的访问. 

为了验证其效果, 我们以**GeLU**激活函数为例. PyTorch的`torch.nn.functional.gelu`是一个高度优化的**融合**实现. 我们可以手动用多个基础PyTorch操作模拟一个**非融合**版本:

```python
def pytorch_gelu(x: torch.Tensor):
    # 使用tanh近似以匹配我们自己的实现
    return torch.nn.functional.gelu(x, approximate="tanh")

def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
```

为了测试它们, 我们需要一个辅助函数来创建数据并返回一个可执行的lambda:

```python
def run_operation1(dim: int, operation: Callable) -> Callable:
    # 设置: 创建一个随机的 dim x dim 矩阵
    x = torch.randn(dim, dim, device=get_device())
    # 返回一个执行该操作的函数
    return lambda : operation(x)

# 运行基准测试
manual_time = benchmark("manual_gelu", run_operation1(16384, manual_gelu))
pytorch_time = benchmark("pytorch_gelu", run_operation1(16384, pytorch_gelu))
```

在一个大规模张量上进行基准测试, 结果惊人:

- **manual_gelu (非融合)**: 8.1 ms
- **pytorch_gelu (融合)**: 1.1 ms

融合版本的速度是朴素实现的**近8倍**. 查看Profiler输出会发现, `manual_gelu`触发了多个独立的CUDA核函数(乘法, 加法, tanh等), 而`pytorch_gelu`只调用了一个`gelu_kernel`. 这有力地证明了减少全局内存访问次数带来的巨大性能提升. 

### 4. 编写自定义核函数: 掌握极致性能

既然核函数融合如此强大, 当PyTorch没有提供我们需要的融合算子时, 我们可以自己编写. 

#### 4.1 传统方式: CUDA C++

这是最底层、控制力最强的方式. 它使用C++的扩展来编写可以直接在GPU上执行的核函数. 我们需要一个`load_inline`这样的工具, 在Python中方便地编译和加载我们的C++和CUDA代码. 

```python
from torch.utils.cpp_extension import load_inline

def create_cuda_gelu():
    # CUDA代码: 包含完整的核函数逻辑
    cuda_gelu_src = """
    #include <cmath> // for tanhf
    __global__ void gelu_kernel(const float* in, float* out, int num_elements) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < num_elements) {
            float x = in[i];
            const float c1 = 0.79788456f;
            const float c2 = 0.044715f;
            float x_cubed = x * x * x;
            float inner = c1 * (x + c2 * x_cubed);
            float tanh_val = tanhf(inner);
            out[i] = 0.5f * x * (1.0f + tanh_val);
        }
    }
    """
  
    # C++代码: 定义Python调用的接口函数
    cpp_gelu_src = """
    #include <torch/extension.h>
    // 前置声明CUDA核函数
    void gelu_kernel(const float* in, float* out, int num_elements);
  
    torch::Tensor gelu(torch::Tensor x) {
        TORCH_CHECK(x.is_cuda() && x.is_contiguous());
        auto y = torch::empty_like(x);
        const int num_elements = x.numel();
        const int block_size = 1024;
        const int num_blocks = (num_elements + block_size - 1) / block_size;
        gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
        return y;
    }
    """
  
    # 编译CUDA代码并将其绑定到Python模块
    if not torch.cuda.is_available(): return None
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        name="inline_gelu",
        verbose=True,
    )
    return module.gelu

# cuda_gelu = create_cuda_gelu()
# if cuda_gelu:
#     benchmark("cuda_gelu", run_operation1(16384, cuda_gelu))
```

这个实现性能优异(约1.8 ms), 但开发和调试都非常复杂. 

- [代码解析: 手写CUDA C++ GeLU核函数](./Lecture6-Code-CUDA-GeLU.md)

#### 4.2 现代方式: Triton

Triton是OpenAI开发的基于Python的DSL, 它将编程抽象从单个线程提升到线程块, 极大地简化了高性能核函数的编写. 

```python
import triton
import triton.language as tl

@triton.jit
def triton_gelu_kernel(
    x_ptr,      # 指向输入张量的指针
    y_ptr,      # 指向输出张量的指针
    num_elements, # 张量中的元素总数
    BLOCK_SIZE: tl.constexpr, # 每个程序实例处理的元素数量
):
    # 获取当前程序(线程块)的ID
    pid = tl.program_id(axis=0)
    # 计算当前块要处理的数据的偏移量向量
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建边界掩码, 防止处理超出范围的内存
    mask = offsets < num_elements
    # 从全局内存向量化加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    # 向量化计算
    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Triton没有内置tanh, 我们需要手动实现 tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp_val = tl.exp(2 * a)
    tanh_val = (exp_val - 1) / (exp_val + 1)
    y = 0.5 * x * (1 + tanh_val)
    # 向量化存储结果回全局内存
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_gelu(x: torch.Tensor):
    """Triton GeLU的封装函数"""
    assert x.is_cuda and x.is_contiguous()
    y = torch.empty_like(x)
    num_elements = x.numel()
    block_size = 1024  # 每个线程块中的线程数
    # 使用triton.cdiv计算需要的块数 (向上取整)
    num_blocks = triton.cdiv(num_elements, block_size)
    # 启动核函数
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)
    return y
```

Triton版本同样达到了约1.8 ms的性能, 但代码完全在Python中, 更易读写. 

- [代码解析: 使用Triton编写GeLU核函数](./Lecture6-Code-Triton-GeLU.md)

#### 4.3 便捷方式: `torch.compile`

编写和调试自定义核函数仍然是一项耗时的工作. 幸运的是, `torch.compile`能够自动分析Python代码并进行算子融合. 

```python
# 只需一行代码, 即可编译我们之前写的朴素Python版本!
compiled_gelu = torch.compile(manual_gelu)
```

令人惊讶的是, `torch.compile`的性能非常出色, 在GeLU的例子中达到了约1.47 ms, 甚至比我们手写的版本更快. Profiler显示它在底层自动生成了高效的Triton核函数. 

### 5. 进阶核函数: 以Softmax为例处理归约操作

Softmax包含**归约(reduction)** 操作(求最大值, 求和), 比逐元素操作更复杂. 一个朴素的实现需要多次遍历数据, 是典型的内存密集型操作. 

```python
def manual_softmax(x: torch.Tensor):
    # M: 行数, N: 列数
    M, N = x.shape
    # 1. 第一次读写: 计算每行的最大值 (MN次读, M次写)
    x_max = x.max(dim=1, keepdim=True)[0]
    # 2. 第二次读写: 减去最大值 (MN + M次读, MN次写)
    x_shifted = x - x_max
    # 3. 第三次读写: 计算指数 (MN次读, MN次写)
    numerator = torch.exp(x_shifted)
    # 4. 第四次读写: 计算分母 (MN次读, M次写)
    denominator = numerator.sum(dim=1, keepdim=True)
    # 5. 第五次读写: 归一化 (MN + M次读, MN次写)
    y = numerator / denominator
    return y
```

我们可以使用Triton设计一个融合的"一行一块(One Row per Block)"核函数, 一次性完成所有计算. 

- [代码解析: 使用Triton实现高性能Softmax](./Lecture6-Code-Triton-Softmax.md)

### 6. 总结与技术选型指南

经过本次课程的探索, 我们掌握了从高到低不同层次的GPU性能优化方法. 在实际项目中, 我们应该如何选择呢?

- **首先尝试 `torch.compile`**: 它是易用性最高、侵入性最小的选择, 对常见的算子融合模式有很好的自动优化效果, **永远应该作为你的第一尝试**. 
- **当 `torch.compile` 不足时, 选择Triton**: 如果你需要实现一个PyTorch没有的独特算子, 或者需要融合一个非常复杂的操作序列, Triton在易用性和性能之间达到了完美的平衡. 
- **最后才考虑CUDA C++**: 只有在追求硬件的最后一丝性能, 或者需要与庞大的C++代码库深度集成时, 才选择这种开发效率最低、调试最困难的方式. 

总而言之, 编写高性能代码的理念是永恒的, 但工具在不断进步. 理解底层原理, 并善用现代工具, 才能在性能优化的道路上游刃有余. 
