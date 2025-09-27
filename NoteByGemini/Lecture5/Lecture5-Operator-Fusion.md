### 1. 概念定义

**算子融合 (Operator Fusion)**，也称为**核函数融合 (Kernel Fusion)**，是一种编译器优化技术，旨在通过将多个独立的计算操作（算子或 CUDA 核函数）合并成一个单一的、更复杂的核函数来提升 GPU 性能。

其核心目标是**减少对 GPU 全局内存的访问**并**降低核函数启动开销**。

### 2. 为何需要算子融合？

在像 PyTorch 这样的深度学习框架中，一个简单的数学表达式，如 `y = sin(x)**2 + cos(x)**2`，在未经优化的情况下，会被分解成多个独立的计算步骤，每个步骤都会启动一个独立的 CUDA 核函数：
1.  `tmp1 = sin(x)` (启动一个 `sin` 核函数, 结果写入全局内存)
2.  `tmp2 = cos(x)` (启动一个 `cos` 核函数, 结果写入全局内存)
3.  `tmp3 = tmp1 ** 2` (启动一个 `pow` 核函数, 结果写入全局内存)
4.  `tmp4 = tmp2 ** 2` (启动一个 `pow` 核函数, 结果写入全局内存)
5.  `y = tmp3 + tmp4` (启动一个 `add` 核函数, 结果写入全局内存)

这个过程存在两大性能问题：

1.  **巨大的内存 I/O 开销**: 每个中间结果（`tmp1` 到 `tmp4`）都被写入到缓慢的**[全局内存](./Lecture5-GPU-Memory-Hierarchy.md)**，然后又在下一步被读取出来。这种来回的数据传输（如讲座中的工厂输送带比喻）是极其低效的，尤其对于这些计算量很小但内存访问量很大的**内存密集型 (memory-bound)** 操作。

2.  **核函数启动开销 (Kernel Launch Overhead)**: 每次从 CPU 调用一个 CUDA 核函数，都有一定的固定开销（涉及到驱动程序调用、上下文切换等）。对于执行时间极短的小操作，这个启动开销可能会占到总执行时间的很大一部分。

### 3. 融合的工作原理

算子融合通过将上述多个操作合并成一个核函数来解决这些问题。融合后的核函数会在一个线程内，一次性完成所有计算：

```c
// 融合后的伪代码
__global__ void fused_kernel(float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val_x = x[i]; // 从全局内存读取一次
        float val_sin = sinf(val_x);
        float val_cos = cosf(val_x);
        float val_sin_sq = val_sin * val_sin;
        float val_cos_sq = val_cos * val_cos;
        y[i] = val_sin_sq + val_cos_sq; // 写入全局内存一次
    }
}
```

**融合后的优势**:
- **最大化数据局部性**: 输入数据 `x[i]` 被读取到 **[SM](./Lecture5-Streaming-Multiprocessor.md)** 的快速寄存器后，所有的中间计算（sin, cos, pow, add）都在寄存器中完成，完全避免了对全局内存的中间读写。
- **减少开销**: 原本 5 次的核函数启动被减少到只有 1 次。
- **提升算术强度**: 通过消除大量的内存 I/O，融合操作显著提升了程序的**[算术强度](./Lecture5-Roofline-Model.md)**，使其更能发挥 GPU 的计算潜力。

### 4. 应用

算子融合是现代深度学习编译器（如 PyTorch 2.0 的 `torch.compile` 和 TensorFlow 的 XLA）的核心优化手段之一。它们能够自动分析计算图，识别出可以融合的连续操作序列，并生成高效的融合核函数。

特别地，**逐元素操作 (element-wise operations)**、**广播操作 (broadcasting)** 和**归约操作 (reductions)** 是最常见的融合候选项。在像 **[FlashAttention](./Lecture5-FlashAttention.md)** 这样的高级优化中，算子融合也被用来将 Softmax 的指数运算等操作与矩阵乘法紧密结合起来，以避免物化巨大的中间注意力矩阵。