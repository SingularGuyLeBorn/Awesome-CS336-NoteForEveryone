### 1. 核心功能与目标

**分块矩阵乘法 (Tiled Matrix Multiplication)** 是一种在 GPU 上实现高性能矩阵乘法 `C = A * B` 的核心算法。其主要目标是通过应用**[分块技术 (Tiling)](./Lecture5-Tiling.md)**，最大化地利用 **[GPU 内存层级](./Lecture5-GPU-Memory-Hierarchy.md)**，特别是高速的共享内存 (Shared Memory)，来减少对慢速全局内存 (Global Memory) 的访问次数，从而解决内存带宽瓶颈。

### 2. 参数解析

- `Matrix A`: 输入矩阵，维度为 `M x K`，存储在全局内存中。
- `Matrix B`: 输入矩阵，维度为 `K x N`，存储在全局内存中。
- `Matrix C`: 输出矩阵，维度为 `M x N`，存储在全局内存中。
- `TILE_SIZE`: 分块的大小，这是一个关键的调优参数，例如 `16` 或 `32`。我们将使用 `TILE_SIZE x TILE_SIZE` 的方块。

### 3. 核心逻辑 (伪代码与注释)

这个算法通常由一个 CUDA 核函数实现，其中每个**[线程块 (Block)](./Lecture5-GPU-Execution-Model.md)** 负责计算输出矩阵 C 的一个 `TILE_SIZE x TILE_SIZE` 的子块（瓦片）。

```cpp
// 定义共享内存，用于存放 A 和 B 的瓦片
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// 获取当前线程在线程块内的局部索引 (tx, ty)
int tx = threadIdx.x;
int ty = threadIdx.y;

// 获取当前线程块在网格中的索引 (block_x, block_y)
int block_x = blockIdx.x;
int block_y = blockIdx.y;

// 计算当前线程负责计算的输出 C 元素在全局内存中的最终位置
int row = block_y * TILE_SIZE + ty;
int col = block_x * TILE_SIZE + tx;

// 初始化一个寄存器变量，用于累加部分和
float partial_sum = 0.0f;

// --- 主循环：遍历 A 的行和 B 的列来计算一个 C 瓦片 ---
// num_tiles = K / TILE_SIZE
for (int t = 0; t < num_tiles; ++t) {

    // --- 步骤 1: 协同加载瓦片到共享内存 ---
    // 每个线程从全局内存加载 A 的一个元素到共享内存 As 中
    As[ty][tx] = A[row * K + (t * TILE_SIZE + tx)];

    // 每个线程从全局内存加载 B 的一个元素到共享内存 Bs 中
    Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];

    // --- 步骤 2: 块内同步 ---
    // 确保所有线程都已将数据加载到共享内存，再进行下一步计算
    __syncthreads();

    // --- 步骤 3: 在共享内存中进行计算 ---
    // 每个线程负责计算其对应 C 元素的部分和
    // 循环 TILE_SIZE 次，重用共享内存中的数据
    for (int k = 0; k < TILE_SIZE; ++k) {
        partial_sum += As[ty][k] * Bs[k][tx];
    }

    // --- 步骤 4: 再次同步 (为下一次循环做准备) ---
    // 确保所有线程都已完成本次计算，再进入下一次瓦片加载
    __syncthreads();
}

// --- 步骤 5: 将最终结果写回全局内存 ---
// 所有瓦片处理完毕后，将寄存器中累加的最终结果写入 C
C[row * N + col] = partial_sum;
```

### 4. 与理论的连接

- **[分块技术 (Tiling)](./Lecture5-Tiling.md)**: 算法的核心就是将大矩阵分解为小的瓦片进行处理。
- **[GPU 内存层级](./Lecture5-GPU-Memory-Hierarchy.md)**:
    - **全局内存**: 矩阵 A, B, C 的初始和最终存储位置。
    - **共享内存**: `As` 和 `Bs` 数组被明确声明在 `__shared__` 内存中，用于缓存瓦片，这是性能提升的关键。
    - **寄存器**: `partial_sum` 变量存储在每个线程私有的、最快的寄存器中，用于高效地进行累加。
- **[GPU 执行模型](./Lecture5-GPU-Execution-Model.md)**:
    - **线程块 (Block)**: 每个线程块计算 C 的一个瓦片，这是任务划分和数据共享的基本单位。
    - **线程 (Thread)**: 每个线程负责加载 A 和 B 的一个元素到共享内存，并计算 C 瓦片中的一个最终元素值。
    - **`__syncthreads()`**: 这是块内同步指令，确保了加载和计算步骤的正确顺序。
- **[内存合并 (Memory Coalescing)](./Lecture5-Memory-Coalescing.md)**: 为了达到最佳性能，上述代码中的加载部分（`A[...]` 和 `B[...]`）需要精心设计，以确保同一个 Warp 内的线程访问的是连续的全局内存地址，从而触发合并读取。实际的高性能库（如 cuBLAS）中的实现会比这个伪代码更复杂，以确保完美的内存合并。