### 模板D: 代码实现深度解析

#### 1. 核心功能与目标 (Core Function & Goal)
`manual_softmax` 函数的目标是使用 PyTorch 的基础操作，分步实现一个数值稳定的 **[Softmax](./Lecture6-Softmax.md)** 函数。与 `manual_gelu` 类似，它的主要教学目的是作为一个性能不佳的**基线**，清晰地展示出非融合实现方式的性能缺陷。因为它包含了一个归约操作（求和），所以它暴露出的内存访问问题比逐元素操作更严重。

#### 2. 参数解析 (Parameters)
*   `x` (torch.Tensor): 输入的二维张量（矩阵），形状为 `(M, N)`。Softmax 将对每一行独立进行操作。

#### 3. 核心逻辑 (Core Logic)
```python
import torch

def manual_softmax(x: torch.Tensor):
    # M: 行数, N: 列数
    M, N = x.shape
    
    # 这是一个非常低效的实现，涉及多次对全局内存的完整读写
    
    # 步骤 1: 求每行的最大值（用于数值稳定性）
    # 内存访问: 读取整个矩阵 (M*N reads)，写入结果向量 (M writes)
    x_max = x.max(dim=1, keepdim=True)[0]
    
    # 步骤 2: 每行减去其最大值
    # 内存访问: 读取矩阵 (M*N reads) 和最大值向量 (M reads, 但会广播), 写入新矩阵 (M*N writes)
    x = x - x_max
    
    # 步骤 3: 计算指数
    # 内存访问: 读取矩阵 (M*N reads), 写入新矩阵 (M*N writes)
    numerator = torch.exp(x)
    
    # 步骤 4: 求每行指数的总和（归一化分母）
    # 内存访问: 读取 numerator 矩阵 (M*N reads), 写入结果向量 (M writes)
    denominator = numerator.sum(dim=1, keepdim=True)
    
    # 步骤 5: 归一化
    # 内存访问: 读取 numerator (M*N reads) 和 denominator (M reads, 广播), 写入最终结果 (M*N writes)
    y = numerator / denominator
    
    # 粗略统计总内存访问量：
    # 读: (MN + MN+M + MN + MN + MN+M) ≈ 5MN reads
    # 写: (M + MN + MN + M + MN) ≈ 3MN writes
    # 总计约 8MN 次数据移动。
    # 而一个理想的融合实现，只需要 MN reads 和 MN writes。性能差距巨大。
    return y
```
*注：在课程代码中，部分 `.max()` 和 `.sum()` 未使用 `keepdim=True`，后续需要使用 `[:, None]` 进行广播，逻辑是等价的。这里为了清晰统一使用 `keepdim=True`。*

#### 4. 与理论的连接 (Connection to Theory)
这个实现是**[算子融合](./Lecture6-Kernel-Fusion.md)**重要性的一个教科书级别的案例，特别是在涉及**归约**操作时。

*   **数据依赖与内存瓶颈**：Softmax 的计算具有内在的数据依赖性（例如，必须先知道一行的所有值才能计算最大值和总和）。在非融合的实现中，这种依赖性被转化为了多次对 GPU 全局内存（DRAM）的完整遍历。
*   **DRAM 带宽的浪费**：我们可以看到，输入矩阵 `x` 以及其各种中间形式（`numerator` 等）被完整地从 DRAM 读到 SM 中，处理完后又被写回 DRAM，然后下一个操作再将其读出。这个过程重复了 5 次之多。这极大地浪费了宝贵的内存带宽，使得整个操作成为一个严重的**[内存密集型](./Lecture6-Memory-vs-Compute-Bound.md)**任务。
*   **与融合实现的对比**：一个理想的融合 Kernel（如 **[`triton_softmax`](./Lecture6-Code-triton_softmax.md)**）会将一整行数据一次性加载到 SM 的高速片上内存（寄存器/共享内存）中。然后，所有的计算——求最大值、减法、指数、求和、除法——都在这个高速的“工作台”上完成，完全不需要与 DRAM 进行任何中间交互。最后，计算完成的最终结果被一次性写回 DRAM。这种“一次读，一次写”的模式，与 `manual_softmax` 的多次往返形成了鲜明对比，性能差异由此产生。