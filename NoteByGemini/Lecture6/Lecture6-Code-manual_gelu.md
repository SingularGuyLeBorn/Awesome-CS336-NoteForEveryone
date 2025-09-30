### 模板D: 代码实现深度解析

#### 1. 核心功能与目标 (Core Function & Goal)
`manual_gelu` 函数的目标是使用 PyTorch 中最基础的、非融合的张量操作来手动实现 **[GeLU (高斯误差线性单元)](./Lecture6-GeLU.md)** 激活函数的近似计算公式。这个函数本身不追求性能，恰恰相反，它的存在是为了作为一个**性能不佳的基线 (baseline)**，用以凸显**[算子融合](./Lecture6-Kernel-Fusion.md)**的重要性。通过与 PyTorch 内置的 `F.gelu` 或手写的 **[CUDA](./Lecture6-CUDA.md)**/**[Triton](./Lecture6-Triton.md)** 版本进行对比，我们可以量化地看到将多个操作分解执行所带来的巨大性能损失。

#### 2. 参数解析 (Parameters)
*   `x` (torch.Tensor): 输入的张量。

#### 3. 核心逻辑 (Core Logic)
```python
import torch

def manual_gelu(x: torch.Tensor):
    """
    使用基础 PyTorch 操作手动实现 GeLU 的 tanh 近似公式。
    GeLU(x) ≈ 0.5 * x * (1 + tanh[√(2/π) * (x + 0.044715 * x³)])
    """
    # √(2/π) ≈ 0.79788456
    
    # 这里的每一步数学运算都可能对应一次独立的 CUDA Kernel 调用
    
    # 1. 计算 x³
    x_cubed = x * x * x # 或者 x ** 3
    
    # 2. 计算 0.044715 * x³
    inner_term_1 = 0.044715 * x_cubed
    
    # 3. 计算 (x + 0.044715 * x³)
    inner_term_2 = x + inner_term_1
    
    # 4. 乘以常数 √(2/π)
    inner_term_3 = 0.79788456 * inner_term_2
    
    # 5. 计算 tanh
    tanh_out = torch.tanh(inner_term_3)
    
    # 6. 计算 1 + tanh(...)
    final_term_1 = 1 + tanh_out
    
    # 7. 乘以 0.5 * x
    final_term_2 = 0.5 * x
    
    # 8. 最终乘法
    return final_term_2 * final_term_1
```

#### 4. 与理论的连接 (Connection to Theory)
这个函数是**[算子融合](./Lecture6-Kernel-Fusion.md)**理论的一个完美**反例**。

*   **多 Kernel 调用**：代码中的每一行数学运算（乘法、加法、`tanh`）都可能在底层触发一次独立的 CUDA Kernel 启动。例如，`x * x * x` 就可能被分解为两次乘法 Kernel。
*   **内存往返 (Memory Round-trips)**：每次 Kernel 执行都遵循一个“读-算-写”的模式。例如，在计算 `inner_term_1` 时，GPU 需要从全局内存（DRAM）读取 `x_cubed`，计算乘法，然后将结果 `inner_term_1` 写回全局内存。紧接着，在计算 `inner_term_2` 时，又需要从全局内存中读取 `x` 和 `inner_term_1`。这种中间结果在全局内存中的频繁读写是导致性能低下的主要原因，因为 DRAM 的访问速度远慢于 SM 的计算速度。
*   **低[算术强度](./Lecture6-Arithmetic-Intensity.md)**：每个独立的操作都是典型的**[内存密集型](./Lecture6-Memory-vs-Compute-Bound.md)**操作。整个函数虽然包含多次计算，但由于被分解开，其整体的有效算术强度非常低。

通过**[性能分析](./Lecture6-Profiling.md)**，我们可以清晰地看到 `manual_gelu` 的执行剖面是由一系列 `mul`, `add`, `tanh` 等小型 Kernel 组成的，这与融合后的版本只有一个单一的大 Kernel 形成了鲜明对比。这直观地证明了，仅仅改变代码的组织方式（将多个操作写入一个 Kernel），就能带来数量级的性能提升。