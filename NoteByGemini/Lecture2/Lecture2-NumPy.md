# 专题笔记: NumPy

### 1. 核心概念

**NumPy (Numerical Python)** 是 Python 语言的一个开源库,它是进行科学计算和数据分析的基础. NumPy 的核心是其强大的 **N 维数组对象 `ndarray`**,它是一个由相同类型元素组成的多维网格. 

在深度学习的早期,以及在许多数据预处理和后处理任务中,NumPy 扮演着至关重要的角色. 虽然现代深度学习框架如 **[PyTorch](./Lecture2-PyTorch.md)** 拥有自己的**[张量(Tensor)](./Lecture2-Tensors.md)**对象,但 PyTorch 的张量在设计上深受 NumPy `ndarray` 的启发,并且两者之间可以非常高效地进行转换. 

### 2. NumPy `ndarray` 的关键特性

*   **高效性**: NumPy 数组在内存中是**连续存储**的,并且其核心操作由高度优化的 C 或 Fortran 代码实现. 这使得在 NumPy 数组上进行向量化(vectorized)的数学运算远比在 Python 的原生列表(list)上进行循环快得多. 
*   **同质性**: 一个 `ndarray` 中的所有元素必须是相同的数据类型(如 `int32`, `float64`). 
*   **丰富的函数库**: NumPy 提供了大量用于数组操作的函数,包括数学运算、逻辑运算、形状操作、排序、选择、线性代数、傅里叶变换和随机数生成等. 
*   **广播 (Broadcasting)**: 这是一套强大的规则,允许 NumPy 在处理不同形状的数组时,能够智能地、隐式地扩展较小的数组以匹配较大的数组,从而实现高效的逐元素操作,而无需创建不必要的副本. 

### 3. NumPy 与 PyTorch 的关系

PyTorch 的设计者们显然从 NumPy 中汲取了大量灵感. `torch.Tensor` 和 `np.ndarray` 在 API 和行为上非常相似. 它们之间存在一种特殊且重要的关系: 

**可以零拷贝地相互转换(当在 CPU 上时). **

*   **`torch.from_numpy(numpy_array)`**: 这个函数可以创建一个 PyTorch 张量,该张量与输入的 NumPy 数组**共享同一块内存**. 这意味着转换几乎是瞬时的,并且不消耗额外内存. 修改其中一个会影响另一个. 
*   **`tensor.numpy()`**: 类似地,这个方法可以将一个**在 CPU 上**的 PyTorch 张量转换为 NumPy 数组,同样是共享内存. 

**代码示例: **
```python
import numpy as np
import torch

# 从 NumPy 数组创建 PyTorch 张量
numpy_arr = np.array([1, 2, 3, 4])
torch_tensor = torch.from_numpy(numpy_arr)

print("Original NumPy array:", numpy_arr)
print("Converted PyTorch tensor:", torch_tensor)

# 修改 NumPy 数组,PyTorch 张量也会改变
numpy_arr[0] = 99
print("\nAfter modifying NumPy array:")
print("NumPy array:", numpy_arr)
print("PyTorch tensor:", torch_tensor) # 也变成了 [99, 2, 3, 4]

# --- 反向转换 ---
# 从 PyTorch 张量创建 NumPy 数组
another_tensor = torch.tensor([5, 6, 7])
back_to_numpy = another_tensor.numpy()

print("\nOriginal PyTorch tensor:", another_tensor)
print("Converted NumPy array:", back_to_numpy)

# 修改 PyTorch 张量,NumPy 数组也会改变
another_tensor[0] = 100
print("\nAfter modifying PyTorch tensor:")
print("PyTorch tensor:", another_tensor)
print("NumPy array:", back_to_numpy) # 也变成了 [100, 6, 7]
```

**注意**: 这种零拷贝转换只适用于在 CPU 上的张量. 如果一个张量在 GPU 上,你需要先用 `.cpu()` 方法将其复制回 CPU,然后才能调用 `.numpy()`. 

### 4. 在课程和实践中的应用

尽管训练过程的核心是在 PyTorch 张量上完成的,但 NumPy 在整个工作流中仍然不可或缺: 

*   **数据预处理**: 在将数据加载到模型之前,经常使用 NumPy 进行复杂的数据清洗、转换和增强操作. 
*   **数据加载**: 如课程中所述,使用 `np.memmap` 来高效地处理无法一次性装入内存的超大规模数据集,是 LLM 训练中的关键技术. 
*   **结果分析与可视化**: 当模型训练完成,得到输出后,通常会将其从 PyTorch 张量转换回 NumPy 数组,以便使用 Matplotlib、Seaborn 等库进行可视化和进一步的统计分析. 
*   **生态系统兼容性**: Python 的科学计算生态系统(SciPy, scikit-learn, Pandas 等)都是基于 NumPy 构建的,因此 NumPy 是与这些库进行交互的桥梁. 

理解 NumPy 是掌握 PyTorch 和整个数据科学生态系统的基础. 

---
**关联知识点**
*   [PyTorch](./Lecture2-PyTorch.md)
*   [张量 (Tensors)](./Lecture2-Tensors.md)
*   [数据加载与内存映射 (Data Loading & Memmap)](./Lecture2-Data-Loading-and-Memmap.md)