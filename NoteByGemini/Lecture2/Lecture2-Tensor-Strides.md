# 专题笔记: 张量步长 (Tensor Strides)

### 1. 核心概念

**张量步长 (Tensor Stride)** 是一个理解 **[PyTorch](./Lecture2-PyTorch.md)** 如何在内存中高效地表示和操作**[张量](./Lecture2-Tensors.md)**的关键底层概念. 它描述了为了在张量的某个特定维度上移动到下一个元素,需要在底层的一维**存储(storage)**中“跳跃”多少个元素. 

每个张量都有一个与之关联的步长元组(tuple),其长度与张量的维度数相同. `stride[k]` 的值表示在第 `k` 个维度上,索引增加 1 所对应的内存位置的增量. 

### 2. 张量、存储与步长的关系

要理解步长,必须先理解 PyTorch 的张量存储机制: 

1.  **存储 (Storage)**: 无论一个张量有多少个维度,它的所有数据实际上都存储在一块**连续的一维内存块**中. 这个内存块就是 `storage`. 
2.  **张量 (Tensor)**: 张量对象本身更像是一个“视图”或“解释器”. 它不直接持有数据,而是持有一系列元数据,包括: 
    *   指向底层 `storage` 的指针. 
    *   张量的形状(shape). 
    *   张量的步长(strides). 
    *   数据的偏移量(offset),即张量的第一个元素在 `storage` 中的起始位置. 

**如何通过步长定位元素？**
要访问一个 n 维张量 `T` 在索引 `(i_0, i_1, ..., i_{n-1})` 处的元素,其在底层一维 `storage` 中的位置 `loc` 可以通过以下公式计算: 

`loc = offset + i_0 * stride[0] + i_1 * stride[1] + ... + i_{n-1} * stride[n-1]`

### 3. 示例解析

让我们通过一个具体的例子来理解步长是如何工作的. 

```python
import torch

# 创建一个 3x4 的张量
x = torch.arange(12).reshape(3, 4)
print("Tensor x:\n", x)
# Tensor x:
# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 查看其底层的 storage (一维连续内存)
print("\nStorage:", x.storage())
# Storage: 0 1 2 3 4 5 6 7 8 9 10 11 ...

# 查看其形状和步长
print("Shape:", x.shape)   # torch.Size([3, 4])
print("Stride:", x.stride()) # (4, 1)
```

**步长 `(4, 1)` 的含义**: 

*   **`stride[0] = 4`**: 对应第0维(行). 为了从一行移动到下一行(例如,从 `x[0,0]` 到 `x[1,0]`),你需要- 在 `storage` 中向前跳跃 **4** 个元素(从 `0` 跳到 `4`). 
*   **`stride[1] = 1`**: 对应第1维(列). 为了从一列移动到下一列(例如,从 `x[0,0]` 到 `x[0,1]`),你只需要在 `storage` 中向前移动 **1** 个元素(从 `0` 移动到 `1`). 

### 4. 步长与零拷贝视图操作

步长机制是 PyTorch 能够实现高效、零拷贝视图操作的魔法所在. 当我们执行像 `transpose` 这样的操作时,PyTorch **不会移动底层 `storage` 中的任何数据**. 它只是创建了一个新的张量对象,并赋予它一个新的步长值. 

```python
# 对 x 进行转置
y = x.T # 等价于 x.transpose(0, 1)
print("\nTransposed Tensor y:\n", y)
# Transposed Tensor y:
# tensor([[ 0,  4,  8],
#         [ 1,  5,  9],
#         [ 2,  6, 10],
#         [ 3,  7, 11]])

# y 和 x 共享同一个 storage
print("x storage id:", x.storage().data_ptr())
print("y storage id:", y.storage().data_ptr()) # id 相同

# 查看 y 的形状和步长
print("\nShape of y:", y.shape)   # torch.Size([4, 3])
print("Stride of y:", y.stride()) # (1, 4)
```

**步长 `(1, 4)` 的含义**: 

*   **`stride[0] = 1`**: 对于转置后的 `y`,为了在第0维(行)上移动(例如,从 `y[0,0]` 到 `y[1,0]`),我们实际上是在原始 `storage` 中从 `0` 移动到 `1`,所以需要跳跃 **1** 个元素. 
*   **`stride[1] = 4`**: 为了在第1维(列)上移动(例如,从 `y[0,0]` 到 `y[0,1]`),我们实际上是在原始 `storage` 中从 `0` 移动到 `4`,所以需要跳跃 **4** 个元素. 

通过简单地交换步长值,PyTorch 就实现了一次“免费”的转置操作. 

### 5. 与内存连续性的关系

这个概念与**[连续与非连续张量](./Lecture2-Contiguous-vs-Non-contiguous-Tensors.md)**紧密相关. 
*   一个**连续的(Contiguous)**张量,其步长通常是按维度大小降序排列的(例如 `(4, 1)` for a 3x4 tensor). 这意味着内存布局和逻辑布局是一致的. 
*   经过 `transpose` 后的张量 `y`,其步长是 `(1, 4)`,不符合这个模式,因此它是一个**非连续的(Non-contiguous)**张量. 

理解步长有助于我们深入理解 PyTorch 的内存管理策略,解释为何某些操作高效而某些操作(如 `view`)对张量的内存布局有特定要求. 

---
**关联知识点**
*   [张量 (Tensors)](./Lecture2-Tensors.md)
*   [连续与非连续张量](./Lecture2-Contiguous-vs-Non-contiguous-Tensors.md)
*   [PyTorch](./Lecture2-PyTorch.md)