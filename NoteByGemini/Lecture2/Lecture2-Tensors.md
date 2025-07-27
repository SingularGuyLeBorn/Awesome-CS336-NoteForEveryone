# 专题笔记: 张量 (Tensors)

### 1. 核心概念

**张量 (Tensor)** 是现代深度学习框架(如 PyTorch 和 TensorFlow)中最基本的数据结构. 从概念上讲,你可以将它理解为一个多维数组. 它是对标量(0维张量)、向量(1维张量)、矩阵(2维张量)等概念的泛化. 

在 **[PyTorch](./Lecture2-PyTorch.md)** 中,`torch.Tensor` 是一个包含单一数据类型元素的多维矩阵. 深度学习中的所有数据——输入样本、模型权重、梯度、激活值——都以张量的形式表示和处理. 

### 2. 张量的关键属性

一个 PyTorch 张量主要有以下几个关键属性: 

*   `shape` (或 `.size()`): 一个元组,描述了张量在每个维度上的大小. 例如,一个形状为 `(32, 10, 128)` 的张量表示它有3个维度,第一个维度大小为32,第二个为10,第三个为128. 
*   `dtype`: 张量中存储的数据类型,例如 `torch.float32`、`torch.int64` 或 `torch.bfloat16`. 不同的数据类型决定了内存占用和计算精度. 
*   `device`: 张量所在的计算设备,如 `'cpu'` 或 `'cuda:0'`. 这是实现 GPU 加速的关键. 
*   `requires_grad`: 一个布尔值. 如果设置为 `True`,**[Autograd](./Lecture2-Autograd.md)** 引擎会自动跟踪在该张量上的所有操作,以便进行自动微分(梯度计算). 模型的可学习参数通常都将此属性设为 `True`. 
*   `grad`: 当对某个张量(如损失函数)调用 `.backward()` 后,所有 `requires_grad=True` 的张量的梯度值会累积到这个属性中. 
*   `grad_fn`: 记录了创建该张量的操作函数,构成了**[反向传播](./Lecture2-Backpropagation.md)**的计算图. 叶子节点张量(用户直接创建的)此属性为 `None`. 

### 3. 内存视图与存储 (`storage`)

这是一个非常重要的概念. 在 PyTorch 中,张量对象本身并不直接存储数据,它更像是一个“视图”或“指针”,指向一块连续的一维内存空间,这块空间被称为 `storage`. 张量对象还包含了一些元数据,如 `shape`、`dtype` 和 **[步长 (stride)](./Lecture2-Tensor-Strides.md)**,用于解释如何从这块一维 `storage` 中索引到多维数组中的元素. 

这种设计带来了极高的效率: 

*   **零拷贝视图**: 像 `transpose()`, `permute()`, `view()`, `narrow()` 等操作,通常不会创建新的内存 `storage`. 它们只是创建了一个新的张量对象,但共享同一个底层的 `storage`,仅仅是改变了新张量的 `shape` 和 `stride`. 这使得这些操作几乎是瞬时完成的,并且不消耗额外内存. 
*   **修改的连锁反应**: 由于共享 `storage`,如果你修改了视图张量中的一个元素,原始张量中对应的元素也会被改变. 

**代码示例: **
```python
import torch

# 创建一个 2x3 的张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# 获取其转置,这是一个视图
y = x.T  # 等价于 x.transpose(0, 1)

# x 和 y 共享同一块内存 storage
print(f"x storage id: {x.storage().data_ptr()}")
print(f"y storage id: {y.storage().data_ptr()}")

# 修改 y 会影响 x
y[0, 0] = 99
print("修改 y 后的 x:\n", x)
# 输出:
# tensor([[99.,  2.,  3.],
#         [ 4.,  5.,  6.]])
```

### 4. 连续性 (`Contiguous`)

由于视图操作的存在,张量在内存中的布局可能会变得不规则. 一个**[连续的张量](./Lecture2-Contiguous-vs-Non-contiguous-Tensors.md)**是指其元素在底层 `storage` 中的排列顺序与其在多维张量中按行优先(row-major)遍历的顺序一致. 

*   **为何重要**: 某些 PyTorch 操作(特别是 `view()`)要求张量必须是连续的,因为它们依赖于数据在内存中的连续布局才能正确、高效地工作. 
*   **如何判断与处理**: 
    *   使用 `.is_contiguous()` 方法可以检查张量是否连续. 
    *   使用 `.contiguous()` 方法可以获取一个张量的连续版本. 如果原始张量已经是连续的,此操作返回其自身; 如果不是,它会创建一个新的、内存连续的副本. 

**代码示例: **
```python
# 上例中的 y (x的转置) 是非连续的
print(f"y is contiguous: {y.is_contiguous()}") # 输出: False

# 尝试在非连续张量上使用 view 会报错
try:
    y.view(6)
except RuntimeError as e:
    print(f"\nError: {e}")

# 先将其转换为连续的
y_contiguous = y.contiguous()
print(f"y_contiguous is contiguous: {y_contiguous.is_contiguous()}") # 输出: True

# 现在可以安全地使用 view
print("\nReshaped contiguous tensor:", y_contiguous.view(6))
```

理解张量的 `storage`、`stride` 和 `contiguous` 概念,对于编写高效、无bug的PyTorch代码至关重要. 

---
**关联知识点**
*   [PyTorch](./Lecture2-PyTorch.md)
*   [张量步长 (Tensor Strides)](./Lecture2-Tensor-Strides.md)
*   [连续与非连续张量](./Lecture2-Contiguous-vs-Non-contiguous-Tensors.md)
*   [FP32 / FP16 / BF16 / FP8](./Lecture2-FP32-FP16-BF16-FP8.md)
*   [Autograd](./Lecture2-Autograd.md)