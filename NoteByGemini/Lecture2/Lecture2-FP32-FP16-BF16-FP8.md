# 专题笔记: FP32 / FP16 / BF16 / FP8 (浮点数精度)

### 1. 核心概念

在深度学习中,**浮点数(Floating-Point Number)** 是表示带有小数的实数的主要方式. 不同的浮点数格式(精度)在内存占用、数值范围和计算速度之间做出了不同的权衡. 理解这些格式是进行模型训练和优化,特别是资源核算的关键. 

一个浮点数通常由三部分组成: 
*   **符号位 (Sign)**: 1位,表示正负. 
*   **指数位 (Exponent)**: 决定了数值可以表示的范围(动态范围). 指数位越多,能表示的数字范围越大. 
*   **小数/尾数位 (Fraction/Mantissa)**: 决定了数值的精度. 小数位越多,能表示的数字越精确. 

### 2. 常见浮点数格式对比

| 格式 | 总位数 | 符号位 | 指数位 | 小数位 | 主要特点 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | 1 | 8 | 23 | **黄金标准**: 高精度,大范围. 传统上用于科学计算和深度学习,稳定但资源消耗大.  |
| **FP16** | 16 | 1 | 5 | 10 | **半精度**: 内存和计算减半. 但指数位太少(只有5位),动态范围非常小,极易在训练大模型时发生上溢(变成无穷大)或下溢(变成0)的问题.  |
| **BF16** | 16 | 1 | 8 | 7 | **Brain Float**: 内存与FP16相同. 关键在于它拥有和FP32一样的**8位指数位**,因此动态范围与FP32相同,有效避免了溢出问题. 代价是牺牲了精度(只有7位小数位). 实践证明,对深度学习而言,动态范围比高精度更重要.  |
| **FP8** | 8 | 1 | 4/5 | 3/2 | **前沿技术**: 在 **[NVIDIA H100](./Lecture2-NVIDIA-H100.md)** 中引入,追求极致性能. 它有两种变体(E4M3 和 E5M2),分别在动态范围和精度之间做出不同取舍. 主要通过 **[Transformer引擎](./Lecture2-NVIDIA-H100.md)** 与更高精度格式配合使用.  |

**一句话总结: **
*   **FP32**: 安全但昂贵. 
*   **FP16**: 快速但危险(易溢出). 
*   **BF16**: **现代大模型训练的甜点 (sweet spot)**,兼具FP32的稳定性和FP16的速度/内存优势. 
*   **FP8**: 追求极致推理和训练速度的未来方向. 

### 3. 为何低精度如此重要？

使用更低精度的浮点数(如从 FP32 降到 BF16)会带来多重好处: 

1.  **内存减半**: 模型参数、梯度和优化器状态占用的内存直接减半. 这意味着在同一张 GPU 上可以容纳两倍大的模型,或者使用两倍大的批处理大小(batch size). 
2.  **计算加速**: 
    *   **数据传输更快**: 更小的数据意味着从 GPU 内存读取到计算核心的速度更快. 
    *   **硬件原生支持**: 现代 GPU(如 A100 和 H100)的 Tensor Cores 专门为低精度计算(如 BF16, TF32, FP8)设计了硬件加速路径,其理论 **[FLOPS](./Lecture2-FLOPS.md)** 远高于 FP32. 
3.  **能效更高**: 完成同样多的计算,消耗的能源更少. 

### 4. BFloat16 (BF16) 的崛起

在 FP16 因为其不稳定性而在大型模型训练中逐渐失宠后,**BFloat16 (Brain Floating Point Format)** 成为了事实上的标准. 

它的设计理念非常巧妙: Google Brain 的研究人员发现,深度学习的梯度和激活值通常具有非常大的数值范围,但对精度的要求并不苛刻. 因此,他们保留了 FP32 的 8 位指数部分,确保了相同的数值动态范围,而将小数部分从 23 位削减到 7 位. 

**代码示例: FP16 vs BF16 的稳定性**
```python
import torch

# 一个在 FP16 中会溢出的数字
large_number = torch.tensor(65504.0, dtype=torch.float32) * 2
print(f"原始数字: {large_number.item()}")

# 转换为 FP16,发生上溢 (inf)
fp16_tensor = large_number.to(torch.float16)
print(f"转换为 FP16: {fp16_tensor.item()}")

# 转换为 BF16,可以正确表示
bf16_tensor = large_number.to(torch.bfloat16)
print(f"转换为 BF16: {bf16_tensor.item()}") # 结果会有精度损失,但不会溢出

print("-" * 20)

# 一个在 FP16 中会下溢的数字
small_number = torch.tensor(1e-5, dtype=torch.float32)
print(f"原始数字: {small_number.item()}")

# 转换为 FP16,发生下溢 (0.0)
fp16_tensor_small = small_number.to(torch.float16)
print(f"转换为 FP16: {fp16_tensor_small.item()}")

# 转换为 BF16,可以表示
bf16_tensor_small = small_number.to(torch.bfloat16)
print(f"转换为 BF16: {bf16_tensor_small.item()}") # 结果有精度损失,但不会变成0
```

这个例子清晰地展示了为何 BF16 在训练需要处理极大或极小数值的现代大型模型时,比 FP16 更加稳定和可靠. 这也是 **[混合精度训练](./Lecture2-Mixed-Precision-Training.md)** 中,推荐使用 BF16 作为计算数据类型的原因. 

---
**关联知识点**
*   [混合精度训练 (Mixed Precision Training)](./Lecture2-Mixed-Precision-Training.md)
*   [NVIDIA H100](./Lecture2-NVIDIA-H100.md)
*   [FLOPS (浮点运算)](./Lecture2-FLOPS.md)