# 代码深度解析：算术强度计算 (Arithmetic Intensity Calculation)

### 1. 核心功能与目标 (Core Function & Goal)
本代码模块旨在通过符号计算（Symbolic Computation）精确模拟 Transformer 模型中全连接层（MLP）和注意力层（Attention）的算术强度。它证明了推理阶段（特别是生成阶段）为何本质上是内存受限的。

### 2. 参数解析 (Parameters)
该函数使用 SymPy 符号变量，而非具体数值：
*   `B`: Batch Size (批次大小)
*   `S`: Sequence Length (输入序列长度/Prompt长度)
*   `T`: Target Sequence Length (生成序列长度)
*   `D`: Model Dimension (隐藏层维度)
*   `F`: MLP Hidden Dimension (前馈网络维度)
*   `memory_bandwidth`: 硬件显存带宽

### 3. 核心逻辑 (Core Logic)

```python
def review_of_arithmetic_intensity():
    text("Setup: multiply X (B x D) and W (D x F) matrix")
    # ... (省略初始化代码)

    # 模拟矩阵乘法: X(B, D) @ W(D, F) -> Y(B, F)
    # 1. 计算 FLOPs (浮点运算次数)
    # 每个输出元素需要 D 次乘法和 D 次加法 -> 2*D
    # 总元素数 B*F -> 总 FLOPs = 2 * B * D * F
    flops = 2*B*D*F

    # 2. 计算内存传输量 (Bytes Transferred)
    # 读取输入 X: B*D (假设bf16, 需乘2, 这里简化为元素数)
    # 读取权重 W: D*F
    # 写入输出 Y: B*F
    # 系数2代表每个元素2字节(bf16)
    bytes_transferred = 2*B*D + 2*D*F + 2*B*F

    # 3. 计算算术强度 (Intensity)
    intensity = (flops / bytes_transferred).simplify()

    # 4. 取极限分析 (Limit Analysis)
    # 假设模型维度 D 和 F 远大于 Batch Size B
    # 即 D = c*B, F = c*B, 当 c -> 无穷大时
    intensity = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify()

    # 结果惊人地简单: 对于矩阵乘法，强度 = Batch Size (B)
    assert intensity == B

    # ... (计算硬件加速器强度 H100约为295)

    # 结论: 如果 B < 295 (例如生成阶段 B=1), 则为内存受限
    text("Conclusion: compute-limited iff B > 295")

def arithmetic_intensity_of_inference():
    # ... (MLP层的分析同上) ...

    # 重点: Attention 层的分析
    # Attention 即使 Batch Size 很大，每个序列也必须读取自己独立的 KV Cache
    # Q @ K^T 计算
    flops = 4*B*S*T*D
    # 读取 Q, K, V 和写入输出
    bytes_transferred = 4*B*S*D + 4*B*T*D

    intensity = (flops / bytes_transferred).simplify()

    # 生成阶段: T=1 (每次生成一个token)
    generate_intensity = intensity.subs(T, 1).simplify()

    # 结果: 强度 < 1。这比 MLP 更糟糕，因为 MLP 还可以通过增大 B 来救
    # 但 Attention 无法通过增大 Batch Size 提升强度，因为 KV Cache 不共享。
    assert generate_intensity < 1
```

### 4. 与理论的连接 (Connection to Theory)
*   **矩阵-向量乘法瓶颈**：代码通过 `limit` 运算展示了当 $B=1$ 时，MLP层的算术强度退化为 1。这直接印证了 **[算术强度理论](./Lecture10-Theory-ArithmeticIntensity.md)** 中关于生成阶段是内存受限的结论。
*   **Attention 的特殊性**：代码特别指出 `Attention` 层的强度计算中，Batch Size $B$ 在分子分母中被约掉了。这意味着对于Attention层，增大 Batch Size 并不能缓解内存带宽压力（因为每个请求都有自己独立的KV Cache），这引出了对 **[GQA](./Lecture10-Code-Stats.md)** 和 **PagedAttention** 等技术的迫切需求。