### 1. 概念定义

**在线 Softmax (Online Softmax)** 是一种以流式 (streaming) 方式计算 Softmax 函数的算法。与需要一次性获取所有输入才能进行计算的标准（或称批量）Softmax 不同，在线 Softmax 可以在只看到部分输入的情况下，逐步地、增量地计算并更新结果，最终得到与标准算法完全相同的精确值。

这个算法是 **[FlashAttention](./Lecture5-FlashAttention.md)** 实现其 I/O 感知分块计算的核心数学技巧。

### 2. 标准 Softmax 的挑战

标准 Softmax 的公式为 `y_i = exp(x_i) / Σ_j exp(x_j)`。为了数值稳定性，通常会减去输入中的最大值：`y_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))`。

这个公式存在一个内在的依赖：
- **全局依赖性**: 计算任何一个输出 `y_i` 都需要知道**所有**输入 `x_j`，因为分母 `Σ_j exp(x_j - max(x))` 是对所有输入的求和，并且 `max(x)` 也是全局最大值。

这种全局依赖性使得在**[分块 (Tiling)](./Lecture5-Tiling.md)** 计算中直接应用 Softmax 变得不可能，因为在处理第一个数据块时，我们并不知道后续数据块中的值，也就无法确定全局最大值和全局总和。

### 3. 在线 Softmax 的核心思想

在线 Softmax 通过巧妙地维护和更新两个关键的统计量，解决了这个问题。假设我们正在逐块处理输入向量 `x`。当我们处理到第 `i` 个块时，我们有：

1.  **当前块的最大值 `m_i`** 和 **指数和 `l_i`**:
    - `m_i = max(x_i)` (当前块内的最大值)
    - `l_i = Σ_j exp(x_j - m_i)` (当前块内，以 `m_i` 归一化后的指数和)

2.  **到目前为止的全局统计量 `m_old`** 和 **`l_old`**:
    - 这是处理完前 `i-1` 个块后得到的累积最大值和累积指数和。

现在，我们需要将当前块的信息合并进去，以得到新的全局统计量 `m_new` 和 `l_new`：

1.  **更新全局最大值**:
    - `m_new = max(m_old, m_i)`

2.  **更新全局指数和 (核心技巧)**:
    - 之前计算的 `l_old` 是基于旧的最大值 `m_old` 的，而新计算的 `l_i` 是基于当前块最大值 `m_i` 的。现在我们需要将它们统一到新的全局最大值 `m_new` 下。
    - 我们可以通过乘以一个**伸缩因子 (rescaling factor)** 来实现：
    - `l_new = l_old * exp(m_old - m_new) + l_i * exp(m_i - m_new)`
    - **解释**:
        - `l_old * exp(m_old - m_new)`: 将旧的指数和从 `m_old` 的基准调整到 `m_new` 的基准。如果 `m_new > m_old`，这个因子会缩小 `l_old` 的贡献。
        - `l_i * exp(m_i - m_new)`: 将当前块的指数和从 `m_i` 的基准调整到 `m_new` 的基准。

通过这个递推公式，我们可以逐块地、准确地计算出全局的 Softmax 归一化分母，而无需一次性看到所有数据。

### 4. 在 FlashAttention 中的应用

在 **[FlashAttention](./Lecture5-FlashAttention.md)** 中，这个思想被应用到注意力输出的计算上。当处理 Q 的一个块和 K 的第 `i` 个块时：

- 计算出一个局部的注意力输出 `O_i`。
- 同时计算出局部的 Softmax 统计量 `m_i` 和 `l_i`。
- 使用上述的伸缩技巧，用 `(m_i, l_i)` 来更新全局的输出 `O_new` 和全局统计量 `(m_new, l_new)`。
    - `O_new = (O_old * l_old * exp(m_old - m_new) + O_i * l_i * exp(m_i - m_new)) / l_new`

这样，FlashAttention 可以在一个融合的核函数中，一边进行**[分块矩阵乘法](./Lecture5-Tiled-Matrix-Multiplication-Algorithm.md)**，一边以流式方式完成精确的 Softmax 计算，从而避免了对 HBM 的中间结果读写。