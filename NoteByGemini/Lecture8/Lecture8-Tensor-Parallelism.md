# 专题笔记：张量并行 (Tensor Parallelism, TP)

### 1. 核心思想
当模型参数量过大，单个 GPU 显存无法容纳时，需要对模型本身进行切分。**张量并行 (TP)** 是一种细粒度的模型并行，它将单个运算（如大矩阵乘法）分散到多个 GPU 上执行。

以 MLP 层 $Y = \text{GeLU}(X \cdot W)$ 为例，TP 通常将权重矩阵 $W$ 沿行或列切分。

### 2. 列切分 (Column Parallelism)
将权重矩阵 $W$ 沿列切分为 $[W_1, W_2]$。输入 $X$ 广播到所有 GPU。
*   GPU 1 计算 $Y_1 = X \cdot W_1$
*   GPU 2 计算 $Y_2 = X \cdot W_2$
*   最终输出为 $Y = [Y_1, Y_2]$（拼接）。

### 3. 行切分 (Row Parallelism)
将权重矩阵 $W$ 沿行切分为 $\begin{bmatrix} W_1 \\ W_2 \end{bmatrix}$。输入 $X$ 也需要沿列切分为 $[X_1, X_2]$。
*   GPU 1 计算 $Y_1 = X_1 \cdot W_1$
*   GPU 2 计算 $Y_2 = X_2 \cdot W_2$
*   最终输出为 $Y = Y_1 + Y_2$（求和）。

### 4. Megatron-LM 风格的 TP
在 Transformer 的 FFN 中，通常组合使用列切分（第一个线性层）和行切分（第二个线性层），以减少通信次数。

在我们的简化实现 **[`tensor_parallelism_main`](./Lecture8-Code-TensorParallelism.md)** 中，演示了在前向传播中，每个 Rank 计算出部分的激活值后，必须通过 **[All-Gather](./Lecture8-Collective-Operations.md)** 操作在所有 Rank 间同步，重建完整的激活输入供下一层使用。

### 5. 通信特性
*   **高频通信**：在网络的每一层（前向和反向）都需要进行集合通信（All-Gather 或 All-Reduce）。
*   **高带宽要求**：传输的是激活值（Activations）或梯度，数据量通常很大。
*   **部署限制**：由于极高的通信带宽和超低延迟要求，TP 通常仅限于通过 **NVLink** 连接的单个节点内的 GPU 之间。