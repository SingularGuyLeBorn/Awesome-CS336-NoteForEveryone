# 专题笔记：分布式训练 (Distributed Training)

### 1. 定义与背景
**分布式训练 (Distributed Training)** 是指利用多个计算设备（如 GPU 或 TPU），甚至跨越多个物理节点（Node），协同训练单个机器学习模型的过程。其核心驱动力主要来自两方面：
1.  **模型规模**：现代大模型（如 LLM）的参数量已远超单个 GPU 的显存容量。
2.  **数据规模**：训练数据集极其庞大，单 GPU 训练所需的时间不可接受。

### 2. 核心挑战
分布式训练的本质是在不同的计算单元之间编排计算和数据流动。主要瓶颈通常从计算转移到了**通信（数据传输）**。为了保持较高的算术强度（Arithmetic Intensity）并充分利用 GPU 的计算能力，必须最小化跨设备的数据移动。

### 3. 并行策略分类
根据切分维度的不同，主要的并行策略包括：
*   **[数据并行 (Data Parallelism)](./Lecture8-Data-Parallelism.md)**：沿批次（Batch）维度切分数据，复制模型。
*   **模型并行 (Model Parallelism)**：切分模型本身。
    *   **[张量并行 (Tensor Parallelism)](./Lecture8-Tensor-Parallelism.md)**：沿每一层的宽度（张量维度）切分。
    *   **[流水线并行 (Pipeline Parallelism)](./Lecture8-Pipeline-Parallelism.md)**：沿模型的深度（层数）切分。
*   **完全分片数据并行 (FSDP)**：结合了数据并行和模型分片的思想，将参数、梯度和优化器状态分片存储。

### 4. 编程模型
现代分布式训练通常采用 **[SPMD (单程序多数据)](./Lecture8-SPMD.md)** 模式，依赖底层的 **[集合通信操作](./Lecture8-Collective-Operations.md)**（如 **[All-Reduce](./Lecture8-Collective-Operations.md#all-reduce)**）来同步各个设备的状态。