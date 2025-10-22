# 专题笔记：流水线并行 (Pipeline Parallelism, PP)

### 1. 核心思想
**流水线并行 (PP)** 是另一种模型并行策略，它将模型的不同层（Layer）分配给不同的 GPU（Stages）。数据像流水线一样依次流过各个 Stage。
例如，4 层模型分给 2 个 GPU：GPU 0 负责 Layer 1-2，GPU 1 负责 Layer 3-4。

### 2. 朴素实现与问题
朴素的实现是传递一个完整的批次。GPU 0 计算完传给 GPU 1，GPU 1 计算时 GPU 0 空闲。反向传播同理。这会导致严重的硬件利用率低下。

### 3. 流水线气泡 (Pipeline Bubbles)
由于数据依赖性，某些 GPU 在等待来自其他 GPU 的数据或梯度时处于空闲状态，这些空闲时间片被称为**流水线气泡**。气泡的大小直接降低了并行的效率。

### 4. 优化：微批次 (Micro-batches)
为了减少气泡，将一个全局批次（Global Batch）切分为多个细小的**[微批次 (Micro-batches)](./Lecture8-Micro-batches.md)**。
*   **工作流**：GPU 0 处理微批次 1，将其传给 GPU 1；GPU 0 立即开始处理微批次 2，同时 GPU 1 开始处理微批次 1。
*   通过这种方式，计算和通信可以重叠，多个 GPU 可以同时处于工作状态处理不同的微批次。
*   需要特定的调度算法（如 GPipe 的全前向-全反向，或 1F1B 调度）来管理微批次的执行顺序。

### 5. 通信特性
*   **点对点通信**：只在相邻的 Pipeline Stage 之间传输边界层的激活值和梯度（使用 `send`/`recv`）。代码示例见 **[`pipeline_parallelism_main`](./Lecture8-Code-PipelineParallelism.md)**。
*   **通信量较小**：相比张量并行，传输的数据量较小（仅切口处的激活值）。
*   **跨节点友好**：由于通信频率较低，可以通过以太网或 InfiniBand 跨节点部署。

### 6. 内存开销
为了进行反向传播，每个 Stage 需要保存所有微批次的中间激活值，这会带来显著的内存开销。通常结合**[激活检查点 (Activation Checkpointing)](./Lecture8-Distributed-Training.md)** 技术来用计算换内存。