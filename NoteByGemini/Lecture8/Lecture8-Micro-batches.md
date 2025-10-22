# 专题笔记：微批次 (Micro-batches)

### 1. 定义
在 **[流水线并行](./Lecture8-Pipeline-Parallelism.md)** 及梯度累积（Gradient Accumulation）技术中，**微批次 (Micro-batch)** 指的是将单次参数更新所对应的全局批次（Global Batch）进一步拆分成的更小的数据块。

### 2. 作用与原理
*   **减少流水线气泡**：在流水线并行中，如果一次处理整个批次，后续的 GPU 必须等待前序 GPU 完成所有计算。通过将大批次拆分为微批次并依次注入流水线，前序 GPU 在将第一个微批次的结果发送给后续 GPU 后，可以立即开始处理第二个微批次，从而使多个 GPU 能够同时工作，显著减小 **[流水线气泡](./Lecture8-Pipeline-Parallelism.md#流水线气泡-pipeline-bubbles)** 区域。
*   **降低显存峰值**：在单 GPU 训练中，如果显存不足以容纳目标批次大小，可以将大批次拆分为微批次，依次进行前向和反向传播并累积梯度，最后进行一次参数更新。这称为梯度累积。

### 3. 代码体现
在课程代码 **[`pipeline_parallelism_main`](./Lecture8-Code-PipelineParallelism.md)** 中：
```python
# 将输入数据切分为多个微批次
micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
# 循环处理每个微批次
for x in micro_batches:
    # 接收 -> 计算 -> 发送
    ...```
这种循环处理机制是实现流水线并行的基础。