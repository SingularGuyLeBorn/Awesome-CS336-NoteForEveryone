# 专题笔记：SPMD (单程序多数据)

### 1. 定义
**SPMD (Single Program, Multiple Data)** 是一种并行编程模型。在这种模式下，多个处理器或进程同时执行同一个程序的副本，但每个副本操作不同的数据集。

### 2. 工作机制
*   **统一代码**：所有参与并行的进程运行完全相同的代码二进制文件。
*   **唯一标识**：每个进程通过一个唯一的 ID（通常称为 **Rank**）来区分自己。
*   **数据分片**：程序逻辑利用 Rank 来决定当前进程应该处理哪一部分数据，或者在分布式模型中负责哪一部分参数。例如：
    ```python
    # 伪代码示例
    my_rank = get_rank()
    data_slice = load_data(start_index = my_rank * slice_size)
    process(data_slice)
    ```
*   **协同通信**：进程间通过插入特定的通信原语（如 **[All-Reduce](./Lecture8-Collective-Operations.md#all-reduce)**）来进行同步和数据交换。

### 3. 在分布式深度学习中的应用
PyTorch 的分布式训练主要采用 SPMD 模型。例如，使用 `torch.multiprocessing.spawn` 启动多个进程，每个进程运行相同的训练函数（如课程代码中的 **[`data_parallelism_main`](./Lecture8-Code-DataParallelism.md)**），依靠传递给函数的 `rank` 参数来控制行为（如加载哪部分数据、初始化哪部分模型参数）。