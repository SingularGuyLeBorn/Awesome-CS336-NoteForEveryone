# ZeRO Stage 3 (FSDP) 的执行机制与通信重叠
# (Mechanisms of ZeRO-3/FSDP & Communication Overlap)

## 1. 引言

ZeRO Stage 3, 在 PyTorch 中被实现为全分片数据并行 (**FSDP, Fully Sharded Data Parallel**), 是当今训练超大模型的核心技术. 它打破了数据并行的内存墙, 将参数、梯度和优化器状态全部切分到各个 GPU 上. 

然而, FSDP 的真正难点不在于“切分”, 而在于**在切分的状态下高效地进行计算**. 如果 GPU 在每次计算前都要空等参数传输, 训练速度将无法接受. 本笔记深入解析 FSDP 的动态参数生命周期, 以及如何通过精妙的“通信与计算重叠”来掩盖巨大的通信开销. 

## 2. FSDP 的核心约束与数据状态

在 FSDP 中, 对于模型的任意一层(例如 Transformer Block $L_i$):
*   **静态状态 (Sharded)**:每个 GPU $k$ 只持久存储该层参数的一个分片 $P_{i,k}$, 以及对应的优化器状态 $O_{i,k}$. 
*   **计算需求 (Full)**:要执行 $L_i$ 的前向 (FWD) 或反向 (BWD) 计算, GPU 必须暂时拥有该层完整的参数 $P_i = \text{concat}(P_{i,0}, \dots, P_{i,N-1})$. 

FSDP 的核心机制就是管理 $P_i$ 从 Sharded 到 Full 再回到 Sharded 的动态过程. 

## 3. 参数生命周期与操作序列 (The Step-by-Step Lifecycle)

让我们追踪一个训练步中, 某一层参数的完整生命周期. 这比讲义中的简化描述要复杂得多. 

### 3.1 前向传播 (Forward Pass)

当计算进行到第 $i$ 层时:
1.  **触发 All-Gather (通信)**:当前 GPU 需要完整的 $P_i$. 它发起一个 All-Gather 操作, 广播自己持有的 $P_{i,k}$ 并收集其他 GPU 的分片. 
    *   *状态变化*:参数从分片状态变为在显存中构建出的完整临时状态. 
2.  **执行计算 (Compute)**:利用收集到的完整 $P_i$ 和上一层的激活值 $A_{i-1}$, 计算 $A_i = f(A_{i-1}, P_i)$. 
3.  **立即释放 (Free Memory)**:计算完成后, 完整的 $P_i$ **立即被释放**, 只保留本地原有的分片 $P_{i,k}$. 
    *   *关键点*:必须释放以腾出显存给第 $i+1$ 层的 All-Gather 使用. 这导致了**重新物化 (Rematerialization)** 的需求——反向传播时必须再次收集参数. 

### 3.2 反向传播 (Backward Pass)

当反向传播回到第 $i$ 层时:
1.  **再次触发 All-Gather (通信)**:为了计算梯度, 再次需要完整的 $P_i$. 发起 All-Gather. 
    *   *成本*:这是 FSDP 相比 DDP 增加的额外 $1\times$ 通信成本. 
2.  **执行计算 (Compute)**:利用 $P_i$、保存的激活值和上游传来的梯度, 计算对输入的梯度 $\frac{\partial L}{\partial A_{i-1}}$ 和对参数的梯度 $G_i$. 
3.  **立即释放参数 (Free Memory)**:完整的 $P_i$ 再次被立即释放. 
4.  **触发 Reduce-Scatter (通信)**:此时每个 GPU 都有了关于完整 $P_i$ 的一份**局部**梯度 $G_i$. 需要对其求和. 但 GPU $k$ 只需要其负责的那部分参数的梯度和. 因此, 执行 Reduce-Scatter. 
    *   *状态变化*:完整的梯度 $G_i$ 被规约并切分, GPU $k$ 获得最终的梯度分片 $G_{i,k}$. 
5.  **参数更新 (Optimizer Step)**:GPU $k$ 利用本地持有的 $G_{i,k}$ 和 $O_{i,k}$ 更新本地参数分片 $P_{i,k}$. 

**总通信成本分析**:
*   前向 All-Gather ($1M$) + 反向 All-Gather ($1M$) + 反向 Reduce-Scatter ($1M$) = **$3M$ (3倍参数量)**. 
*   相比之下, DDP 只有一次 All-Reduce = $2M$. FSDP 增加了 50% 的通信量, 换取了线性内存扩展. 

---

## 4. 通信与计算重叠 (Overlap) 的艺术

如果上述步骤是严格串行的(Gather -> Compute -> Free -> Gather next), GPU 将在大部分时间处于空闲等待网络状态. FSDP 的高效依赖于 CPU 在后台提前调度通信流. 

**[插入图片: PDF第23页, 展示FSDP前向和反向传播中通信流(All-Gather/Reduce-Scatter)与计算流重叠的详细时序图, 描述: CPU提前分派通信任务, 使得 GPU 在计算当前层(如 FWD1)时, 网络已经在预取下一层(AG2)的参数. ]**

### 深入解析重叠图 (Decoding the Overlap Diagram)

参考讲义中的图示, 我们可以看到两条并行的流(Streams):**GPU 计算流** 和 **GPU 通信流**. 

1.  **预取 (Prefetching) - 前向阶段**:
    *   当 GPU 计算流正在疯狂进行第 $i$ 层 (`FWDi`) 的矩阵乘法时. 
    *   CPU 并没有闲着, 它已经向 GPU 通信流发出了第 $i+1$ 层参数的 `All-Gather (AG i+1)` 指令. 
    *   **理想状态**:当 `FWDi` 计算完成, 准备进入 `FWDi+1` 时, `AG i+1` 恰好完成, 数据已就绪. 计算无缝衔接, 通信时间被完全掩盖在计算时间内. 
    *   这要求模型的计算密度足够大(大批次、大 hidden size), 使得计算时间 $T_{comp} \ge T_{comm}$. 

2.  **复杂的反向阶段**:
    *   反向阶段更拥挤. 在计算第 $i$ 层的反向 (`BWDi`) 时, 我们需要:
        *   **预取上一层参数**:为 $i-1$ 层启动 `All-Gather (AG i-1)`. 
        *   **同步当前层梯度**:$i$ 层计算出的梯度需要尽快启动 `Reduce-Scatter (RSi)`, 以便尽早释放梯度内存并进行优化器更新. 
    *   调度器必须在有限的带宽下, 合理安排 AG 和 RS 的优先级, 确保关键路径不被阻塞. 

### 结论

FSDP 不仅仅是一个内存管理方案, 更是一个复杂的**延迟隐藏系统**. 它通过将原本原子的 All-Reduce 拆解为细粒度的、按需的 Gather 和 Scatter, 并利用深度神经网络层与层之间的执行间隙, 将增加的通信成本“藏”在了计算的阴影之下. 理解这一点, 是理解为何 FSDP 能在大规模集群上保持高 MFU (Model FLOPs Utilization) 的关键. 