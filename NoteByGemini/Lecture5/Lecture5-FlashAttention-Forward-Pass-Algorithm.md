### 1. 核心功能与目标

**FlashAttention 前向传播算法**的目标是在一个单一的、融合的 CUDA 核函数中, 高效地计算精确的注意力输出 `O = softmax(QKᵀ)V`. 它通过结合**[分块 (Tiling)](./Lecture5-Tiling.md)** 和 **[在线 Softmax](./Lecture5-Online-Softmax-Algorithm.md)** 技术, 实现了对 **[GPU 内存层级](./Lecture5-GPU-Memory-Hierarchy.md)** 的极致优化, 其核心是避免在慢速的全局内存(HBM)中创建和存储巨大的 `N x N` 注意力得分矩阵 `S = QKᵀ`. 

### 2. 参数解析

- `Q, K, V`: 输入矩阵(Query, Key, Value), 维度分别为 `N x d`, 存储在全局 HBM 中. 
- `O`: 输出矩阵, 维度为 `N x d`, 最终结果将写入全局 HBM. 
- `B_c, B_r`: 分块大小, 分别控制 K/V 的列块大小(通常与 d 相同或更小)和 Q 的行块大小. 这是一个重要的调优参数. 

### 3. 核心逻辑 (伪代码与注释)

该算法由一个 CUDA 核函数实现. 每个**[线程块 (Block)](./Lecture5-GPU-Execution-Model.md)** 被分配计算输出矩阵 `O` 的一个行块 `O_i`. 

```cpp
// --- 初始化 ---
// 每个线程块负责计算输出 O 的一个行块 O_i
// O_i 初始化为 0, 维度为 B_r x d
// l_i (累积指数和) 初始化为 0
// m_i (累积最大值) 初始化为 -infinity

// --- 外层循环 (Tr):遍历 K 和 V 的块 ---
// 将 K 和 V 沿行维度(序列长度)切分成 T_c = N / B_c 个块
for j = 1 to T_c:
    // --- 步骤 1: 从 HBM 加载 K_j 和 V_j 到 SRAM ---
    // 线程块内的所有线程协同工作, 将 K 的第 j 个块 K_j (B_c x d)
    // 和 V 的第 j 个块 V_j (B_c x d) 从全局内存加载到 SM 的快速 SRAM (共享内存) 中. 
    // 这个加载过程必须是内存合并的. 
    Load K_j, V_j from HBM to SRAM;
    __syncthreads(); // 同步, 确保加载完成

    // --- 步骤 2: 计算分块得分矩阵 S_ij ---
    // 在 SRAM 中计算当前 Q 块 Q_i 和 K 块 K_j 的点积. 
    // S_ij = Q_i * K_j^T  (维度为 B_r x B_c)
    // 这个计算在寄存器中高效完成. 

    // --- 步骤 3: 在线 Softmax - 局部计算 ---
    // 计算当前块 S_ij 的局部最大值 m_ij 和局部指数和 l_ij
    m_ij = row_max(S_ij);
    P_ij = exp(S_ij - m_ij); // 减去最大值以保证数值稳定
    l_ij = row_sum(P_ij);

    // --- 步骤 4: 在线 Softmax - 全局更新 ---
    // 使用新的局部统计量 (m_ij, l_ij) 更新累积的全局统计量 (m_i, l_i)
    // 这是在线 Softmax 的核心伸缩技巧
    m_i_new = max(m_i, m_ij);
    l_i_new = exp(m_i - m_i_new) * l_i + exp(m_ij - m_i_new) * l_ij;

    // --- 步骤 5: 更新输出 O_i ---
    // 使用伸缩因子校正旧的输出, 并加上当前块的贡献
    O_i = (l_i / l_i_new) * exp(m_i - m_i_new) * O_i + (1 / l_i_new) * P_ij * V_j;

    // 更新累积统计量, 为下一次循环做准备
    l_i = l_i_new;
    m_i = m_i_new;

    __syncthreads(); // 同步, 确保 SRAM 可以被下一轮循环安全覆盖

// --- 循环结束后 ---
// --- 步骤 6: 将最终结果写回 HBM ---
// 将计算完成的输出块 O_i 写回到全局内存. 
Write O_i to HBM;
```

### 4. 与理论的连接

- **[算子融合](./Lecture5-Operator-Fusion.md)**: 整个算法被封装在一个 CUDA 核函数中, 将矩阵乘法、Softmax 计算和与 V 的乘法等多个操作完全融合, 避免了任何中间结果写回 HBM. 
- **[分块技术](./Lecture5-Tiling.md)**: 算法的核心是基于分块的. 数据被分成小块在 SRAM 中处理, 极大地提高了数据重用性, 将算法从内存密集型转变为计算密集型. 
- **[在线 Softmax 算法](./Lecture5-Online-Softmax-Algorithm.md)**: 这是实现精确、分块 Softmax 计算的数学基础. 通过维护和更新 `m_i` 和 `l_i`, 算法能够在不访问全局数据的情况下, 逐步构建出正确的 Softmax 输出. 
- **[GPU 内存层级](./Lecture5-GPU-Memory-Hierarchy.md)**: 算法显式地管理了 HBM(全局内存)和 SRAM(共享内存)之间的数据流动. 频繁访问的数据(`K_j`, `V_j`)被缓存在 SRAM 中, 而累加的输出 `O_i` 则保存在最快的寄存器中, 直到最后才写回 HBM. 