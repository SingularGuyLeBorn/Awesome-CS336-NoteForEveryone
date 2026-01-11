# 代码深度解析: CriticalBatchEstimator

### 1. 核心功能与目标 (Core Function & Goal)
`CriticalBatchEstimator` 用于在训练过程中实时估算 **[Critical Batch Size](./Lecture9-Critical-Batch-Size.md)**。这一工具对于系统优化至关重要：它告诉系统组是否可以继续增加数据并行度而不浪费算力。

### 2. 参数解析 (Parameters)
*   `gradients`: 某一步骤中，各个微批次（micro-batches）计算出的梯度向量列表。
*   `simple_noise_scale`: 一种简化的估算方法，即计算梯度方差与梯度的模之比。

### 3. 核心逻辑 (Core Logic)

```python
import torch

class CriticalBatchEstimator:
    """
    基于梯度统计特性估算临界批量大小 (Critical Batch Size)。
    参考: OpenAI 'An Empirical Model of Large-Batch Training'
    """
    def __init__(self):
        pass

    def estimate_noise_scale(self, model):
        """
        在一次反向传播后，计算简单的噪声尺度 (Simple Noise Scale)。
        这通常需要 access 到每个样本的梯度，或者不同 micro-batch 的梯度。
        """
        # 假设我们将一个大 Batch 拆分为多个小 micro-batches 并保留了梯度
        # grads shape: [num_micro_batches, num_params]
        grads = self._collect_micro_gradients(model)

        if grads is None:
            return 0

        # 1. 计算梯度的均值 (即该大 Batch 的估计真实梯度 G)
        G = torch.mean(grads, dim=0)
        G_norm_sq = torch.sum(G**2)

        # 2. 计算梯度的方差 (Trace of Covariance Sigma)
        # S = E[|g|^2] - |E[g]|^2
        E_g_sq = torch.mean(torch.sum(grads**2, dim=1))
        trace_sigma = E_g_sq - G_norm_sq

        # 3. 临界批量大小 B_crit ≈ Trace(Sigma) / |G|^2
        # 防止除零
        if G_norm_sq < 1e-8:
            return float('inf')

        b_crit = trace_sigma / G_norm_sq
        return b_crit.item()

    def _collect_micro_gradients(self, model):
        """
        这是一个辅助函数，用于收集模型当前的梯度。
        在实际分布式训练中，这需要 hook 到 DDP 或梯度累积过程。
        """
        # 伪代码: 返回堆叠的梯度张量
        pass

# --- 理论结合说明 ---
# 随着训练进行，Loss 变小，G (梯度模长) 通常会变小。
# 分母 |G|^2 变小，导致 B_crit (临界批量) 变大。
# 这验证了课堂上提到的 "Lower loss target -> Larger batch size" 现象。
```

### 4. 与理论的连接 (Connection to Theory)
*   **公式实现**: 代码直接实现了 $B_{crit} \approx \frac{\text{Tr}(\Sigma)}{|G|^2}$ 的理论公式。
*   **动态调整**: 该模块展示了为什么 Llama 3 训练报告中提到“随着 Loss 降低增加 Batch Size”。通过运行 `estimate_noise_scale`，我们可以监控到 `b_crit` 随训练步数逐渐上升的趋势。
*   **收益递减**: 如果当前使用的 `actual_batch_size` 远大于计算出的 `b_crit`，工程师应当知道此时增加 GPU 只会带来边际收益递减，应该停止扩大规模或调整学习率。