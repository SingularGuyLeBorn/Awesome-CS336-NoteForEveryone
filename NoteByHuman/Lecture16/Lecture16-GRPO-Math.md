# 深入探讨: GRPO数学细节

本文是Lecture 16的精英补充笔记，分析GRPO算法的数学细节，包括标准差归一化的潜在问题和Dr. GRPO论文的改进建议。

---

## 一、GRPO回顾

### 1.1 核心公式

GRPO的优势估计:

$$A_i = \frac{R_i - \text{mean}(R_1, ..., R_G)}{\text{std}(R_1, ..., R_G) + \epsilon}$$

其中 $G$ 是每个prompt生成的response数量。

### 1.2 与PPO的关系

| 组件 | PPO | GRPO |
|------|-----|------|
| 基线 | 学习的V(s) | 组内均值 |
| 标准化 | GAE处理 | 组内std归一化 |
| 价值函数 | 需要训练 | 不需要 |

---

## 二、标准差归一化的问题

### 2.1 策略梯度定理回顾

标准的策略梯度定理允许我们**减去**任意只依赖状态的基线:

$$\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot (R - b(s))]$$

**关键**: 只有**减法**是理论保证的，**除法**不在定理范围内。

### 2.2 除法带来的问题

设两个prompt:
- Prompt A：简单，所有response都接近正确，$\text{std}(R) \approx 0.01$
- Prompt B：困难，response质量差异大，$\text{std}(R) \approx 1.0$

归一化后:
- Prompt A的梯度被**放大100倍**
- Prompt B的梯度正常

**结果**: 简单问题获得过大的梯度权重！

### 2.3 数学分析

设 $\delta = R - \bar{R}$，GRPO使用 $\tilde{\delta} = \delta / (\sigma + \epsilon)$。

**问题1**: $\sigma$ 接近0时
$$\tilde{\delta} \approx \frac{\delta}{\epsilon} \quad \text{(被}\epsilon\text{限制，但仍可能很大)}$$

**问题2**: 不满足策略梯度定理
$$\mathbb{E}[\nabla \log \pi \cdot \tilde{\delta}] \neq \mathbb{E}[\nabla \log \pi \cdot \delta] / \text{constant}$$

因为 $\sigma$ 本身依赖于采样的actions。

### 2.4 实验证据

Dr. GRPO论文的实验:

```
设置: 简单排序任务
比较: 
  - GRPO (带std归一化)
  - GRPO-centered (只减均值，不除std)
  - GRPO-raw (直接使用reward)

结果:
  - GRPO-centered 收敛更稳定
  - GRPO 在某些情况下振荡
  - 对于二元reward (0/1)，差别最明显
```

---

## 三、长度归一化的问题

### 3.1 原始GRPO的长度归一化

DeepSeep Math论文的GRPO对loss进行长度归一化:

$$L = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi(y_t | y_{<t}, x) \cdot A$$

### 3.2 问题分析

**场景1**: 答错时 ($A < 0$)
- 更长的response → 每token负梯度更小
- 最优策略: 生成**尽可能长**的错误回答（稀释惩罚）

**场景2**: 答对时 ($A > 0$)
- 更长的response → 每token正梯度更小
- 最优策略: 生成**尽可能短**的正确回答（集中奖励）

**结果**:
- 模型学会长篇废话（错误时）
- 模型学会简短回答（正确时）
- 与期望行为相反

### 3.3 实验证据

R1-Zero实验中观察到:
- 思维链长度持续增长
- 可能不是"深度思考"，而是长度偏差

---

## 四、Dr. GRPO的改进

### 4.1 建议1: 移除标准差归一化

只使用centered rewards:

$$A_i = R_i - \text{mean}(R_1, ..., R_G)$$

**优点**:
- 满足策略梯度定理
- 避免简单prompt的梯度爆炸
- 实现更简单

### 4.2 建议2: 移除长度归一化

直接使用序列级别的loss:

$$L = -\sum_{t=1}^{|y|} \log \pi(y_t | y_{<t}, x) \cdot A$$

或者使用token级别但不归一化:

$$L = -\sum_{t=1}^{|y|} \log \pi(y_t | y_{<t}, x) \cdot A_t$$

### 4.3 建议3: 显式长度奖励

如果需要控制长度，使用显式奖励而非隐式归一化:

```python
def compute_reward(response, ground_truth):
    accuracy = 1.0 if is_correct(response, ground_truth) else 0.0
    
    # 显式长度惩罚（可调）
    length_penalty = -0.001 * len(response)
    
    return accuracy + length_penalty
```

---

## 五、理论视角

### 5.1 基线的本质

基线的目的是**减少方差**，不改变期望:

$$\text{Var}[\nabla \log \pi \cdot (R - b)] < \text{Var}[\nabla \log \pi \cdot R]$$

最优基线:
$$b^*(s) = \frac{\mathbb{E}[(\nabla \log \pi)^2 \cdot R | s]}{\mathbb{E}[(\nabla \log \pi)^2 | s]}$$

### 5.2 GRPO基线的优势

GRPO的组内均值是对 $V(s)$ 的**无偏估计**:

$$\bar{R} = \frac{1}{G} \sum_{i=1}^G R_i \approx \mathbb{E}[R | s]$$

当 $G$ 足够大时，这个估计很准确。

### 5.3 为何不需要价值函数

传统PPO需要价值函数是因为:
1. 只能采样一个trajectory
2. 需要从V(s)估计期望回报

GRPO的特殊结构:
1. 同一个prompt可以采样多个response
2. 组内均值直接估计期望
3. 无需单独的函数近似

---

## 六、实践建议

### 6.1 何时使用标准化

**建议使用**:
- 不同prompt难度差异大
- 需要所有prompt有相似的学习贡献
- reward分布差异大

**建议不使用**:
- 二元reward (0/1)
- prompt难度相近
- 需要理论保证

### 6.2 超参数选择

```python
class GRPOConfig:
    # 组大小
    group_size: int = 8  # 每prompt生成8个response
    
    # 归一化
    use_std_normalization: bool = False  # Dr. GRPO建议
    epsilon: float = 1e-5  # 如果使用std归一化
    
    # 长度
    use_length_normalization: bool = False  # Dr. GRPO建议
    length_penalty: float = 0.0  # 如需显式惩罚
    
    # 其他
    clip_epsilon: float = 0.2
    kl_penalty: float = 0.01
```

### 6.3 调试建议

1. **监控梯度**: 检查不同prompt的梯度幅度是否合理
2. **监控长度**: 检查response长度是否异常增长/缩短
3. **分层分析**: 分别分析简单/困难问题的学习曲线
4. **消融实验**: 对比有/无归一化的效果

---

## 七、代码实现

### 7.1 原始GRPO

```python
def grpo_loss_original(log_probs, rewards, group_size):
    """原始GRPO实现（带std归一化）"""
    batch_size = rewards.shape[0]
    num_groups = batch_size // group_size
    
    # reshape为组
    rewards = rewards.view(num_groups, group_size)
    log_probs = log_probs.view(num_groups, group_size, -1)
    
    # 组内归一化
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-5)
    
    # 长度归一化的loss
    seq_lengths = (log_probs != 0).sum(dim=-1)
    normalized_log_probs = log_probs.sum(dim=-1) / seq_lengths
    
    loss = -(normalized_log_probs * advantages).mean()
    return loss
```

### 7.2 Dr. GRPO

```python
def grpo_loss_dr(log_probs, rewards, group_size):
    """Dr. GRPO实现（移除归一化）"""
    batch_size = rewards.shape[0]
    num_groups = batch_size // group_size
    
    # reshape为组
    rewards = rewards.view(num_groups, group_size)
    log_probs = log_probs.view(num_groups, group_size, -1)
    
    # 只减均值，不除std
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - mean_rewards
    
    # 不做长度归一化
    total_log_probs = log_probs.sum(dim=-1)
    
    loss = -(total_log_probs * advantages).mean()
    return loss
```

### 7.3 带显式长度惩罚

```python
def grpo_loss_with_length_penalty(log_probs, rewards, lengths, 
                                   group_size, length_penalty=0.001):
    """带显式长度惩罚的GRPO"""
    # 在reward中加入长度惩罚
    adjusted_rewards = rewards - length_penalty * lengths
    
    # 使用Dr. GRPO的loss
    return grpo_loss_dr(log_probs, adjusted_rewards, group_size)
```

---

## 八、总结

### 关键结论

1. **标准差归一化不是理论保证的**，在某些情况下可能有害
2. **长度归一化会导致反向激励**（长错误、短正确）
3. **简单的centered rewards往往效果更好**
4. **如需长度控制，使用显式奖励**

### 实践checklist

- [ ] 移除std归一化（或至少做消融实验）
- [ ] 移除长度归一化
- [ ] 监控response长度变化
- [ ] 检查梯度在不同prompt上的分布
- [ ] 考虑显式长度奖励（如果需要）

---

## 参考资料

1. Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning
2. Liu et al. (2025). Dr. GRPO: Understanding R1-Zero-Like Training
3. Schulman et al. (2017). Proximal Policy Optimization Algorithms
4. Greensmith et al. (2004). Variance Reduction Techniques for Gradient Estimates in RL
