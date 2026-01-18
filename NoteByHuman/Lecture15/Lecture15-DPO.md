# 深入探讨: DPO数学推导

本文是Lecture 15的精英补充笔记，完整推导DPO (Direct Preference Optimization) 的数学原理，解释为何可以绕过显式奖励模型直接从偏好数据优化策略。

---

## 一、RLHF的标准目标

### 1.1 目标函数

在RLHF中，我们希望找到策略 $\pi_\theta$ 最大化:

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) \right] - \beta D_{KL}(\pi_\theta(\cdot|x) || \pi_{ref}(\cdot|x))$$

其中:
- $r(x, y)$: 奖励函数（通常从偏好数据学习）
- $\pi_{ref}$: 参考策略（通常是SFT模型）
- $\beta$: KL惩罚系数

### 1.2 Bradley-Terry偏好模型

假设人类偏好由潜在奖励决定:

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是sigmoid函数。

### 1.3 标准RLHF流程

1. 从偏好数据训练奖励模型 $r_\phi$
2. 使用PPO最大化 $J(\theta)$

**DPO的问题**: 能否跳过步骤1，直接从偏好数据训练策略？

---

## 二、DPO核心推导

### 2.1 非参数最优策略

**关键洞察**: 对于固定的奖励 $r$，我们可以写出最优策略的**解析形式**。

将目标函数写成积分形式:

$$J(\theta) = \int p(x) \left[ \int \pi_\theta(y|x) \left( r(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right) dy \right] dx$$

对于每个 $x$，内部优化是关于分布 $\pi_\theta(\cdot|x)$ 的优化。

### 2.2 变分推导

固定 $x$，对 $\pi = \pi_\theta(\cdot|x)$ 求极值:

$$\max_\pi \mathbb{E}_{y \sim \pi} \left[ r(x,y) - \beta \log \frac{\pi(y)}{\pi_{ref}(y|x)} \right]$$

改写为:

$$\max_\pi \mathbb{E}_{y \sim \pi} \left[ r(x,y) \right] - \beta D_{KL}(\pi || \pi_{ref})$$

这是一个 **带KL正则化的最大化问题**。

### 2.3 最优解

通过变分法或拉格朗日乘子法，可以证明最优分布为:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

其中配分函数:

$$Z(x) = \sum_y \pi_{ref}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

### 2.4 证明

设拉格朗日量:

$$\mathcal{L}(\pi, \lambda) = \mathbb{E}_{y \sim \pi}[r(x,y)] - \beta D_{KL}(\pi || \pi_{ref}) - \lambda \left( \sum_y \pi(y) - 1 \right)$$

对 $\pi(y)$ 求导并令其为0:

$$\frac{\partial \mathcal{L}}{\partial \pi(y)} = r(x,y) - \beta \log \frac{\pi(y)}{\pi_{ref}(y|x)} - \beta - \lambda = 0$$

解得:

$$\pi(y) = \pi_{ref}(y|x) \exp\left( \frac{r(x,y) - \lambda - \beta}{\beta} \right)$$

归一化后得到上述最优解形式。

---

## 三、从策略反推奖励

### 3.1 关键转换

从最优策略公式 $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left( \frac{r(x,y)}{\beta} \right)$

取对数并整理:

$$\log \pi^*(y|x) = \log \pi_{ref}(y|x) + \frac{r(x,y)}{\beta} - \log Z(x)$$

解出奖励:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

### 3.2 奖励的重参数化

**核心发现**: 奖励可以用策略的对数比率表示（加上一个只依赖于 $x$ 的项）。

$$r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

对于最优策略 $\pi = \pi^*$，这个等式成立。

---

## 四、代入Bradley-Terry

### 4.1 偏好概率

将重参数化的奖励代入Bradley-Terry模型:

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))$$

代入 $r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$:

$$P(y_1 \succ y_2 | x) = \sigma\left( \beta \log \frac{\pi(y_1|x)}{\pi_{ref}(y_1|x)} + \cancel{\beta \log Z(x)} - \beta \log \frac{\pi(y_2|x)}{\pi_{ref}(y_2|x)} - \cancel{\beta \log Z(x)} \right)$$

### 4.2 $Z(x)$相消

注意 $\beta \log Z(x)$ 项相消！

$$P(y_1 \succ y_2 | x) = \sigma\left( \beta \log \frac{\pi(y_1|x)}{\pi_{ref}(y_1|x)} - \beta \log \frac{\pi(y_2|x)}{\pi_{ref}(y_2|x)} \right)$$

**这意味着**: 偏好概率完全由策略与参考策略的比率决定，不需要配分函数！

---

## 五、DPO损失函数

### 5.1 最大似然估计

给定偏好数据集 $\mathcal{D} = \{(x, y_w, y_l)\}$（$y_w$是偏好的response），最大化似然:

$$\max_\theta \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log P_\theta(y_w \succ y_l | x) \right]$$

### 5.2 DPO损失

代入偏好概率公式:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

### 5.3 简化记号

定义隐式奖励:

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

则:

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E} \left[ \log \sigma(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)) \right]$$

**这与奖励模型训练的形式完全相同**！

---

## 六、DPO梯度分析

### 6.1 梯度求导

对 $\mathcal{L}_{DPO}$ 求关于 $\theta$ 的梯度:

$$\nabla_\theta \mathcal{L}_{DPO} = -\mathbb{E} \left[ \sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w)) \cdot \beta \cdot \left( \nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x) \right) \right]$$

### 6.2 梯度解释

$$\nabla_\theta \mathcal{L}_{DPO} \propto -\underbrace{w(\theta)}_{\text{自适应权重}} \cdot \left( \underbrace{\nabla \log \pi_\theta(y_w)}_{\text{提高 } y_w} - \underbrace{\nabla \log \pi_\theta(y_l)}_{\text{降低 } y_l} \right)$$

其中 $w(\theta) = \sigma(\hat{r}_\theta(y_l) - \hat{r}_\theta(y_w))$。

**权重的含义**:
- 当 $\hat{r}_\theta(y_l) > \hat{r}_\theta(y_w)$（模型认为 $y_l$ 更好）时，$w$ 较大
- 此时梯度更强，更积极地纠正错误判断
- 这是一种**隐式的困难样本挖掘**

### 6.3 与策略梯度的联系

DPO梯度可以看作一种**对比策略梯度**:
- 正样本 $y_w$ 向上推
- 负样本 $y_l$ 向下推
- 权重根据当前模型的错误程度动态调整

---

## 七、DPO vs RLHF对比

### 7.1 流程对比

| 步骤 | RLHF | DPO |
|------|------|-----|
| 1. SFT | ✓ | ✓ |
| 2. 奖励模型 | 训练单独的RM | ❌ |
| 3. 采样 | 需要在线采样 | ❌ |
| 4. 优化 | PPO（复杂） | 监督学习（简单） |

### 7.2 理论等价性

在以下假设下，DPO与RLHF等价:
1. Bradley-Terry偏好模型成立
2. 策略类足够灵活（非参数假设）
3. 配分函数可以被吸收

### 7.3 实际差异

| 方面 | RLHF | DPO |
|------|------|-----|
| 实现复杂度 | 高 | 低 |
| 在线采样 | 需要 | 不需要 |
| 内存需求 | 高（多模型） | 低 |
| 探索能力 | 有 | 无 |
| 稳定性 | 需要调参 | 较稳定 |

---

## 八、DPO的局限与变体

### 8.1 离线的局限

DPO是离线算法：
- 不从当前策略采样
- 可能无法发现新的好回复
- 分布偏移问题

### 8.2 IPO (Identity Preference Optimization)

解决DPO的过拟合问题:

$$\mathcal{L}_{IPO} = \mathbb{E}\left[ \left( \hat{r}_\theta(y_w) - \hat{r}_\theta(y_l) - \frac{1}{\beta} \right)^2 \right]$$

### 8.3 SimPO

去除参考模型，添加长度归一化:

$$\mathcal{L}_{SimPO} = -\log \sigma\left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l) - \gamma \right)$$

### 8.4 KTO (Kahneman-Tversky Optimization)

只需要单个回复的好/坏标签，不需要成对比较:

$$\mathcal{L}_{KTO} = \mathbb{E}_{y_w}[1 - \sigma(\hat{r}_\theta(y_w))] + \mathbb{E}_{y_l}[\sigma(\hat{r}_\theta(y_l))]$$

---

## 九、代码实现

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_logprobs_w, policy_logprobs_l, 
             ref_logprobs_w, ref_logprobs_l, beta=0.1):
    """
    计算DPO损失
    
    Args:
        policy_logprobs_w: [B] 当前策略对winner的log概率
        policy_logprobs_l: [B] 当前策略对loser的log概率
        ref_logprobs_w: [B] 参考策略对winner的log概率
        ref_logprobs_l: [B] 参考策略对loser的log概率
        beta: KL正则化系数
    
    Returns:
        loss: 标量
    """
    # 计算隐式奖励
    implicit_reward_w = beta * (policy_logprobs_w - ref_logprobs_w)
    implicit_reward_l = beta * (policy_logprobs_l - ref_logprobs_l)
    
    # DPO损失 = -log(sigmoid(r_w - r_l))
    loss = -F.logsigmoid(implicit_reward_w - implicit_reward_l).mean()
    
    return loss

# 使用示例
# 假设已计算好各log概率
loss = dpo_loss(
    policy_logprobs_w=torch.tensor([-2.5, -3.0, -2.8]),
    policy_logprobs_l=torch.tensor([-3.5, -4.0, -3.8]),
    ref_logprobs_w=torch.tensor([-2.8, -3.2, -3.0]),
    ref_logprobs_l=torch.tensor([-3.2, -3.8, -3.5]),
    beta=0.1
)
```

---

## 参考资料

1. Rafailov, R. et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model
2. Azar, M. G. et al. (2023). A General Theoretical Paradigm to Understand Learning from Human Preferences
3. Ethayarajh, K. et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization
4. Meng, Y. et al. (2024). SimPO: Simple Preference Optimization with a Reference-Free Reward
