# CS336 Lecture 17: 手把手讲解GRPO (Hands-on Explanation of GRPO)

> **编辑蓝图 (Editorial Blueprint)**
> 
> **核心主题**: 本讲座是CS336课程RL系列的收官之作，提供了**GRPO算法的完整代码实现**。从策略梯度的数学推导，到基线的直观理解，再到完整的训练循环，配合一个简单的排序任务进行演示。
> 
> **知识结构**: 
> - 第一部分：RL在语言模型中的设定（状态、动作、奖励）
> - 第二部分：策略梯度数学推导
> - 第三部分：基线与优势函数
> - 第四部分：完整GRPO实现与实验
> 
> **精英补充笔记**: 无（本讲座本身就是实现级别的深度讲解）

---

## 一、语言模型RL设定 (RL Setup for Language Models)

### 1.1 基本定义

| 概念 | 语言模型上下文 |
|------|---------------|
| **状态 (State)** $s$ | Prompt + 已生成的response |
| **动作 (Action)** $a$ | 生成下一个token |
| **奖励 (Reward)** $R$ | Response的好坏程度 |
| **策略 (Policy)** $\pi$ | 语言模型 $P_\theta(a|s)$ |
| **轨迹 (Trajectory)** | $s \to a_1 \to a_2 \to ... \to R$ |

### 1.2 本课程聚焦的设定

我们关注**结果奖励 (Outcome Rewards)**：
- 奖励是整个response的函数
- 奖励是**可验证的**（不是学习的）

**示例**: 数学问题
```
Prompt: "2 + 3 * 4 = ?"
Response: "Let me think... 2 + 3 = 5, then 5 * 4 = 20... 
           Wait, order of operations! 3 * 4 = 12, then 2 + 12 = 14.
           Therefore, the answer is 14."
           
Reward Function: Extract "14", compare to ground truth → R = 1
```

### 1.3 语言模型RL的特殊性

**转移动态确定性**: $T(s'|s,a) = \delta(s' = s + a)$

这意味着：
- 可以进行**规划/测试时计算**（机器人做不到）
- 状态是"虚构的"（token序列，而非物理状态）
- 任何状态都可达（只要写出tokens）
- 挑战不是"到达"状态，而是"正确"状态

### 1.4 目标

最大化期望奖励：

$$J(\theta) = \mathbb{E}_{s \sim p(s), a \sim \pi_\theta(a|s)}[R(s, a)]$$

---

## 二、策略梯度 (Policy Gradient)

### 2.1 符号简化

为了简化，令 $a$ 表示**整个response**（而非单个token）。

在结果奖励设定下，这是合理的——可以认为一次性生成整个回复。

### 2.2 梯度推导

期望奖励：
$$J(\theta) = \int p(s) \pi_\theta(a|s) R(s,a) \, ds \, da$$

对 $\theta$ 求梯度：
$$\nabla J(\theta) = \int p(s) \nabla \pi_\theta(a|s) R(s,a) \, ds \, da$$

使用 $\nabla \log f = \frac{\nabla f}{f}$ 变换：
$$\nabla J(\theta) = \int p(s) \pi_\theta(a|s) \nabla \log \pi_\theta(a|s) R(s,a) \, ds \, da$$

写成期望形式：
$$\nabla J(\theta) = \mathbb{E}_{s,a}[\nabla \log \pi_\theta(a|s) \cdot R(s,a)]$$

### 2.3 朴素策略梯度

```python
def naive_policy_gradient(model, prompts, reward_fn, lr):
    """朴素策略梯度一步更新"""
    # 1. 从当前策略采样
    responses = model.generate(prompts)
    
    # 2. 计算奖励
    rewards = [reward_fn(p, r) for p, r in zip(prompts, responses)]
    
    # 3. 计算梯度并更新
    # ∇θ ← E[R · ∇log π_θ(a|s)]
    log_probs = model.log_prob(prompts, responses)
    loss = -(log_probs * torch.tensor(rewards)).mean()
    loss.backward()
    optimizer.step()
```

### 2.4 与SFT的关系

策略梯度 = **奖励加权的SFT**

$$\nabla J \propto R \cdot \nabla \log \pi_\theta(y|x)$$

- 如果 $R > 0$：提高该response的概率
- 如果 $R < 0$：降低该response的概率
- 如果 $R = 0$：不更新

### 2.5 稀疏奖励问题

考虑二元奖励 $R \in \{0, 1\}$：

**问题**: 如果策略很差，大部分response都得到 $R=0$
- 梯度大多为零
- 几乎没有学习信号
- 模型"卡住"

**对比RLHF**: 奖励模型给出连续分数，信号更丰富

---

## 三、基线与方差减少 (Baselines and Variance Reduction)

### 3.1 高方差问题

直观例子：

| 状态 | 动作 | 奖励 |
|------|------|------|
| s1 | a1 | 11 |
| s1 | a2 | 9 |
| s2 | a1 | 0 |
| s2 | a2 | 2 |

- 最优策略: s1→a1, s2→a2
- 但 R(s1,a2)=9 > R(s2,a2)=2
- 单看奖励会误导！

**根本问题**: 不同状态的奖励尺度不同

### 3.2 基线的数学

**关键定理**: 对于任意只依赖状态的函数 $b(s)$：

$$\mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot b(s)] = 0$$

**证明**:
$$\mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot b(s)] = \int p(s) b(s) \left[\int \nabla \pi_\theta(a|s) da\right] ds$$

而 $\int \nabla \pi_\theta(a|s) da = \nabla \int \pi_\theta(a|s) da = \nabla 1 = 0$

**结论**: 可以减去任意 $b(s)$ 而不改变梯度期望：
$$\nabla J = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot (R - b(s))]$$

### 3.3 方差减少效果

回到例子，设 $b(s_1)=10, b(s_2)=1$：

| 状态 | 动作 | 原奖励 | 基线后奖励 |
|------|------|--------|-----------|
| s1 | a1 | 11 | +1 |
| s1 | a2 | 9 | -1 |
| s2 | a1 | 0 | -1 |
| s2 | a2 | 2 | +1 |

```python
import torch

# 原始奖励方差
raw_rewards = torch.tensor([11., 9., 0., 2.])
raw_variance = torch.var(raw_rewards)  # 约22.7

# 基线后方差
baselined_rewards = torch.tensor([1., -1., -1., 1.])
baseline_variance = torch.var(baselined_rewards)  # 约1.3

print(f"方差减少: {raw_variance:.1f} → {baseline_variance:.1f}")
```

### 3.4 最优基线

理论上最优的基线：
$$b^*(s) = \frac{\mathbb{E}[(\nabla \log \pi)^2 \cdot R | s]}{\mathbb{E}[(\nabla \log \pi)^2 | s]}$$

实际中难以计算，常用**启发式**:
$$b(s) \approx \mathbb{E}[R|s] = V(s)$$

这就是**价值函数**的来源！

### 3.5 优势函数 (Advantage Function)

定义：
- **价值函数**: $V(s) = \mathbb{E}[R|s]$
- **Q函数**: $Q(s,a) = \mathbb{E}[R|s,a]$（在结果奖励下 = R）
- **优势函数**: $A(s,a) = Q(s,a) - V(s)$

**直觉**: 优势衡量"动作 $a$ 比平均水平好多少"

使用优势作为更新权重：
$$\nabla J = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot A(s,a)]$$

---

## 四、GRPO实现 (GRPO Implementation)

### 4.1 组结构 (Group Structure)

语言模型的**特殊优势**：对同一个prompt可以生成多个responses！

```python
prompts = ["What is 2+2?"]
responses_per_prompt = [
    ["4", "Let me calculate... 4", "It's 4", "The answer is 4"]
]
```

这些responses形成一个**组**，可以用组内均值作为基线！

### 4.2 GRPO优势计算

$$A_i = \frac{R_i - \text{mean}(R_1, ..., R_G)}{\text{std}(R_1, ..., R_G) + \epsilon}$$

```python
def compute_deltas(rewards: torch.Tensor, mode: str) -> torch.Tensor:
    """
    计算GRPO的delta（优势估计）
    
    Args:
        rewards: [batch, num_responses] 每个response的奖励
        mode: "rewards" | "centered_rewards" | "normalized_rewards"
    
    Returns:
        deltas: [batch, num_responses] 用于更新的权重
    """
    if mode == "rewards":
        # 朴素策略梯度
        return rewards
    
    if mode == "centered_rewards":
        # 减去组内均值
        mean_rewards = rewards.mean(dim=-1, keepdim=True)
        return rewards - mean_rewards
    
    if mode == "normalized_rewards":
        # 减去均值，除以标准差
        mean_rewards = rewards.mean(dim=-1, keepdim=True)
        std_rewards = rewards.std(dim=-1, keepdim=True)
        centered = rewards - mean_rewards
        return centered / (std_rewards + 1e-5)
    
    raise ValueError(f"Unknown mode: {mode}")
```

### 4.3 简单任务：数字排序

```python
def sort_inclusion_ordering_reward(prompt: list[int], response: list[int]) -> float:
    """
    评估排序response的奖励
    
    给分规则:
    1. 每个prompt中的数字出现在response中 → +1
    2. 每对相邻数字是升序 → +1
    """
    # 包含奖励
    inclusion_reward = sum(1 for x in prompt if x in response)
    
    # 排序奖励
    ordering_reward = sum(1 for i in range(len(response)-1) 
                          if response[i] <= response[i+1])
    
    return inclusion_reward + ordering_reward

# 示例
prompt = [3, 1, 0, 2]
correct_response = [0, 1, 2, 3]  # 奖励 = 4(包含) + 3(排序) = 7
wrong_response = [7, 2, 2, 5]    # 奖励 = 1(包含) + 2(排序) = 3
```

### 4.4 简单模型

```python
class Model(nn.Module):
    """简化的非自回归排序模型"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 prompt_length: int, response_length: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 每个位置有独立的编码/解码权重
        self.encode_weights = nn.Parameter(
            torch.randn(prompt_length, embedding_dim, embedding_dim) / math.sqrt(embedding_dim)
        )
        self.decode_weights = nn.Parameter(
            torch.randn(response_length, embedding_dim, embedding_dim) / math.sqrt(embedding_dim)
        )
    
    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompts: [batch, prompt_length]
        Returns:
            logits: [batch, response_length, vocab_size]
        """
        # 嵌入
        embeddings = self.embedding(prompts)  # [batch, pos, dim]
        
        # 编码：对prompt位置加权求和
        encoded = einsum(embeddings, self.encode_weights, 
                        "batch pos dim1, pos dim1 dim2 -> batch dim2")
        
        # 解码：为每个response位置生成向量
        decoded = einsum(encoded, self.decode_weights,
                        "batch dim2, pos dim2 dim1 -> batch pos dim1")
        
        # 转换为logits（输入输出共享embedding）
        logits = einsum(decoded, self.embedding.weight,
                       "batch pos dim1, vocab dim1 -> batch pos vocab")
        
        return logits
```

### 4.5 生成Responses

```python
def generate_responses(prompts: torch.Tensor, model: Model, 
                       num_responses: int) -> torch.Tensor:
    """
    为每个prompt生成多个responses
    
    Returns:
        responses: [batch, num_responses, response_length]
    """
    logits = model(prompts)  # [batch, pos, vocab]
    
    # 采样
    batch_size = prompts.shape[0]
    flattened_logits = rearrange(logits, "batch pos vocab -> (batch pos) vocab")
    flattened_responses = torch.multinomial(
        F.softmax(flattened_logits, dim=-1), 
        num_samples=num_responses, 
        replacement=True
    )
    responses = rearrange(
        flattened_responses, 
        "(batch pos) trial -> batch trial pos", 
        batch=batch_size
    )
    
    return responses
```

### 4.6 计算Log概率

```python
def compute_log_probs(prompts: torch.Tensor, responses: torch.Tensor, 
                      model: Model) -> torch.Tensor:
    """
    计算responses的log概率
    
    Returns:
        log_probs: [batch, num_responses, response_length]
    """
    logits = model(prompts)  # [batch, pos, vocab]
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, pos, vocab]
    
    # 扩展以匹配responses维度
    num_responses = responses.shape[1]
    log_probs = repeat(log_probs, "batch pos vocab -> batch trial pos vocab", 
                       trial=num_responses)
    
    # 索引获取实际选择的token的log概率
    log_probs = log_probs.gather(dim=-1, index=responses.unsqueeze(-1)).squeeze(-1)
    
    return log_probs
```

### 4.7 损失计算

```python
def compute_loss(log_probs: torch.Tensor, deltas: torch.Tensor, 
                 mode: str, old_log_probs: torch.Tensor = None) -> torch.Tensor:
    """
    计算策略梯度损失
    
    Args:
        log_probs: [batch, trial, pos] 当前策略的log概率
        deltas: [batch, trial] 优势/奖励
        mode: "naive" | "clipped"
        old_log_probs: [batch, trial, pos] 旧策略的log概率（用于裁剪）
    """
    if mode == "naive":
        # 朴素策略梯度: -E[δ · log π]
        loss = -einsum(log_probs, deltas, 
                       "batch trial pos, batch trial -> batch trial pos").mean()
        return loss
    
    if mode == "clipped":
        epsilon = 0.1
        # 计算概率比
        ratios = log_probs / old_log_probs  # 注意：这里应该是exp(log差)
        
        unclipped = einsum(ratios, deltas, 
                          "batch trial pos, batch trial -> batch trial pos")
        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        clipped = einsum(clipped_ratios, deltas,
                        "batch trial pos, batch trial -> batch trial pos")
        
        return -torch.minimum(unclipped, clipped).mean()
    
    raise ValueError(f"Unknown mode: {mode}")
```

### 4.8 KL惩罚

```python
def compute_kl_penalty(log_probs: torch.Tensor, 
                       ref_log_probs: torch.Tensor) -> torch.Tensor:
    """
    计算KL散度惩罚
    
    使用低方差估计:
    KL(p||q) = E_p[q/p - log(q/p) - 1]
    """
    # ref/current的概率比
    ratio = torch.exp(ref_log_probs - log_probs)
    
    # 低方差KL估计
    kl = ratio - (ref_log_probs - log_probs) - 1
    
    return kl.sum(dim=-1).mean()
```

### 4.9 完整训练循环

```python
def run_policy_gradient(num_epochs: int = 100,
                        num_steps_per_epoch: int = 10,
                        num_responses: int = 10,
                        deltas_mode: str = "centered_rewards",
                        kl_penalty: float = 0.0):
    """完整的GRPO训练循环"""
    
    # 数据
    prompts = torch.tensor([[1, 0, 2], [3, 2, 4], [1, 2, 3]])
    vocab_size = prompts.max() + 1
    
    # 模型
    model = Model(vocab_size=vocab_size, embedding_dim=10,
                  prompt_length=3, response_length=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 参考模型（用于KL惩罚）
    ref_model = None
    
    for epoch in range(num_epochs):
        # 定期更新参考模型
        if kl_penalty != 0 and epoch % 10 == 0:
            ref_model = copy.deepcopy(model)
            for p in ref_model.parameters():
                p.requires_grad = False
        
        # 生成responses
        responses = generate_responses(prompts, model, num_responses)
        
        # 计算奖励
        rewards = compute_reward(prompts, responses, sort_inclusion_ordering_reward)
        
        # 计算delta（优势）
        deltas = compute_deltas(rewards, mode=deltas_mode)
        
        # 保存旧的log概率（用于裁剪）
        with torch.no_grad():
            old_log_probs = compute_log_probs(prompts, responses, model)
        
        # 参考模型log概率（用于KL）
        if ref_model is not None:
            with torch.no_grad():
                ref_log_probs = compute_log_probs(prompts, responses, ref_model)
        
        # 内层循环：多步更新
        for step in range(num_steps_per_epoch):
            # 当前log概率
            log_probs = compute_log_probs(prompts, responses, model)
            
            # 策略梯度损失
            loss = compute_loss(log_probs, deltas, mode="naive")
            
            # KL惩罚
            if kl_penalty != 0:
                loss += kl_penalty * compute_kl_penalty(log_probs, ref_log_probs)
            
            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 打印进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: mean_reward = {rewards.mean():.2f}")
```

![GRPO算法伪代码](./images/grpo-algorithm.png)

---

## 五、实验观察 (Experimental Observations)

### 5.1 朴素奖励 vs 中心化奖励

**朴素奖励问题**:
- 如果所有response获得相同奖励（如都是3）
- 仍然会做更新（不应该！）

**中心化奖励解决方案**:
- 减去均值后，相同奖励 → delta = 0 → 不更新
- 只在有差异时才学习

```python
# 示例
rewards = torch.tensor([3., 3., 3., 3.])

# 朴素
deltas_naive = rewards  # [3, 3, 3, 3] → 会更新！

# 中心化
deltas_centered = rewards - rewards.mean()  # [0, 0, 0, 0] → 不更新
```

### 5.2 标准差归一化

**效果**: 使得所有prompt的梯度尺度一致

**潜在问题** (Dr. GRPO论文):
- 简单/困难问题的标准差小 → 梯度被放大
- 可能导致在简单题上浪费更新

### 5.3 损失曲线的误导性

**重要警告**: RL中的损失曲线**不像监督学习那样有意义**！

```
每个epoch:
- 生成新的responses
- 计算新的奖励和delta
- 损失是针对*新*数据计算的

⟹ 损失不是在同一分布上计算的，不能直接比较
```

**应该看**: 平均奖励（而非损失）

### 5.4 实验结论

从排序任务的实验：
1. **中心化有帮助**: 避免在无信号样本上更新
2. **标准差归一化效果有限**: 在这个设置下差异不大
3. **容易陷入局部最优**: 部分正确但不完全正确
4. **部分奖励是双刃剑**: 给太多部分分可能阻碍进步

---

## 六、工程注意事项 (Engineering Considerations)

### 6.1 梯度计算的陷阱

**关键**: 区分"常数"和"变量"

```python
# ❌ 错误：对两个都求梯度
w = torch.tensor(2., requires_grad=True)
p = torch.sigmoid(w)
p_old = torch.sigmoid(w)  # 这也有梯度！
ratio = p / p_old
ratio.backward()
print(w.grad)  # 0! 因为梯度相消

# ✅ 正确：冻结p_old
w = torch.tensor(2., requires_grad=True)
p = torch.sigmoid(w)
with torch.no_grad():
    p_old = torch.sigmoid(w)  # 作为常数
ratio = p / p_old
ratio.backward()
print(w.grad)  # 非零
```

### 6.2 多模型管理

GRPO需要管理多个模型状态：

| 模型 | 用途 | 更新频率 |
|------|------|----------|
| $\pi_\theta$ | 当前策略 | 每步更新 |
| $\pi_{old}$ | 重要性采样 | 每epoch更新 |
| $\pi_{ref}$ | KL正则化 | 每N epoch更新 |

**技巧**: $\pi_{old}$ 不需要存储模型，只需存储log_probs

### 6.3 系统复杂性

真实GRPO系统需要：
- **推理工作节点**: 专门做生成（GPU密集）
- **训练工作节点**: 专门做梯度更新（GPU密集）
- **奖励计算**: 可能需要执行代码、查数据库等
- **模型权重同步**: 训练后同步到推理节点
- **长CoT处理**: 不均匀batch的负载均衡

---

## 七、关键要点总结 (Key Takeaways)

### 核心公式

$$\nabla J = \mathbb{E}\left[\nabla \log \pi_\theta(a|s) \cdot \underbrace{(R - b(s))}_{\text{基线后奖励}}\right]$$

### GRPO特色

1. **无需价值函数**: 用组内均值替代
2. **利用LM结构**: 多response采样提供基线
3. **简单有效**: 已被R1、K1.5等验证

### 实践建议

- **总是使用基线**: 减少方差
- **监控奖励，不只是损失**: 损失不可比
- **注意局部最优**: RL容易陷入
- **奖励设计是关键**: 比算法更重要

### RL的本质

> "If you can measure it, you can optimize it."
> 
> 如果你能衡量它，你就能优化它。

但关键是：
1. 衡量标准是否可靠？（RLHF的挑战）
2. 优化过程是否稳定？（方差的挑战）
3. 奖励是否可泛化？（过拟合的挑战）

---

## 附录：完整代码参考

完整实现请参考课程代码：
`spring2025-lectures/lecture_17.py`

包含：
- 排序任务定义
- 简化模型实现
- GRPO训练循环
- 多种delta模式对比实验
- 可视化训练曲线

---

## 参考资料

1. **Policy Gradient Theorem**: Sutton & Barto, Reinforcement Learning: An Introduction, Chapter 13
2. **GRPO**: Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning
3. **Dr. GRPO**: Liu et al. (2025). Understanding R1-Zero-Like Training: A Critical Perspective
4. **CS224R**: Stanford Deep Reinforcement Learning, Lecture Notes on Policy Gradients
