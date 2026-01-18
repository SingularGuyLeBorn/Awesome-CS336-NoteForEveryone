# 深入探讨: InstructGPT流水线

本文是Lecture 15的精英补充笔记，详细剖析InstructGPT论文中的三阶段后训练流程，这是理解现代大模型对齐技术的基础。

---

## 一、背景：从GPT-3到InstructGPT

### 1.1 GPT-3的问题

GPT-3 虽然能力强大，但存在根本性问题：
- **不遵循指令**: 用户说"写一首诗"，模型可能输出"诗是什么？诗有很多种..."
- **不安全**: 可能输出有害、偏见内容
- **不可控**: 行为难以预测

### 1.2 InstructGPT的目标

让模型做到三个H:
- **Helpful (有帮助)**: 遵循用户意图，提供有用回复
- **Honest (诚实)**: 不编造事实，承认不确定性
- **Harmless (无害)**: 拒绝有害请求，不输出危险内容

---

## 二、第一阶段：监督微调 (SFT)

### 2.1 数据收集

**来源**: 雇佣约40名标注员，编写高质量prompt-response对

**数据规模**: 约13,000条示范

**数据格式**:
```json
{
  "prompt": "Explain quantum entanglement to a 10-year-old",
  "response": "Imagine you have two magical coins that are best friends..."
}
```

### 2.2 标注指南摘要

**Helpful 原则**:
- 使用清晰、简洁的语言
- 直接回答问题
- 考虑国际化（如"football"可能指不同运动）
- 必要时请求澄清

**Honest 原则**:
- 说出真实想法
- 不编造来源或引用
- 对不确定的内容表达不确定

**Harmless 原则**:
- 不侮辱、贬低用户
- 不输出性暗示、暴力内容
- 避免敏感话题的极端立场

### 2.3 SFT训练细节

```python
# 伪代码
for epoch in range(num_epochs):
    for batch in dataloader:
        prompts, responses = batch
        
        # 标准语言模型损失
        loss = -log_prob(responses | prompts)
        
        loss.backward()
        optimizer.step()
```

**超参数**:
- 学习率: 9.65e-6（余弦衰减）
- Batch size: 32
- Epochs: 16
- 使用Dropout避免过拟合

### 2.4 SFT的局限

SFT能让模型"像样"地回答问题，但:
- 数据收集成本高
- 标注员能力有限（不如模型的最佳表现）
- 可能教会模型编造（如果训练数据超出模型能力）

---

## 三、第二阶段：奖励模型训练 (RM)

### 3.1 从示范到偏好

**洞察**: 让人类**比较**两个回复比**编写**回复更容易且质量更高

**数据收集流程**:
1. 给定prompt，让SFT模型生成多个回复
2. 标注员对回复进行排序（不只是成对比较）
3. 从排序生成所有成对偏好

### 3.2 数据规模

- 约33,000个prompts
- 每个prompt约4-9个回复排序
- 生成约300,000个成对比较

### 3.3 Bradley-Terry模型

假设每个回复有潜在分数 $r(x, y)$，偏好概率为:

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))$$

其中 $\sigma$ 是sigmoid函数。

### 3.4 奖励模型架构

奖励模型 = GPT-3架构 + 标量头

```python
class RewardModel(nn.Module):
    def __init__(self, gpt_model):
        self.backbone = gpt_model  # 共享GPT架构
        self.reward_head = nn.Linear(hidden_dim, 1)  # 输出标量
    
    def forward(self, prompt, response):
        # 编码prompt + response
        hidden = self.backbone.encode(prompt + response)
        # 取最后一个token的hidden state
        last_hidden = hidden[:, -1, :]
        # 输出标量奖励
        reward = self.reward_head(last_hidden)
        return reward
```

### 3.5 RM训练

使用成对比较损失:

$$\mathcal{L}_{RM} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

```python
for batch in dataloader:
    prompt, winner, loser = batch
    
    r_winner = reward_model(prompt, winner)
    r_loser = reward_model(prompt, loser)
    
    loss = -torch.log(torch.sigmoid(r_winner - r_loser)).mean()
    
    loss.backward()
    optimizer.step()
```

### 3.6 RM训练技巧

**同一排序的多对一起训练**:
- 如果有排序 A > B > C > D
- 一次性计算 (A,B), (A,C), (A,D), (B,C), (B,D), (C,D) 的损失
- 更高效，避免重复前向传播

**正则化**:
- 模型从GPT-3初始化，可能无需从头学习
- 早期停止防止过拟合

---

## 四、第三阶段：PPO强化学习

### 4.1 目标函数

$$\max_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta D_{KL}(\pi_\theta(y|x) || \pi_{SFT}(y|x)) \right]$$

**解释**:
- $r_\phi$: 奖励模型打分
- $\beta D_{KL}$: 惩罚偏离SFT模型太远（防止reward hacking）

### 4.2 KL惩罚的重要性

**没有KL惩罚**:
- 模型可能找到"欺骗"奖励模型的方式
- 生成高分但无意义的输出
- 语言能力退化

**实验发现**:
- $\beta$ 太小: reward hacking严重
- $\beta$ 太大: 几乎不更新，等于SFT模型
- 最优 $\beta$ 需要调参

### 4.3 PPO实现细节

#### 4.3.1 四个模型

| 模型 | 作用 | 更新 |
|------|------|------|
| Policy $\pi_\theta$ | 生成回复 | 每步更新 |
| Value $V_\psi$ | 估计期望奖励 | 每步更新 |
| Reward $r_\phi$ | 评估回复质量 | 冻结 |
| Reference $\pi_{SFT}$ | KL惩罚参考 | 冻结 |

#### 4.3.2 单步流程

```python
def ppo_step(prompts, policy, value, reward_model, ref_policy):
    # 1. 生成回复
    responses = policy.generate(prompts)
    
    # 2. 计算奖励（奖励模型 - KL惩罚）
    rm_scores = reward_model(prompts, responses)
    log_probs = policy.log_prob(responses | prompts)
    ref_log_probs = ref_policy.log_prob(responses | prompts)
    kl_penalty = beta * (log_probs - ref_log_probs)
    rewards = rm_scores - kl_penalty
    
    # 3. 计算优势
    values = value(prompts, responses)
    advantages = rewards - values  # 简化版
    
    # 4. PPO更新
    # ... (裁剪策略梯度)
    
    # 5. 更新价值函数
    value_loss = (values - rewards) ** 2
```

#### 4.3.3 每token奖励 vs 结果奖励

InstructGPT将KL惩罚**分配到每个token**:

$$r_t = -\beta \cdot (\log \pi_\theta(a_t|s_t) - \log \pi_{SFT}(a_t|s_t))$$

只有最后一个token获得RM分数:

$$r_T = r_\phi(x, y) + r_T^{KL}$$

### 4.4 PreTrain混合

为防止语言能力退化，InstructGPT混合预训练目标:

$$\mathcal{L} = \mathcal{L}_{PPO} + \gamma \mathcal{L}_{pretrain}$$

其中 $\mathcal{L}_{pretrain}$ 是在GPT-3预训练数据上的语言模型损失。

---

## 五、评估方法

### 5.1 人类偏好评估

- 随机采样prompts
- InstructGPT vs GPT-3 生成回复
- 标注员选择偏好

**结果**: InstructGPT获胜率 ~85%

### 5.2 TruthfulQA

测试模型是否会编造虚假信息。

**结果**: InstructGPT真实性显著提高

### 5.3 毒性评估

使用RealToxicityPrompts数据集。

**结果**: InstructGPT毒性显著降低

### 5.4 "对齐税" (Alignment Tax)

对齐可能损害某些能力:
- 在一些NLP benchmark上略有下降
- 但对于实际应用场景是值得的权衡

---

## 六、关键发现与教训

### 6.1 规模化的重要性

- 1.3B参数的InstructGPT优于175B的GPT-3
- 对齐比单纯增大规模更重要

### 6.2 数据质量 > 数据数量

- SFT只用13K示范
- 关键是标注员的质量和指南明确性

### 6.3 人类反馈的局限

- 标注员会犯错（事实核查时间不足）
- 标注员有偏见（文化、语言背景）
- 复杂任务难以评估

### 6.4 迭代改进

InstructGPT是迭代产物:
- 用当前模型生成数据
- 收集反馈
- 训练新模型
- 重复

---

## 七、后续发展

### 7.1 Constitutional AI (Anthropic)

用AI反馈替代部分人类反馈:
- 定义"宪法"原则
- 让AI自我批评和修正
- 减少人类标注需求

### 7.2 RLHF的简化

- DPO: 无需训练单独的奖励模型
- SLiC: 类似SFT但使用偏好排序
- 各种*PO变体

### 7.3 可验证奖励的兴起

- 数学、代码等领域
- 不需要人类偏好，直接验证正确性
- DeepSeek R1 等

---

## 参考资料

1. Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback
2. Christiano, P. et al. (2017). Deep reinforcement learning from human preferences
3. Stiennon, N. et al. (2020). Learning to summarize with human feedback
4. Bai, Y. et al. (2022). Training a Helpful and Harmless Assistant with RLHF
