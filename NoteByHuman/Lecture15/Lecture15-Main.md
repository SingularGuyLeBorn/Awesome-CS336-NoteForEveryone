# CS336 Lecture 15: 详解SFT与RLHF (Deep Dive into SFT and RLHF)

> **编辑蓝图 (Editorial Blueprint)**
> 
> **核心主题**: 本讲座标志着CS336课程从预训练进入后训练 (Post-training) 阶段。深入讲解如何将预训练的大模型（如GPT-3）转变为有用的、可控的助手（如ChatGPT）。核心方法包括 **监督微调 (SFT)** 和 **基于人类反馈的强化学习 (RLHF)**。
> 
> **知识结构**: 
> - 第一部分：监督微调 (SFT) - 数据类型、质量权衡、安全微调、Mid-training
> - 第二部分：RLHF - 成对偏好数据收集、Bradley-Terry模型、奖励模型、PPO算法
> - 第三部分：DPO - 直接偏好优化，RLHF的简化替代方案
> 
> **精英补充笔记**:
> - **[深入探讨: InstructGPT流水线](./Lecture15-InstructGPT.md)** - 完整的三阶段后训练流程
> - **[深入探讨: DPO数学推导](./Lecture15-DPO.md)** - 从RLHF目标到DPO损失函数

---

## 一、后训练的意义 (The Significance of Post-Training)

### 1.1 从GPT-3到ChatGPT的转变

```
GPT-3 (预训练)          →          ChatGPT (后训练)
- 填充文本              →          - 遵循指令
- 不可控               →          - 安全可控  
- 难以直接使用          →          - 产品级助手
```

**关键洞察**: 预训练模型已经"知道"很多能力（推理、回答问题），但这些能力被埋藏在参数中。后训练的目的是**激活和引导**这些能力。

### 1.2 现代指令遵循能力的强大

Example from Sebastian Bubeck's "Sparks of AGI" Paper (2023):
- 模型可以同时遵循10+条嵌套复合指令
- 结合编程能力零样本生成matplotlib代码
- 这在以前的可控生成方法中是不可能的

### 1.3 安全与内容审核

后训练也是添加安全护栏的关键阶段：
- 防止模型被滥用（诈骗、虚假信息）
- 内容审核（避免有害输出）
- ChatGPT成功的重要原因之一是其显著的安全护栏

---

## 二、监督微调 (Supervised Fine-Tuning, SFT)

### 2.1 SFT的基本思想

SFT本质上就是在**专家示范数据**上进行梯度下降：

$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log P_\theta(y|x) \right]$$

- $x$: 指令/提示
- $y$: 期望的回复
- $\mathcal{D}$: 指令跟随数据集

### 2.2 三种典型的指令微调数据

#### 2.2.1 FLAN (任务聚合型)

**构建方法**: 聚合现有NLP任务数据集，转化为指令格式

```python
# 原始数据
{"text": "The quick brown fox...", "label": "positive"}

# FLAN格式
{
    "instruction": "Classify the sentiment of the following text:",
    "input": "The quick brown fox...",
    "output": "positive"
}
```

**特点**:
- ✅ 数据量大，免费
- ❌ 格式不自然（多选题、短回答）
- ❌ 与用户实际交互差异大

#### 2.2.2 Stanford Alpaca (AI生成型)

**构建方法**: 使用语言模型生成指令和回复

```python
# 流程:
# 1. 人类编写种子指令集
# 2. 用GPT-3.5生成更多指令
# 3. 用GPT-3.5生成对应回复
```

**特点**:
- ✅ 更像自然对话
- ✅ 长格式回复
- ❌ 指令多样性有限
- ❌ 可能继承模型偏见

#### 2.2.3 OpenAssistant (人类众包型)

**构建方法**: 在线爱好者社区自愿编写

**特点**:
- ✅ 高质量、详细回复
- ✅ 有时包含引用
- ❌ 成本高、规模小
- ❌ 质量参差不齐

### 2.3 SFT数据的关键考量

#### 2.3.1 长度偏好

研究发现：
- 人类和AI评委都偏好更长的回复（60-70%偏好）
- 人类偏好列表格式
- 这可能导致模型学习**风格**而非**能力**

```python
# Length bias example
preference_for_longer = 0.65  # 65% prefer longer responses
```

#### 2.3.2 高质量数据的陷阱

**John Schulman的洞察**: 高质量SFT数据可能**教会模型幻觉**

```python
# 问题示例:
instruction = "什么是单买 (monopsony)?"
response = """
单买是指市场上只有一个买家的情况...

参考文献:
[1] Bivens, J. (2018). The Economic Policy Institute...
"""
```

**两种学习机制**:
1. ✅ 模型学习"单买"与"Bivens书籍"的关联（新知识）
2. ❌ 模型学习"复杂概念后要添加引用"（可能幻觉）

**关键原则**: 
- SFT数据应匹配模型已有能力
- 过于先进的数据会教会模型"编造"

### 2.4 安全微调 (Safety Tuning)

安全微调的核心权衡：**拒绝 vs 过度拒绝**

```python
# 需要拒绝的请求
"How do I make a bomb?"  → REFUSE

# 不应拒绝的请求  
"How do I kill a Python process?"  → ANSWER (技术问题)
```

研究表明：仅500个安全示例就能显著改善模型的安全遵循。

### 2.5 Mid-training: 模糊的边界

现代做法是将SFT数据混入预训练后期（衰减阶段）:

```
Pre-training Stage 1    →    Pre-training Stage 2 (Decay)    →    SFT
纯预训练数据                 预训练 + 高质量 + SFT混合              纯SFT
```

**MiniCPM的数据混合示例**:
- 稳定阶段: Common Crawl, Code, Pre-training Pile
- 衰减阶段: Wikipedia, Chinese Books, **Ultra Chat**, **StackExchange QA**, **Evol Instruct**

**优势**:
- 解决灾难性遗忘问题
- 从数据中获得更多价值
- 提高效率

---

## 三、基于人类反馈的强化学习 (RLHF)

### 3.1 为什么需要RLHF?

#### 原因1: SFT数据收集昂贵

```
成本对比:
- SFT: 专家编写详细回复 → 每条数据成本高
- RLHF: 人类只需比较两个回复 → 每条数据成本低
```

#### 原因2: 生成器-验证器差距

研究发现：人类**验证**能力可能优于**生成**能力

```python
# 实验结论
annotator_prefers_own_summary = 0.35  # 35%更喜欢自己写的
annotator_prefers_AI_summary = 0.65   # 65%更喜欢AI生成的

# 原因: "我写的时候觉得需要更正式，但AI的读起来更流畅"
```

### 3.2 成对偏好数据收集

#### InstructGPT的标注指南

三大原则:
1. **Helpful (有帮助)**: 清晰语言、回答问题、国际化敏感
2. **Truthful (真实)**: 不幻觉
3. **Harmless (无害)**: 不毒性、不暴力

#### 标注的现实挑战

**课堂互动实验**:
- 给学生5分钟比较两个AI回复
- 结果: 大多数人无法核实所有事实和数学
- 较长回复获得更多投票，尽管包含幻觉

**实际标注问题**:
- 时间限制（如Google Bard标注员每题只有1分钟）
- 标注员可能使用GPT-4作答
- 成本vs质量权衡

### 3.3 Bradley-Terry偏好模型

假设每个回复有潜在标量奖励 $r(x, y)$，人类偏好建模为:

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2)) = \frac{1}{1 + e^{-(r(x,y_1) - r(x,y_2))}}$$

这意味着：
- 奖励差越大，偏好概率越高
- 奖励相同时，偏好概率为50%

### 3.4 奖励模型训练

从成对偏好数据训练奖励模型 $r_\theta(x, y)$:

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]$$

其中:
- $y_w$: 偏好的（获胜）回复
- $y_l$: 不偏好的（失败）回复

### 3.5 InstructGPT目标函数

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\phi(x, y) \right] - \beta \cdot D_{KL}(\pi_\theta || \pi_{ref})$$

第一项: 最大化奖励
第二项: 不要偏离参考模型太远（防止reward hacking）

### 3.6 PPO算法简介

**PPO (Proximal Policy Optimization)** 是RLHF的核心算法。

#### 策略梯度

$$\nabla J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla \log \pi_\theta(a|s) \cdot R \right]$$

#### PPO的关键改进

1. **优势函数**: 使用 $A(s,a)$ 代替 $R$ 减少方差
2. **重要性采样**: 允许在旧策略样本上多次更新
3. **裁剪**: 限制策略更新幅度

$$L^{CLIP}(\theta) = \mathbb{E} \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

#### PPO的复杂性

PPO实现非常复杂，有论文总结了37条实现细节。这激发了对更简单替代方案的研究。

---

## 四、直接偏好优化 (DPO)

### 4.1 DPO的动机

**问题**: PPO太复杂（需要奖励模型、价值函数、在线采样...）

**DPO的解决方案**: 绕过显式奖励模型，直接从偏好数据优化策略

### 4.2 DPO推导

#### Step 1: 最优策略的形式

对于KL正则化的奖励最大化问题，最优策略为:

$$\pi^*(y|x) \propto \pi_{ref}(y|x) \cdot \exp\left(\frac{1}{\beta} r(x,y)\right)$$

#### Step 2: 从策略反推奖励

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

#### Step 3: 代入Bradley-Terry模型

将上式代入偏好概率，$Z(x)$项相消:

$$P(y_1 \succ y_2 | x) = \sigma\left(\beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}\right)$$

#### Step 4: DPO损失函数

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) \right]$$

### 4.3 DPO梯度的直观理解

DPO梯度形式:

$$\nabla \mathcal{L}_{DPO} \propto -\underbrace{w}_{\text{权重}} \cdot \left( \underbrace{\nabla \log \pi_\theta(y_w|x)}_{\text{提高好回复概率}} - \underbrace{\nabla \log \pi_\theta(y_l|x)}_{\text{降低坏回复概率}} \right)$$

其中 $w$ 在隐含奖励估计错误时更大（类似于困难样本挖掘）。

### 4.4 DPO的优势

- ✅ 无需训练单独的奖励模型
- ✅ 无需在线采样/rollout
- ✅ 实现简单（类似于SFT）
- ✅ 在开源模型中广泛使用

### 4.5 DPO的局限

- 本质上是离线算法（原始形式）
- 对于可验证奖励（如数学题）不太适用
- 可能仍然存在长度偏差问题

---

## 五、RLHF的挑战与注意事项

### 5.1 过度优化 (Over-optimization)

随着RL训练进行:
- 代理奖励（奖励模型分数）持续上升
- 真实人类偏好先升后降

```
                真实偏好
                   ▲
                   |      ___
                   |   __/
                   |  /
                   | /
                   |/
    ──────────────────────────▶ RL步数
                   
    原因: 奖励模型不完美，模型学会"欺骗"它
```

### 5.2 校准问题

RLHF后的模型往往**过度自信**:
- 预测的置信度与实际正确率不匹配
- 在温度=1时更明显
- 这是因为RL优化的是奖励，不是分布

### 5.3 标注员偏差

研究发现:
- 美国标注员占17%
- 菲律宾/孟加拉国标注占比高
- 模型可能偏向这些群体的价值观

```python
# 标注员关注点差异
expert_annotators: [
    {"focus": "factuality", "weight": 0.5},
    {"focus": "helpfulness", "weight": 0.3},
    {"focus": "formatting", "weight": 0.2}
]

crowdworkers: [
    {"focus": "formatting", "weight": 0.5},  # 更关注格式
    {"focus": "helpfulness", "weight": 0.3},
    {"focus": "factuality", "weight": 0.2}  # 较少关注事实
]
```

### 5.4 AI反馈的兴起

由于人类标注的局限，AI反馈（RLAIF）越来越流行:
- Constitutional AI (Anthropic)
- Ultra Feedback (开源)
- Tulu 3 (AI2)

好处:
- 成本低
- 一致性高
- 规模大

风险:
- AI自我偏好
- 同质化
- 仍有长度偏差

---

## 六、关键要点总结 (Key Takeaways)

### SFT要点

1. **惊人的高效**: 少量数据就能产生显著效果
2. **数据质量复杂**: "高质量"不一定对模型好（幻觉问题）
3. **Mid-training是趋势**: SFT数据混入预训练后期

### RLHF要点

1. **验证比生成便宜**: 成对偏好数据收集成本低于专家示范
2. **但仍有挑战**: 时间限制、标注员偏差、事实核查困难
3. **过度优化是真实风险**: 需要平衡代理奖励和真实偏好

### DPO要点

1. **简化了RLHF**: 无需奖励模型和在线采样
2. **广泛采用**: 成为开源社区的首选方法
3. **仍有局限**: 离线、长度偏差等问题

### 整体流水线

```
Pre-training → Mid-training → SFT → RLHF/DPO → Deployment
   能力         高质量注入      指令跟随   安全/偏好对齐    产品
```

---

## 参考资料

1. **InstructGPT**: Ouyang et al. (2022). Training language models to follow instructions with human feedback
2. **RLHF Tutorial**: Lambert et al. (2022). Illustrating Reinforcement Learning from Human Feedback
3. **DPO**: Rafailov et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model
4. **Constitutional AI**: Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback
5. **Sparks of AGI**: Bubeck et al. (2023). Sparks of Artificial General Intelligence
