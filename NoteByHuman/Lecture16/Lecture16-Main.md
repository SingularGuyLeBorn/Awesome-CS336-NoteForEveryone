# CS336 Lecture 16: 详解大模型RL算法 (Deep Dive into Large Model RL Algorithms)

> **编辑蓝图 (Editorial Blueprint)**
> 
> **核心主题**: 本讲座是CS336课程RL系列的高潮，从RLHF过渡到**可验证奖励的强化学习 (RL from Verifiable Rewards)**。详细讲解PPO算法、GRPO算法，并深入分析三个重要的推理模型案例：**DeepSeek R1**、**Kimi K1.5**和**Qwen 3**。
> 
> **知识结构**: 
> - 第一部分：RLHF收尾 - DPO变体、过度优化、校准问题
> - 第二部分：PPO详解 - 策略梯度、重要性采样、裁剪
> - 第三部分：GRPO详解 - 移除价值函数、组内基线
> - 第四部分：推理模型案例分析 - R1、K1.5、Qwen 3
> 
> **精英补充笔记**:
> - **[深入探讨: DeepSeek R1技术报告](./Lecture16-DeepSeek-R1.md)** - R1-Zero、SFT初始化、语言一致性
> - **[深入探讨: GRPO数学细节](./Lecture16-GRPO-Math.md)** - 标准差归一化问题、Dr. GRPO

---

## 一、RLHF收尾 (RLHF Wrap-up)

### 1.1 DPO回顾

DPO更新形式:

$$\nabla \mathcal{L}_{DPO} \propto \beta \cdot w(\theta) \cdot \left( \nabla \log \pi_\theta(y_w|x) - \nabla \log \pi_\theta(y_l|x) \right)$$

其中 $w(\theta)$ 在奖励估计错误时更大——这是一种隐式的困难样本挖掘。

**核心思想**: RL算法本质上都是"上调好的，下调坏的"，区别在于如何确定"好坏"和"多少"。

### 1.2 DPO变体

#### SimPO

SimPO做了两个简化:
1. **长度归一化**: 除以回复长度，避免长度偏差
2. **移除参考模型**: 不再计算与参考模型的比率

$$\mathcal{L}_{SimPO} = -\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x)\right)$$

#### 实验发现的矛盾结论

**Tulu 2研究**: PPO优于DPO（因为在线特性）
**Tulu 3研究**: 如果SFT做得好，DPO和PPO差距消失，长度归一化DPO最佳

> **重要启示**: RL的实验结论高度依赖具体设置（模型、数据、评估方法），不应盲目泛化单篇论文的结论。

### 1.3 过度优化 (Over-optimization)

```
代理奖励 vs 真实偏好:

代理奖励 ▲                    真实偏好 ▲
         |    _____                    |    ____
         |   /                         |   /    \
         |  /                          |  /      \
         | /                           | /        \
         |/                            |/          \_____
         ───────────▶ RL步数           ───────────▶ RL步数
```

**原因分析**:
- 奖励模型是有噪声的近似
- 模型学会"欺骗"奖励模型
- 类似于过拟合，但发生在策略层面

**实验证据** (来自作者实验室):
- RLHF on 人类偏好: 过度优化
- RLHF on 有噪声AI反馈: 过度优化
- RLHF on 无噪声AI反馈: 无过度优化

### 1.4 校准问题

RLHF后的模型**不再是校准的概率模型**:

- 我们优化的是奖励，不是概率分布
- 温度=1时，模型表现过度自信
- 预测的置信度 ≠ 实际正确率

来自多个论文的证据:
- Anthropic论文
- GPT-4发布论文
- 学术独立研究

---

## 二、从RLHF到可验证奖励 (From RLHF to Verifiable Rewards)

### 2.1 RLHF的局限

| 问题 | 描述 |
|------|------|
| 人类偏好噪声大 | 标注员会犯错、有偏见 |
| 难以规模化 | 人类标注成本高 |
| 过度优化风险 | 模型学会欺骗奖励模型 |
| 无法验证正确性 | 对于数学/代码，正确性是客观的 |

### 2.2 可验证奖励的优势

**核心思想**: 如果我们有**确定性的奖励函数**（不是学习的），就可以充分发挥RL的威力。

| 领域 | 可验证奖励 |
|------|-----------|
| 数学 | 答案是否正确 |
| 代码 | 测试用例是否通过 |
| 游戏 | 是否获胜 |
| 形式验证 | 证明是否有效 |

**类比AlphaGo/AlphaFold**: 这些成功案例都有**确定性的奖励函数**。

---

## 三、PPO详解 (PPO Deep Dive)

### 3.1 策略梯度基础

目标: 最大化期望奖励

$$J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[R(a)]$$

梯度:

$$\nabla J(\theta) = \mathbb{E}_{a \sim \pi_\theta}\left[\nabla \log \pi_\theta(a|s) \cdot R\right]$$

**朴素策略梯度**: 采样 $a \sim \pi_\theta$，更新 $\theta \leftarrow \theta + \alpha \nabla \log \pi_\theta(a|s) R$

**问题**: 纯在线，每次采样后只能更新一次。

### 3.2 TRPO: 引入重要性采样

核心想法: 从旧策略 $\pi_{\theta_{old}}$ 采样，但对新策略 $\pi_\theta$ 进行更新。

$$\nabla J(\theta) = \mathbb{E}_{a \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \nabla \log \pi_\theta(a|s) \cdot A \right]$$

其中 $A$ 是优势函数（Advantage），是$R$的低方差版本。

TRPO添加KL约束: $D_{KL}(\pi_\theta || \pi_{\theta_{old}}) \leq \delta$

### 3.3 PPO: 裁剪替代KL约束

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，$\epsilon$通常取0.2。

**直观解释**:
- 如果新策略偏离太远（$r_t$ 超出 $[1-\epsilon, 1+\epsilon]$），梯度被裁剪
- 这自然地限制了策略更新幅度

### 3.4 PPO的复杂性

```
PPO完整实现需要:
┌─────────────────────────────────────────────┐
│  策略模型 π_θ                                │
│  价值模型 V_φ (用于计算优势)                   │
│  奖励模型 r (RLHF设置)                        │
│  参考模型 π_ref (KL正则化)                    │
│  广义优势估计 (GAE)                           │
│  多步更新 + 重要性采样                        │
│  37条实现细节...                             │
└─────────────────────────────────────────────┘
```

**论文**: "Implementation Matters in Deep RL: A Case Study on PPO and TRPO"

PPO实现不当可能导致:
- 计算的甚至不是正确的策略梯度
- 但居然可能效果更好

---

## 四、GRPO详解 (GRPO Deep Dive)

### 4.1 GRPO的动机

**核心问题**: 能否**移除价值函数**，同时保持PPO的效果？

**语言模型的特殊性**:
- 对于同一个prompt，可以生成多个responses
- 这提供了**自然的基线估计**

### 4.2 GRPO公式

#### 优势估计

$$A_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

其中 $G$ 是每个prompt生成的response数量。

**解释**:
- 不需要训练价值函数
- 使用同组responses的奖励均值作为基线
- 标准差归一化（可选）

#### GRPO损失

$$L^{GRPO}(\theta) = \mathbb{E}\left[\min\left( r_t(\theta) A_i, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_i \right)\right]$$

### 4.3 GRPO vs PPO

| 特性 | PPO | GRPO |
|------|-----|------|
| 价值函数 | 需要 | 不需要 |
| 内存占用 | 2x模型 | 1x模型 |
| 基线来源 | 学习的V(s) | 组内经验均值 |
| 适用场景 | 通用RL | 语言模型 |

### 4.4 GRPO的潜在问题

#### 问题1: 标准差归一化

除以标准差是否合法？

**理论分析**: 标准的策略梯度定理只允许**减去**与动作无关的基线，**除法**不在允许范围内。

**问题**:
- 标准差小 → 放大梯度 → 问题太简单或太难时梯度过大
- 可能影响收敛

#### 问题2: 长度归一化

GRPO原始公式对奖励进行长度归一化:

**问题**:
- 答错时: 最优策略是生成最长回复（稀释负奖励）
- 答对时: 最优策略是生成最短回复（集中正奖励）
- 导致: 乱七八糟的长回复

**Dr. GRPO论文**建议移除这两个归一化。

![GRPO算法](./images/grpo-algorithm.png)

---

## 五、推理模型案例分析 (Reasoning Model Case Studies)

### 5.1 DeepSeek R1

#### R1-Zero: 纯RL实验

**设置**:
- 基础模型: DeepSeek V3（预训练+mid-training，无RLHF）
- 奖励: 准确性（正误）+ 格式（thinking标签）
- 算法: GRPO

**结果**:
- 性能接近OpenAI o1
- 思维链长度自然增长
- 出现"aha moment"（顿悟时刻）

**争议**:
- 长度增长可能是因为GRPO的长度偏差（Dr. GRPO论文）
- "aha moment"可能在预训练时就存在

#### R1完整流程

```
DeepSeek V3 Base
     │
     ▼ SFT初始化 (长CoT数据)
     │
     ▼ 推理RL (GRPO + 准确性 + 格式 + 语言一致性奖励)
     │
     ▼ 后续后训练 (通用能力保留)
     │
     ▼ R1
```

**关键发现**:
- **过程奖励模型 (PRM) 不如结果奖励**: 与DeepSeek Math结论相反
- **蒙特卡洛树搜索 (MCTS) 也不太有效**: 简单RL就够了
- **语言一致性奖励**: 防止CoT语言混合

![DeepSeek R1性能](./images/deepseek-r1-benchmarks.png)

#### 蒸馏到小模型

可以将R1的CoT蒸馏到Qwen等开源模型:
- 约100万条CoT数据
- 微调后显著提升数学性能

### 5.2 Kimi K1.5

#### 与R1的相似点

- SFT初始化
- 结果奖励RL
- 性能匹配o1

#### 独特贡献

##### 数据选择策略

```python
# 使用SFT模型评估难度
def select_training_data(problems, sft_model, num_samples=10):
    selected = []
    for problem in problems:
        # 生成多个回答
        responses = [sft_model.generate(problem) for _ in range(num_samples)]
        # 计算通过率
        pass_rate = sum(is_correct(r) for r in responses) / num_samples
        # 只保留"有挑战但可学"的问题
        if 0 < pass_rate < 1:  # 不全对也不全错
            selected.append(problem)
    return selected
```

##### 长度奖励

Kimi设计了专门控制CoT长度的奖励:

$$r_{length} = \begin{cases}
\lambda & \text{if correct (鼓励短回答)} \\
\text{batch average} & \text{if incorrect (不过度惩罚长回答)}
\end{cases}$$

其中 $\lambda$ 与回复长度在batch内的相对位置有关。

**注意**: 长度奖励不能太早启用，否则会导致RL停滞。

##### RL算法

Kimi使用了不同于GRPO的目标:

1. 非参数假设 → 奖励可写成策略比率形式（类似DPO推导）
2. 使用平方损失驱动等式成立
3. 梯度形式类似GRPO，但有不同的正则化

$$\nabla L \propto \underbrace{(r_i - \bar{r})}_{\text{基线}} \cdot \underbrace{\nabla \log \pi_\theta(y_i|x)}_{\text{策略梯度}} - \underbrace{(\log \pi_\theta - \log \pi_{ref})^2}_{\text{正则化}}$$

##### 系统工程

Kimi详细讨论了RL系统:
- RL worker和Inference worker分离
- 权重需要从RL worker传递到Inference worker
- 长CoT导致batch不均衡

### 5.3 Qwen 3

#### 整体流程

```
Base Model
    │
    ▼ Long CoT SFT
    │
    ▼ Reasoning RL
    │
    ▼ Thinking Mode Fusion (新!)
    │
    ▼ General RL
    │
    ▼ Qwen 3
```

#### 数据选择

与Kimi类似，使用best-of-N过滤:
- 如果base model已经能做对 → 太简单，排除
- 去污染: 移除与测试集相似的数据
- 人工筛选: 确保SFT数据无猜测

**惊人发现**: 仅用**3995个样本**进行RL就能获得显著提升！

#### Thinking Mode Fusion

**目标**: 在同一模型中支持"思考"和"不思考"两种模式

```
User: <think> 解释相对论 </think>
Model: [长CoT推理过程] 相对论是...

User: <no_think> 解释相对论 </no_think>  
Model: 相对论是... [直接回答]
```

**训练方法**:
- 用R1模型生成 `<think>` 数据
- 生成 `<no_think>` 直接回答数据
- 混合微调

**额外能力**: 可以在思考过程中**提前终止**

```
User: 考虑到时间有限，我需要直接给出答案...
Model: [停止思考] </think> 答案是...
```

#### 测试时计算扩展

通过控制思考token预算，实现平滑的性能-延迟权衡:

```
思考预算 ◀────────────────────────────▶
   低          中          高          无限
   │           │           │           │
   ▼           ▼           ▼           ▼
 快速回答   中等推理   深度推理   完整R1模式
```

#### 有趣发现

```
                    推理RL    Thinking Fusion   通用RL
通用任务               ↑            ↑              ↑
指令遵循               ↑            ↑              ↑
数学(think)           ↑            ↑              ↓ (!)
数学(no_think)        ↑            ↑              ↑
```

**观察**: 通用RL可能**损害**思考模式下的数学性能——存在tradeoff。

---

## 六、关键要点总结 (Key Takeaways)

### 从RLHF到可验证奖励

```
RLHF的困境:
- 人类偏好噪声 → 过度优化
- 难以规模化 → 成本限制
- 无法验证 → 不适合数学/代码

可验证奖励的解决方案:
- 确定性奖励函数 → 无过度优化风险
- 无需人类 → 无限规模化
- 客观验证 → 适合推理任务
```

### GRPO核心洞察

1. **语言模型特有结构**: 多response提供自然基线
2. **移除价值函数**: 减半内存，简化实现
3. **注意陷阱**: 标准差归一化和长度归一化可能有害

### 推理模型的共同模式

| 共同点 | 描述 |
|--------|------|
| SFT初始化 | 先用长CoT数据微调 |
| 结果奖励 | 只看最终答案正确性 |
| GRPO/类似算法 | 简单有效 |
| 难度课程 | 选择适当难度的训练数据 |

| 不同点 | R1 | K1.5 | Qwen 3 |
|--------|-----|------|--------|
| 语言一致性 | 有 | 不明确 | 不明确 |
| 长度控制 | 无明确 | 专门奖励 | 不明确 |
| 思考模式融合 | 无 | 无 | 有 |
| 开源程度 | 高 | 中 | 高 |

### 关键不确定性

1. **PRM vs ORM**: DeepSeek说ORM更好，但其他研究可能不同
2. **MCTS**: 目前似乎不如简单RL，但可能还有探索空间
3. **最优RL算法**: GRPO、Kimi的变体、还是其他？

---

## 参考资料

1. **PPO**: Schulman et al. (2017). Proximal Policy Optimization Algorithms
2. **GRPO**: Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
3. **DeepSeek R1**: DeepSeek (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
4. **Kimi K1.5**: Moonshot AI (2025). Kimi k1.5: Scaling Reinforcement Learning with LLMs
5. **Qwen 3**: Qwen Team (2025). Qwen3 Technical Report
6. **Dr. GRPO**: Liu et al. (2025). Understanding R1-Zero-Like Training: A Critical Perspective
