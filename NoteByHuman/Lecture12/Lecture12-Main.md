# Lecture 12: 模型评估详解 (Deep Dive into Model Evaluation)

**课程**: CS336 · **主题**: 语言模型评估的哲学、框架与实践

## 0. 课程开场: 评估危机

> **Andre Karpathy**: "There is an evaluation crisis."

![Karpathy 评估危机推文](images/karpathy-crisis.png)

**评估** (Evaluation) 看似简单——给定一个固定模型, 问它有多"好"? 但讲师开宗明义地指出: 这其实是一个**极其深刻且复杂**的话题. 它不仅是一个机械过程 (抛出 Prompt, 计算指标), 更是**决定语言模型未来发展方向**的关键力量. 因为顶级模型开发者都在追踪这些评估指标, 而你追踪什么, 就会优化什么.

---

## 1. 你所看到的: 评估数据的多元面貌

当你看到一个语言模型发布时, 通常会看到各种"基准分数":

### 1.1 官方基准报告

![DeepSeek-R1 基准分数](images/deepseek-r1-benchmarks.png)
![Llama 4 基准分数](images/llama4-benchmarks.png)

近期的语言模型 (DeepSeek-R1, Llama 4, OLMo2 等) 通常在**相似但不完全相同**的基准上评估, 包括 MMLU, MATH, GPQA 等. 但这些基准到底是什么? 这些数字到底意味着什么?

### 1.2 聚合排行榜

![HELM Capabilities 排行榜](images/helm-capabilities-leaderboard.png)

**[HELM](https://crfm.stanford.edu/helm/capabilities/latest/)** 是斯坦福开发的统一评估框架, 将多种标准基准聚合在一起, 提供更全面的视角.

### 1.3 成本效益分析

![Artificial Analysis 帕累托前沿](images/artificial-analysis.png)

**[Artificial Analysis](https://artificialanalysis.ai/)** 提供了一个有趣的视角: **智能指数 vs 每 Token 价格**的帕累托前沿. 例如, O3 可能非常强大, 但也非常昂贵; 而某些模型可能性价比更高.

> **关键洞察**: 只看准确率是不够的, 还要考虑**成本**!

### 1.4 用户选择数据

![OpenRouter 流量排名](images/openrouter.png)

另一种思路: 如果人们愿意**付费使用**某个模型, 那它可能就是"好的". **[OpenRouter](https://openrouter.ai/rankings)** 基于实际流量 (发送到各模型的 Token 数) 生成排名.

### 1.5 人类偏好排名

![Chatbot Arena 排行榜](images/chatbot-arena-leaderboard.png)

**[Chatbot Arena](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)** 是目前最流行的人类偏好评估平台之一, 通过配对比较计算 ELO 评分.

### 1.6 社交媒体氛围 (Vibes)

除了硬指标, 还有"氛围"——人们在 X (Twitter) 上分享的酷炫示例. 这也是评估模型能力的一种非正式数据来源.

---

## 2. 如何思考评估: 目的决定方法

讲师强调: **评估没有"唯一真理" (No One True Evaluation)**, 它取决于你试图回答什么问题.

### 2.1 评估的不同目的

| 评估者身份                | 评估目的       | 示例问题                                      |
| ------------------------- | -------------- | --------------------------------------------- |
| **用户/企业**       | 购买决策       | Claude vs Gemini vs O3, 哪个适合我的客服场景? |
| **研究者**          | 测量原始能力   | 衡量"智能"水平, AI 是否在进步?                |
| **政策制定者/企业** | 理解收益与风险 | 模型带来的价值和危害分别是什么?               |
| **模型开发者**      | 获取改进反馈   | 评估作为开发循环的反馈信号                    |

每种情况下, 都有一个**抽象目标** (Abstract Goal) 需要被转化为**具体评估** (Concrete Evaluation). 选择什么评估方式, 取决于你的目标.

### 2.2 评估框架: 四个核心问题

1. **输入 (Inputs)**: Prompt 从哪里来? 覆盖哪些用例? 是否包含困难的长尾情况? 输入是否需要适应模型 (如多轮对话)?
2. **模型调用 (How to Call LM)**: 使用什么 Prompting 策略 (Zero-shot, Few-shot, Chain-of-Thought)? 是否使用工具、RAG? 评估的是**模型本身**还是**整个 Agent 系统**?
3. **输出评估 (How to Evaluate Outputs)**: 参考答案是否无误? 使用什么指标 (Pass@k)? 如何考虑成本? 如何处理不对称错误 (如医疗场景的幻觉)? 如何评估开放式生成?
4. **结果解读 (How to Interpret)**: 91% 意味着什么? 能否部署? 如何评估泛化能力 (考虑训练-测试重叠)? 评估的是模型还是方法?

> **总结**: 做评估时有**大量问题**需要思考, 它绝不仅仅是"跑个脚本"那么简单.

---

## 3. 困惑度 (Perplexity) 评估

### 3.1 基本概念

回顾: 语言模型是 Token 序列上的概率分布 $p(x)$.

**困惑度 (Perplexity)** 衡量模型是否对某个数据集 $D$ 分配了高概率:

$$
\text{Perplexity} = \left( \frac{1}{p(D)} \right)^{1/|D|}
$$

在预训练中, 我们**最小化训练集的困惑度**. 自然地, 评估时我们**测量测试集的困惑度**.

### 3.2 传统语言建模评估

在 2010 年代, 语言建模研究的标准流程是:

1. 选择一个标准数据集 (Penn Treebank, WikiText-103, 1 Billion Word Benchmark)
2. 在指定训练集上训练
3. 在指定测试集上评估困惑度

这是 N-gram 向神经网络过渡期的主要范式. 2016 年 Google 的论文展示了纯 CNN+LSTM 模型在 1 Billion Word Benchmark 上将困惑度从 **51.3 降到 30.0**.

### 3.3 GPT-2 的范式转变

![GPT-2 困惑度实验](images/gpt2-perplexity.png)

GPT-2 改变了游戏规则:

- 在 **40GB 的 WebText** 上训练 (Reddit 链接的网站)
- **零微调 (Zero-shot)**, 直接在传统困惑度基准上评估
- 这是**分布外评估** (Out-of-Distribution), 但理念是训练数据足够广泛

**结果**: 在小数据集 (Penn Treebank) 上超越 SOTA, 但在大数据集 (1 Billion Words) 上仍落后——这说明当数据集足够大时, 直接在该数据集上训练仍然更好.

自 GPT-2/3 以来, 语言模型论文更多转向**下游任务准确率** (Downstream Task Accuracy).

### 3.4 困惑度为何仍然有用

1. **比下游任务准确率更平滑**: 提供每个 Token 的细粒度概率, 适合拟合 Scaling Law
2. **具有普遍性**: 关注每一个 Token, 而任务准确率可能遗漏某些细节 (可能"对了但理由错了")
3. **可用于下游任务的 Scaling Law**: 可以测量**条件困惑度** (Conditional Perplexity), 直接针对特定任务拟合曲线

**注意**: 困惑度也可以通过条件方式应用于下游任务——给定 Prompt, 计算答案的概率.

### 3.5 困惑度的陷阱 (Leaderboard 警告)

如果你在运营一个排行榜:

- 对于**任务准确率**: 只需从黑盒模型获取生成输出, 然后用你的代码评估
- 对于**困惑度**: 需要模型提供概率, 并**信任它们加起来等于 1**

如果模型有 Bug (比如对所有 Token 都输出概率 0.8), 它会看起来非常好, 但那不是有效的概率分布.

### 3.6 困惑度最大化主义者 (Perplexity Maximalist) 的观点

设真实分布为 $t$, 模型为 $p$:

- 最佳困惑度是 $H(t)$ (信息熵), 当且仅当 $p = t$
- 如果你达到了 $t$, 你就解决了所有任务 → **AGI**
- 因此, 通过不断压低困惑度, 最终会达到 AGI

**反驳**: 这可能不是最高效的路径, 因为你可能在压低分布中"不重要"的部分.

### 3.7 与困惑度相近的任务

**LAMBADA** (完形填空):

![LAMBADA 示例](images/lambada.png)

给定需要长上下文理解的句子, 预测最后一个词.

**HellaSwag** (常识推理):

![HellaSwag 示例](images/hellaswag.png)

给定句子, 选择最合理的续写. 本质上是比较候选项的 Likelihood.

> **注意**: WikiHow 是一个网站. 虽然 HellaSwag 数据经过处理, 但如果你访问 WikiHow, 会看到与数据集非常相似的内容——这涉及**训练-测试重叠**问题.

---

## 4. 知识基准 (Knowledge Benchmarks)

### 4.1 MMLU (Massive Multitask Language Understanding)

![MMLU 示例](images/mmlu.png)

- **发布年份**: 2020 (GPT-3 发布后不久)
- **内容**: 57 个学科, 全部为多选题
- **数据来源**: "由研究生和本科生从网上免费资源收集"
- **当时评估方式**: GPT-3 Few-shot Prompting
- **当时 SOTA**: GPT-3 约 **45%**

> **讲师吐槽**: 尽管名字叫"语言理解", 但它更像是在测试**知识记忆** (Knowledge) 而非语言能力. 我的语言理解能力不错, 但我可能做不好 MMLU, 因为很多都是我不知道的事实 (如外交政策).

**Few-shot Prompt 格式**:

```
The following are multiple choice questions (with answers) about [subject].

[Example 1]
...
Answer: A

[Example 2]
...
Answer: C

[Actual Question]
...
Answer:
```

**HELM 可视化**: [HELM MMLU](https://crfm.stanford.edu/helm/mmlu/latest/) 允许你查看各模型在各子任务上的表现, 并深入到具体问题和预测.

**当前状态**: 顶级模型 (Claude, O3) 已达 **90%+**. 已被认为接近**饱和**, 可能被过度优化.

> **重要区分**: MMLU 最初设计用于评估**基础模型** (Base Model), 而现在常用于评估**指令微调模型**. 对于基础模型, 如果它在没有专门训练的情况下就能做好 MMLU, 说明它具有良好的泛化能力. 但如果你专门针对这 57 个学科进行微调, 高分可能只反映了对基准的过度拟合, 而非真正的通用智能.

### 4.2 MMLU-Pro

![MMLU-Pro 示例](images/mmlu-pro.png)

- **发布年份**: 2024
- **改进**:
  - 移除了 MMLU 中的噪声/简单问题
  - 选项从 4 个增加到 **10 个**
  - 使用 **Chain-of-Thought** 评估 (给模型更多思考机会)
- **结果**: 模型准确率下降 **16%-33%** (不再饱和)

**HELM 可视化**: [HELM MMLU-Pro](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro)

### 4.3 GPQA (Graduate-Level Google-Proof Q&A)

![GPQA 数据创建流程](images/gpqa.png)

- **发布年份**: 2023
- **特点**:
  - 由 **61 位 PhD 承包商** (来自 Upwork) 撰写问题
  - 经过专家验证、非专家测试的多阶段流程
  - **Google-Proof**: 非专家用 Google 搜索 30 分钟也只能达到 ~34% 准确率
- **PhD 专家准确率**: **65%**
- **当时 SOTA (GPT-4)**: **39%**
- **当前 SOTA (O3)**: **~75%** (一年内提升显著!)

**HELM 可视化**: [HELM GPQA](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa)

> **课堂问答**:
>
> - Q: GPQA 是 Google-Proof, 但怎么知道 O3 没有偷偷调用互联网?
> - A: 需要使用明确禁用搜索的 API 端点, 并信任这是真的.

### 4.4 HLE (Humanity's Last Exam)

![HLE 示例](images/hle-examples.png)

- **发布年份**: 2025
- **特点**:
  - **多模态** (包含图像)
  - 2500 个问题, 多学科, 多选题 + 简答题
  - **$500K 奖金池 + 论文署名**激励出题
  - 由前沿 LLM 过滤"太简单"的问题
  - 多阶段审核

![HLE 数据流程](images/hle-pipeline.png)

![HLE 结果](images/hle-results.png)

- **当前 SOTA (O3)**: **~20%**
- **最新排行榜**: [agi.safe.ai](https://agi.safe.ai/)

> **课堂批评**: 公开征集问题会带来**偏见**——响应者往往是那些已经非常熟悉 LLM 的人, 可能出的题目非常"LLM-adversarial" (刻意针对 LLM 弱点).

---

## 5. 指令遵循基准 (Instruction Following Benchmarks)

到目前为止, 我们评估的都是结构化任务 (多选题/简答题). 但 ChatGPT 以来, **指令遵循** (Instruction Following) 成为核心能力: 用户直接描述任务, 模型执行.

**核心挑战**: 如何评估**开放式回复**?

### 5.1 Chatbot Arena

![Chatbot Arena 排行榜](images/chatbot-arena-leaderboard.png)

**工作原理**:

1. 网络用户输入 Prompt
2. 获得两个**匿名**模型的回复
3. 用户选择哪个更好
4. 基于配对排名计算 **ELO 评分**

**优点**:

- **动态 (Live)**: 不是静态基准, 始终有新数据流入
- **易于添加新模型**: ELO 系统天然支持

**问题与争议**:

- **The Leaderboard Illusion** (论文): 揭示了某些模型提供者获得了**特权访问权限** (多次提交), 协议存在问题
- **用户分布偏见**: "网络上随机的人"代表什么分布?

**[深入探讨: Chatbot Arena 与 ELO 评分系统](./Lecture12-Chatbot-Arena.md)**

### 5.2 IFEval (Instruction-Following Eval)

![IFEval 约束类型](images/ifeval-categories.png)

- **工作原理**: 给指令添加**合成约束** (如"回答不超过 10 个词", "必须包含某个词", "使用特定格式")
- **优点**: 约束可以**自动验证** (用简单脚本)
- **缺点**: 只评估是否遵循约束, **不评估回复的语义质量**; 约束有些人为造作

**HELM 可视化**: [HELM IFEval](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/ifeval)

### 5.3 AlpacaEval

![AlpacaEval 排行榜](images/alpacaeval-leaderboard.png)

- **内容**: 805 条来自不同来源的指令
- **指标**: 由 **GPT-4 Preview 作为裁判**, 计算相对于 GPT-4 Preview 的**胜率**
- **潜在偏见**: GPT-4 作为裁判可能偏向自己

**历史趣闻**: 早期版本被 Gaming——一些小模型通过生成更长的回复获得高分, 因为 GPT-4 偏好长回复. 后来引入了**长度校正**版本.

**相关性**: 与 Chatbot Arena 相关性较高 (提供类似信息, 但更自动化/可复现).

### 5.4 WildBench

![WildBench](images/wildbench.png)

- **数据来源**: 从 100 万真实人机对话中抽取 1024 个示例
- **评估方式**: GPT-4 Turbo 作为裁判, 使用 **Checklist** (类似 CoT for Judging) 确保评估全面
- **与 Chatbot Arena 相关性**: **0.95** (非常高)

**HELM 可视化**: [HELM WildBench](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench)

> **有趣观察**: 评估基准的"评估" (Evaluation of Evaluation) 似乎是看它与 Chatbot Arena 的相关性.

---

## 6. Agent 基准 (Agent Benchmarks)

许多任务需要**工具使用** (如运行代码、访问互联网) 和**多轮迭代**. 这就是 **Agent** 登场的地方.

**Agent = 语言模型 + Agent 脚手架** (决定如何调用 LM 的程序逻辑)

### 6.1 SWE-Bench

![SWE-Bench](images/swebench.png)

- **任务**: 给定代码库 + GitHub Issue 描述, 提交一个 **Pull Request**
- **数据**: 12 个 Python 仓库的 2294 个任务
- **评估指标**: **单元测试是否通过**

**流程**: Issue 描述 → Agent 阅读代码 → 生成 Patch → 运行测试

### 6.2 CyBench (网络安全)

![CyBench 任务](images/cybench.png)

- **任务**: **Capture The Flag (CTF)** 风格的渗透测试
- **内容**: 40 个 CTF 任务
- **难度衡量**: 使用人类**首次解决时间 (First-Solve Time)** 作为难度指标

![CyBench Agent 架构](images/cybench-agent.png)

**Agent 架构** (标准模式):

1. LM 思考并制定计划
2. 生成命令
3. 执行命令
4. 更新 Agent 记忆
5. 迭代直到成功或超时

![CyBench 结果](images/cybench-results.png)

- **当前准确率**: ~20%
- **亮点**: O3 能解决人类团队需要 42 分钟才能解决的问题 (最长挑战需 24 小时)

**[深入探讨: Agent 基准: SWE-Bench 与 CyBench](./Lecture12-Agent-Benchmarks.md)**

### 6.3 MLE-Bench (Kaggle)

![MLE-Bench 任务](images/mlebench.png)

- **任务**: 75 个 Kaggle 竞赛, Agent 需要编写代码、训练模型、调参、提交
- **评估**: 是否获得奖牌 (某个性能阈值)

![MLE-Bench 结果](images/mlebench-results.png)

- **当前准确率**: 即使最好的模型, 获得任何奖牌的准确率也 **< 20%**

---

## 7. 纯推理基准 (Pure Reasoning Benchmarks)

之前所有任务都需要**语言知识**和**世界知识**. 能否将**推理**从知识中分离出来?

> **论点**: 推理捕捉了一种更纯粹的"智能"形式, 不仅仅是记忆事实.

### 7.1 ARC-AGI

**ARC-AGI** 由 François Chollet 于 2019 年提出 (早于当前 LLM 浪潮).

**任务**: 给定输入输出模式, 推断规则并填充测试案例.

**特点**:

- **无语言**, 纯视觉模式识别
- 设计为人类容易、机器困难 (与传统基准相反)

**ARC-AGI-1 结果**:

- 传统 LLM (GPT-4o): ≈ **0%**
- O3: ≈ **75%** (使用大量计算, 每个任务可能花费数百美元)

**ARC-AGI-2**: 更难, 目前准确率仍然很低.

> **ARC Prize 网站**: [arcprize.org](https://arcprize.org/arc-agi)

---

## 8. 安全基准 (Safety Benchmarks)

在汽车行业有碰撞测试, 食品行业有卫生评级. **AI 的安全评估应该是什么样的?**

目前没有明确答案——AI 还太早期, 人们还在探索"安全"意味着什么.

### 8.1 HarmBench

- **内容**: 510 种违反法律或规范的**有害行为**
- **评估**: 模型是否拒绝执行有害指令

**HELM 可视化**: [HELM HarmBench](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench)

> **观察**: 不同模型拒绝率差异很大. 例如, 某些模型 (如 DeepSeek V3) 在某些有害请求上遵从率较高.

### 8.2 AIR-Bench

- **特点**: 将"安全"概念锚定在**法规框架和公司政策**上
- **内容**: 基于法律和政策构建 314 个风险类别, 5694 个 Prompt
- **优点**: 更有依据 (Grounded), 而非任意定义"安全"

**HELM 可视化**: [HELM AIR-Bench](https://crfm.stanford.edu/helm/air-bench/latest/)

### 8.3 越狱 (Jailbreaking)

语言模型被训练拒绝有害指令, 但可以被**绕过**.

**GCG Attack (Greedy Coordinate Gradient)**:

![GCG 攻击示例](images/gcg-examples.png)

- **方法**: 自动优化 Prompt 后缀以绕过安全机制
- **惊人发现**: 在开源模型 (Llama) 上优化的后缀可以**迁移**到闭源模型 (GPT-4)

> **意义**: 即使模型表面上安全, 越狱攻击表明底层**能力**仍然存在 (并可能被释放).

### 8.4 安全 vs 能力: 一个复杂的关系

**两个维度**:

- **能力 (Capability)**: 模型是否*能够*做某事
- **倾向 (Propensity)**: 模型是否*愿意*做某事

**API 模型**: 只需控制 Propensity (可以拒绝)
**开源模型**: Capability 也重要, 因为安全措施可以通过**微调轻松移除**

**双重用途 (Dual-Use)**: CyBench 是安全评估还是能力评估?

- 恶意: 用 Agent 黑入系统
- 善意: 用 Agent 进行渗透测试保护系统

### 8.5 预部署测试

美国和英国的 AI 安全研究所与模型开发者 (Anthropic, OpenAI 等) 建立了**自愿协议**:

- 公司在发布前给安全研究所早期访问权限
- 安全研究所运行评估并生成报告
- 目前是**自愿的**, 无法律强制力

---

## 9. 真实性 (Realism)

语言模型在实践中被大量使用:

- OpenAI: 每天 1000 亿 Token
- Cursor: 10 亿行代码

![OpenAI 日流量](images/openai-100b-tokens.png)

然而, 大多数基准 (如 MMLU) 与**真实使用场景**相去甚远.

### 9.1 两种 Prompt

1. **Quizzing (测试)**: 用户**知道答案**, 试图测试系统 (如标准化考试)
2. **Asking (询问)**: 用户**不知道答案**, 试图使用系统获取信息

**Asking** 更真实, 能为用户带来价值. 标准化考试显然不够真实.

### 9.2 Clio (Anthropic)

![Clio 分析表](images/clio-table4.png)

Anthropic 使用语言模型分析**真实用户数据**, 揭示人们实际使用 Claude 的方式. **编码**是最常见的用途之一.

> **意义**: 一旦部署系统, 你就有了真实数据, 可以在真实用例上评估.

### 9.3 MedHELM

以往的医疗基准基于标准化考试. **MedHELM** 不同:

- 从 **29 位临床医生**处征集 **121 个临床任务**
- 混合公开和私有数据集

**HELM 可视化**: [MedHELM](https://crfm.stanford.edu/helm/medhelm/latest/)

> **权衡**: 真实性和隐私有时是矛盾的. 真实的医疗数据涉及患者隐私.

---

## 10. 有效性 (Validity)

我们如何知道评估是**有效的**?

### 10.1 训练-测试重叠 (Train-Test Contamination)

**机器学习 101**: 不要在测试集上训练!

- **Pre-基础模型时代 (ImageNet, SQuAD)**: 有明确定义的训练/测试分割
- **如今**: 在互联网上训练, 不告诉你数据是什么

**应对策略**:

**路线 1: 从模型行为推断重叠**

![污染检测方法](images/contamination-exchangeability.png)

利用数据点的**可交换性** (Exchangeability): 如果模型对测试集中某个特定顺序表现出偏好 (与数据集顺序相关), 则可能是训练过了.

**路线 2: 鼓励报告规范**

就像论文应报告置信区间一样, 模型发布者应报告**训练-测试重叠检测结果**.

**[深入探讨: 训练-测试污染 (Train-Test Contamination)](./Lecture12-Contamination.md)**

### 10.2 数据集质量

许多基准存在**错误**:

- **SWE-Bench Verified**: OpenAI 修复了 SWE-Bench 中的一些错误
- **Platinum 版本**: 创建高质量标注的"白金版"基准

> **影响**: 如果你看到 MATH/GSM8K 准确率 90%+, 并认为问题很难, 实际上可能有一半是标签噪声. 修复后分数会上升.

---

## 11. 我们到底在评估什么?

换句话说: **游戏规则是什么?**

| 时代                   | 评估对象                             | 规则                              |
| ---------------------- | ------------------------------------ | --------------------------------- |
| **Pre-基础模型** | **方法** (Methods)             | 标准化训练/测试分割, 比较学习算法 |
| **如今**         | **模型/系统** (Models/Systems) | Anything goes                     |

**例外 (鼓励算法创新)**:

![NanoGPT Speedrun](images/karpathy-nanogpt-speedrun.png)

- **NanoGPT Speedrun**: 固定数据, 最小化达到特定验证 Loss 的时间
- **DataComp-LM**: 给定原始数据集, 使用标准训练流程获得最佳准确率 (比较**数据选择**策略)

**关键点**: 无论评估什么, 都需要**明确定义游戏规则**!

---

## 12. 总结: 核心要点

1. **没有唯一正确的评估**: 根据你要测量的内容选择评估方式.
2. **始终查看具体实例和预测**: 不要只看数字, 深入到具体问题.
3. **评估有多个维度**: 能力、安全、成本、真实性.
4. **明确游戏规则**: 你在评估**方法**还是**模型/系统**?

---

## 附录: 配套代码结构 (`lecture_12.py`)

课程配套代码 `lecture_12.py` 定义了本讲的完整结构:

```python
def main():
    text("**Evaluation**: given a **fixed model**, how \"**good**\" is it?")
    what_you_see()              # 1. 你所看到的 (基准分数、排行榜、氛围)
    how_to_think_about_evaluation()  # 2. 如何思考评估
    perplexity()                # 3. 困惑度评估
    knowledge_benchmarks()      # 4. 知识基准 (MMLU, GPQA, HLE)
    instruction_following_benchmarks()  # 5. 指令遵循基准
    agent_benchmarks()          # 6. Agent 基准
    pure_reasoning_benchmarks() # 7. 纯推理基准 (ARC-AGI)
    safety_benchmarks()         # 8. 安全基准
    realism()                   # 9. 真实性
    validity()                  # 10. 有效性
    what_are_we_evaluating()    # 11. 我们在评估什么
    # 总结...
```

代码中引用的关键图片和链接已在本笔记中嵌入.

---

## 参考链接

- **HELM**: https://crfm.stanford.edu/helm/
- **Chatbot Arena**: https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard
- **Artificial Analysis**: https://artificialanalysis.ai/
- **OpenRouter Rankings**: https://openrouter.ai/rankings
- **ARC Prize**: https://arcprize.org/
- **HLE Leaderboard**: https://agi.safe.ai/
