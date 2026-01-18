# Lecture 13: 训练数据策略 (Training Data Strategy)

**课程**: CS336 · **主题**: 预训练、中期训练、后期训练的数据来源与处理

---

## 0. 课程开场: 数据是最重要的

> **讲师 Hot Take**: 数据是训练语言模型最重要的事情.

之前的讲座讨论了**给定数据**如何训练模型 (架构、优化器、Tokenization、Scaling Laws、并行计算). 现在, 我们讨论**训练什么数据**.

**为什么数据最重要?** 看看公司实际披露什么:

![Llama 3 数据描述 (几乎没有信息)](images/llama3-data.png)

即使是开源模型 (如 Llama 3, DeepSeek), 也完全披露架构甚至训练细节, 但**关于数据几乎什么都不说**:

> "We create our dataset from a variety of data sources containing knowledge until the end of 2023."

**保密原因**:

1. **竞争动态**: 数据是核心竞争力
2. **版权责任**: 不想被起诉

---

## 1. 训练阶段概览

数据工作贯穿训练的各个阶段, 但侧重点不同:

| 阶段                               | 数据特点                                  | 目标               |
| ---------------------------------- | ----------------------------------------- | ------------------ |
| **Pre-training (预训练)**    | 大量低质量原始数据 (通常来自网络)         | 获得广泛的语言能力 |
| **Mid-training (中期训练)**  | 较小量高质量数据 (如数学、代码、长上下文) | 增强特定能力       |
| **Post-training (后期训练)** | 指令跟随数据、对话数据、RLHF              | 使模型可对话、安全 |

**术语**:

- **Base Model (基础模型)**: Pre-training + Mid-training 后的模型
- **Instruct Model (指令模型)**: Post-training 后的模型

### 1.1 示例: OLMo (AI2)

**Pre-training**:

![OLMo Pre-training 数据混合](images/olmo2-pretraining.png)

- DCLM Baseline: 3.7T tokens (主体)
- 代码、学术论文、数学、Wikipedia

**Mid-training**:

![OLMo Mid-training 数据混合](images/olmo2-dolmino.png)

- 仍然有 DCLM Baseline, 但从 3.7T 过滤到 700B
- 新增合成数据, 甚至 GSM8K 训练集

**Post-training**:

![Tulu Post-training 数据](images/tulu.png)

- 各种来源的对话数据
- 大量合成数据

> **关键洞察**: 从"大量低质量"到"少量高质量", 界限模糊但趋势明确.

---

## 2. 预训练数据: 历史演进

### 2.1 BERT (2018): Books + Wikipedia

**BooksCorpus**:

- 来源: Smashwords (2008 年成立的自出版平台)
- 内容: 7,000 本免费自出版书籍
- 现状: 因违反服务条款已被下架

**Wikipedia**:

- 2001 年成立, 目前 6200 万篇文章, 329 种语言
- **不包含原创思想**: 无观点、无个人网页
- **基于可查证性**: 需要可靠来源
- **定期 Dump**: 每隔几周提供完整数据下载

> **数据投毒警告**: 攻击者可以在 Wikipedia Dump 前注入恶意编辑, 在回滚前被收录. 这已被用于操纵语言模型的情感判断 (如对"iPhone"产生负面情绪).

### 2.2 GPT-2 (2019): WebText

**核心思路**: Web 很大但质量低, 如何快速获得高质量子集?

**方法**: 利用 Reddit 作为"质量过滤器"

- 收集 Reddit 帖子中 **karma ≥ 3** 的外链
- 结果: 800 万页面, 40GB 文本

**OpenWebText**: WebText 的开源复制版本

### 2.3 Common Crawl: 学术界的"互联网"

**[深入探讨: Common Crawl 与网络爬虫](./Lecture13-Common-Crawl.md)**

**基本概况**:

- 2007 年成立的非营利组织
- 每月进行一次网络爬虫, 至今已有 ~100 次
- 最新爬虫: 2025 年 4 月

**两种格式**:

- **WARC**: 原始 HTTP 响应 (如 HTML)
- **WET**: 转换为纯文本 (有损过程)

**HTML → 文本转换器**的差异:

![DCLM: HTML 转换器对准确率的影响](images/dclm-wet.png)

使用 **trafilatura** 比使用 WET 文件高 **4 个百分点**!

> **注意**: Common Crawl **不是完整的互联网**! 它刻意保守和礼貌. 甚至不是所有 Wikipedia 文章都在 Common Crawl 中.

### 2.4 CCNet (2019): 用 Wikipedia 过滤 Common Crawl

**目标**: 自动构建大规模高质量多语言数据集

**流程**:

1. **去重**: 基于轻量级规范化移除重复段落
2. **语言识别**: fastText 分类器, 只保留目标语言
3. **质量过滤**: 保留在 **KenLM 5-gram 模型**下看起来像 Wikipedia 的文档

**关键洞察**: Wikipedia 作为"高质量"的代理. 但 Wikipedia 不覆盖所有内容, 这个方法也不会.

### 2.5 T5 / C4 (2019): 规则过滤

**C4 (Colossal Clean Crawled Corpus)**:

- 从一个 Common Crawl 快照 (2019 年 4 月) 开始: 1.4T tokens
- **纯规则过滤** (无模型):
  - 保留以标点结尾、≥5 词的行
  - 移除 < 3 句的页面
  - 移除包含"坏词"的页面
  - 移除包含 `{` 的页面 (移除代码!)
  - 只保留英语 (概率 ≥ 0.99)
- 结果: 806 GB (156B tokens)

**规则 vs 模型过滤的权衡**:

- **规则**: 更广泛 (非 Wikipedia 风格的好句子也能保留), 但可能包含垃圾
- **模型**: 更精准, 但只能复制"正例"的分布

### 2.6 GPT-3 (2020): 多源混合

- Common Crawl (处理后)
- WebText2 (WebText 的扩展)
- 神秘的书籍语料库 (Books1, Books2)
- Wikipedia
- **总计**: 400B tokens

**Common Crawl 处理**: 训练质量分类器, 区分 {WebText, Wikipedia, Books} 与其余.

### 2.7 The Pile (2021): 社区驱动

为了对抗 GPT-3 的封闭, EleutherAI 社区在 Discord 上协调, 众包构建了 **22 个高质量领域**:

- Pile-CC (Common Crawl, 使用 WARC + jusText)
- OpenWebText
- Wikipedia, arXiv
- **PubMed Central**: 500 万篇论文 (NIH 资助的必须公开)
- **Enron Emails**: 50 万封邮件 (来自 2002 年调查, 因为**几乎没有其他公开邮件数据集**)
- **Project Gutenberg**: 7.5 万本公版书
- **Books3**: 19.6 万本书 (来自影子库, 因版权问题已下架)
- **Stack Exchange**: QA 风格, 接近指令遵循
- **GitHub**: 代码

**结果**: 825 GB (~275B tokens)

### 2.8 MassiveText / Gopher (2021): 规则主导

**MassiveWeb** 过滤:

- 只保留英语
- **手动规则**过滤 (如: 80% 的词至少包含一个字母字符)
- **Google SafeSearch** 过滤毒性 (非词表)

> **当时的理由**: 避免弱模型的偏见. 但这个范式后来被 DCLM 打破.

**结果**: 10.5 TB 文本, 但 Gopher 只训练了 300B tokens (12%)

### 2.9 LLaMA (2022): 综合方案

- Common Crawl + CCNet (分类器: 是否被 Wikipedia **引用**)
- C4
- GitHub (保留宽松许可)
- Wikipedia, Project Gutenberg, **Books3** (惹上大麻烦!)
- arXiv, Stack Exchange
- **总计**: 1.2T tokens

**复制版本**:

- **RedPajama v1** (Together): 开源复制
- **SlimPajama** (Cerebras): 去重后的 627B 子集

### 2.10 RefinedWeb (2023): Web Data is All You Need

**论点**: 如果过滤做得好, **只需要网络数据**.

**方法**:

- trafilatura 提取 (WARC 而非 WET)
- Gopher 规则过滤, **避免 ML 过滤**以避免偏见
- MinHash 模糊去重
- **结果**: 5T tokens (发布 600B)

**FineWeb** (HuggingFace): RefinedWeb 的改进版

- 95 个 Common Crawl 快照
- Gopher + C4 规则
- PII 匿名化
- **结果**: 15T tokens (仍是**轻度过滤**, 适合进一步模型过滤)

### 2.11 Dolma (2024): AI2 的综合数据集

- Common Crawl (语言识别 + 规则过滤 + 去重)
- Reddit (Pushshift 项目)
- PeS2o: 4000 万篇学术论文 (Semantic Scholar)
- C4, Project Gutenberg, Wikipedia
- **结果**: 3T tokens

### 2.12 DCLM (2024): 模型过滤的胜利

**[深入探讨: DCLM 与模型基质量过滤](./Lecture13-DCLM.md)**

**DataComp-LM** 的目标是创建一个**数据处理算法的标准竞赛**.

**DCLM-pool**: 处理所有 Common Crawl → **240T tokens**

**DCLM-baseline**: 使用质量分类器过滤 → **3.8T tokens** (只保留 1.4%!)

![DCLM 过滤流程](images/dclm-filter.png)

**模型过滤方法**:

- **正例** (20 万): OpenHermes-2.5 (GPT-4 生成的指令数据) + ELI5 (Reddit 子版块)
- **负例** (20 万): RefinedWeb 随机样本
- 训练 **fastText 分类器**

![DCLM 质量分类器效果](images/dclm-quality.png)

> **关键转折**: 这打破了"避免 ML 过滤"的旧范式. 使用模型过滤**显著提升**下游任务表现.

### 2.13 Nemotron-CC (2024): 更多 Token

**问题**: DCLM 过滤太激进 (240T → 3.8T). 想要更多 Token!

**方法**:

1. **HTML → 文本**: 使用 jusText (而非 trafilatura), 因为保留更多 Token
2. **分类器集成**:
   - Nemotron-340B 评分教育价值, 蒸馏到快速模型
   - DCLM 分类器
   - 按分数分桶, 从每个桶采样 (保证覆盖)
3. **合成数据改写**:
   - 低质量数据: 用 LM 改写成高质量
   - 高质量数据: 用 LM 生成 QA 对 / 摘要 / 关键信息提取

**结果**: 6.3T tokens (高质量子集 1.1T)

![Nemotron-CC 效果](images/nemotron-results.png)

> **对比**: Llama 3 训练 15T, Qwen 3 训练 36T (含多模态).

---

## 3. 版权法与数据合法性

**[深入探讨: 版权法与 Fair Use](./Lecture13-Copyright.md)**

### 3.1 版权法基础

- **目的**: 激励知识产品的创造
- **范围**: "固定在任何有形表达媒介中的原创作品"
- **无需注册**: 你的网站已经是版权作品 (只是起诉前需要注册, $65)
- **期限**: 75 年后进入公版

> **关键**: 互联网上的**大多数内容都是版权作品**.

### 3.2 如何合法使用版权作品

**方式一: 获得许可 (License)**

- 签订合同 (如 Google-Reddit, OpenAI-Shutterstock)
- Creative Commons 许可 (如 Wikipedia, Khan Academy)

**方式二: 援引 Fair Use**

四个因素:

1. **使用目的**: 教育 > 商业, 变革性 > 复制性
2. **作品性质**: 事实性 > 虚构性
3. **使用量**: 片段 > 全部
4. **市场影响**: 不替代原作品

**LLM 训练的挑战**:

- 复制数据 (训练第一步) 本身**可能已违规**, 即使你什么都不做
- 可以论证 ML 训练是**变革性**的
- ML 系统关心的是**想法** (如停车标志), 而非**表达** (某张图的艺术选择)
- **但**: LLM 明显影响市场 (作家、艺术家)

### 3.3 服务条款 (Terms of Service)

即使有许可或 Fair Use, **服务条款可能施加额外限制**.

例: YouTube 的服务条款禁止下载视频, 即使视频本身是 Creative Commons.

---

## 4. 中期训练与后期训练

### 4.1 长上下文扩展 (Long Context)

**需求**:

- DeepSeek v3: 128K tokens
- Claude 3.5: 200K tokens
- Gemini 1.5 Pro: 1.5M tokens

**问题**: Transformer 与序列长度呈**二次方**关系, 预训练阶段不高效.

**解决**: 在 Mid-training 阶段添加长上下文能力

- **数据来源**: 书籍 (PG-19), 数学证明 (Proof-Pile)
- **技术**: Shifted sparse attention, Positional interpolation

### 4.2 任务/NLP 数据集

**思路**: 将传统 NLP 数据集转换为 Prompt 格式

**Super-Natural Instructions (2022)**:

- 1,600+ 任务, 社区贡献
- 微调 T5 → Tk-Instruct

**Flan (2022-2023)**:

- 1,800+ 任务
- Zero-shot, Few-shot, Chain-of-Thought 版本

> **问题**: Prompt 太模板化, 不够自然.

### 4.3 指令遵循与对话数据

**Alpaca (2023)**:

- 使用 **Self-Instruct** 从 text-davinci-003 生成 52K 示例
- 微调 LLaMA 7B

**Vicuna**:

- 使用 ShareGPT (用户分享的 ChatGPT 对话, 已废弃) 的 70K 对话
- 微调 LLaMA

**Baize**:

- GPT-3.5 自我对话 (以 Quora/StackOverflow 问题为种子)
- 111.5K 示例

**WizardLM**:

- **Evol-Instruct**: 让问题"进化"以增加难度/广度

**MAmmoTH2**:

- 从 Common Crawl 中用 fastText 识别"测验网站"
- 用 GPT-4/Mixtral 提取 QA 对
- 10M 指令

**OpenHermes 2.5**:

- 多个数据集的聚合
- 1M GPT-4 生成的示例

**Llama 2 Chat**:

- 27,540 条**人工标注**的高质量指令
- 声称优于使用数百万开源示例

**Llama-Nemotron Post-training (2024)**:

- 从公开数据集 (WildChat 等) 或合成生成 Prompt
- 使用 Llama, Mixtral, DeepSeek R1, Qwen 生成回复 (商业可用, 不像 GPT-4)
- 包含推理轨迹

---

## 5. 总结: 核心要点

1. **数据不会从天上掉下来**: 需要大量工作获取

   - **Live Service → Raw Dump → Processed Data**
   - 涉及转换、过滤、去重
2. **数据是区分语言模型的关键**: 架构已趋同, 数据决定质量
3. **法律和伦理问题**: 版权、隐私、服务条款
4. **目前一切都是启发式的**: 大量机会改进!

---

## 附录: 配套代码结构 (`lecture_13.py`)

课程代码定义了完整的讲座结构:

```python
def main():
    introduction()              # 数据最重要
    # Pre-training
    bert()                      # Wikipedia + Books (2019)
    gpt2_webtext()              # Reddit 链接 (2019)
    common_crawl()              # 网络爬虫
    ccnet()                     # Wikipedia 过滤 (2019)
    t5_c4()                     # 规则过滤 (2019)
    gpt3()                      # 多源混合 (2020)
    the_pile()                  # 社区众包 (2021)
    gopher_massivetext()        # 规则过滤 (2021)
    llama()                     # 综合方案 (2022)
    refinedweb()                # Web Only (2023)
    dolma()                     # AI2 综合 (2024)
    dclm()                      # 模型过滤 (2024)
    nemotron_cc()               # 更多 Token (2024)
    copyright()                 # 版权法
    # Mid/Post-training
    long_context()              # 长上下文
    tasks()                     # NLP 任务转换
    instruction_chat()          # 指令/对话数据
```

---

## 参考链接

- **Common Crawl**: https://commoncrawl.org/
- **DCLM**: https://arxiv.org/abs/2406.11794
- **FineWeb**: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- **The Pile**: https://arxiv.org/abs/2101.00027
- **Nemotron-CC**: https://arxiv.org/abs/2412.xxxxx (待确认)
- **CS324 版权笔记**: https://stanford-cs324.github.io/winter2022/lectures/legality/
