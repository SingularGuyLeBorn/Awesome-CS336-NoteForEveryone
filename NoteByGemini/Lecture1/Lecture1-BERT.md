# 专题: BERT (Bidirectional Encoder Representations from Transformers)
## 1. 核心贡献
BERT 是 Google AI 在 2018 年的论文《BERT: Pre-training of Deep Bidirectional Representations for Language Understanding》中提出的一个里程碑式的**[语言模型](./Lecture1-Language-Models.md)**. 它的出现彻底改变了自然语言处理(NLP)的研究范式,开启了“预训练-微调”(Pre-training and Fine-tuning)的新时代. 
BERT 的核心贡献在于: **通过创新的预训练任务,成功地训练了一个深度的、双向的 Transformer 表示模型,使其能够捕捉到丰富的上下文信息. **
与之前的模型(如 GPT-1 是单向的,ELMo 是浅层拼接的双向)不同,BERT 真正实现了在模型的每一层都能同时“看到”句子的左右两边的上下文,这对于语言理解任务至关重要. 
## 2. 模型/方法概述
### 2.1 架构: 仅编码器
BERT 的模型架构采用了 **[Transformer](./Lecture1-Transformer.md)** 的编码器(Encoder)部分. 它有两个主要版本: 
*   **BERT-Base:** 12 层,768 隐藏单元,12 个注意力头,总计 1.1 亿参数. 
*   **BERT-Large:** 24 层,1024 隐藏单元,16 个注意力头,总计 3.4 亿参数. 
### 2.2 预训练任务
为了实现深度的双向表示,BERT 设计了两个巧妙的预训练任务: 
1.  **掩码语言模型 (Masked Language Model, MLM):**
    *   **思想:** 传统的语言模型是单向的(从左到右预测下一个词),无法直接用于训练双向模型(否则模型就能“作弊”看到要预测的词). 为了解决这个问题,MLM 任务随机地“掩码”(mask)掉输入句子中 15% 的 token,然后让模型去预测这些被掩码的 token 的原始值. 
    *   **具体实现:**
        *   80% 的概率,被选中的 token 替换为 `[MASK]`.  (e.g., `my dog is hairy` -> `my dog is [MASK]`)
        *   10% 的概率,被选中的 token 替换为一个随机的词.  (e.g., `my dog is hairy` -> `my dog is apple`)
        *   10% 的概率,保持不变.  (e.g., `my dog is hairy` -> `my dog is hairy`)
    *   **作用:** 这个任务迫使模型必须依赖左右双向的上下文来推断被掩码的词,从而学习到深度的、与上下文相关的词语表示. 
2.  **下一句预测 (Next Sentence Prediction, NSP):**
    *   **思想:** 为了让模型能够理解句子与句子之间的关系(这对于问答、自然语言推断等任务很重要),NSP 任务让模型判断两个句子 A 和 B 是否是连续的. 
    *   **具体实现:**
        *   50% 的情况下,句子 B 是句子 A 的真实下一句. 
        *   50% 的情况下,句子 B 是从语料库中随机选择的一个句子. 
    *   **作用:** 模型通过这个任务学习句子级别的表示. 
### 2.3 微调 (Fine-tuning)
预训练完成后,BERT 可以通过在特定的下游任务数据上进行微调来适应各种任务. 只需在 BERT 的输出之上添加一个小的、任务特定的分类层,然后用少量标注数据端到端地训练整个模型即可. 这种范式极大地简化了 NLP 任务的开发流程,并刷新了 11 项 NLP 任务的 SOTA (State-of-the-Art) 记录. 
## 3. 影响力与局限性
*   **影响力:**
    *   **开启了预训练时代:** BERT 确立了“大规模无监督预训练 + 下游任务微调”的范式,至今仍是 NLP 领域的主流. 
    *   **成为 NLP 的基石:** BERT 及其变体(如 RoBERTa, ALBERT, DeBERTa)在很长一段时间内都是各种 NLP 应用的核心组件,也是学术研究的强大基线. 
    *   **推动了模型规模的竞赛:** BERT-Large 的成功证明了更大模型的潜力,间接推动了后续 **[GPT-3](./Lecture1-GPT-4.md)** 等巨型模型的出现. 
*   **局限性:**
    *   **预训练与微调的差异:** MLM 任务中使用的 `[MASK]` token 在微调阶段是不会出现的,这造成了训练和使用时的不一致. 
    *   **计算效率:** 对每个被掩码的 token 进行独立预测,而不是像自回归模型那样一次性预测,计算效率相对较低. 
    *   **NSP 任务的有效性:** 后续研究(如 RoBERTa)发现 NSP 任务可能不是必要的,甚至可能对性能有害,并提出了更有效的预训练目标. 
    *   **不擅长生成任务:** 作为仅编码器模型,BERT 的设计天然不适合做自由形式的文本生成. 
---
**关键论文:** [BERT: Pre-training of Deep Bidirectional Representations for Language Understanding](https://arxiv.org/abs/1810.04805)