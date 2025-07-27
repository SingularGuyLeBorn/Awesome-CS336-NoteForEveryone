# 专题：Tokenization (令牌化)
## 1. 核心定义
**Tokenization (令牌化)** 是自然语言处理(NLP)中的一个基础且关键的步骤. 它的核心任务是：**将人类可读的原始文本(一个字符串)分割成一个由更小的单元(称为 token 或令牌)组成的序列,然后将这些 token 转换为模型可以处理的数字 ID. **
这个过程包含两个主要阶段：
1.  **分割 (Segmentation):** 将字符串拆分成一个子串列表. 
2.  **编码 (Encoding):** 使用一个“词汇表”(Vocabulary)将每个子串映射到一个唯一的整数 ID. 
这个过程必须是可逆的,即能够通过一个解码(Decoding)过程,将整数 ID 序列完美地恢复成原始的文本字符串. 
## 2. 为何需要 Tokenization？
神经网络模型,如 **[Transformer](./Lecture1-Transformer.md)**,无法直接处理原始文本. 它们需要输入的格式是定长的、数值化的向量. Tokenization 正是连接原始文本和模型输入的桥梁. 一个好的 Tokenizer 需要在以下几个方面取得平衡：
*   **词汇表大小 (Vocabulary Size):** 词汇表不能太小(否则一个词会被拆成太多 token,序列变长),也不能太大(否则会增加模型末端 softmax 层的计算负担,并可能包含大量罕用词). 
*   **序列长度 (Sequence Length):** **[注意力机制](./Lecture1-Self-Attention.md)**的计算复杂度与序列长度的平方成正比. 因此,一个好的 Tokenizer 应该有较高的**压缩率**(即每个 token 代表的平均字符/字节数),以产生更短的 token 序列. 
*   **处理未登录词 (Out-of-Vocabulary, OOV):** Tokenizer 必须能够优雅地处理在训练词汇表时从未见过的词语或字符. 
## 3. 主流 Tokenization 方法
### 3.1 基于词 (Word-based)
*   **方法:** 使用空格或标点符号作为分隔符来切分单词. 
*   **优点:** 符合人类直觉,每个 token 都是一个有意义的词. 
*   **缺点:**
    *   **巨大的词汇表:** 语言中的词汇几乎是无限的(如 "loving", "loved", "lover" 都会被视为不同的词). 
    *   **严重的 OOV 问题:** 无法处理拼写错误、新造词或罕见词. 所有这些词都必须被映射到一个特殊的 `<UNK>` (unknown) token,导致信息丢失. 
### 3.2 基于字符 (Character-based)
*   **方法:** 将文本切分成单个字符. 
*   **优点:**
    *   **无 OOV 问题:** 任何文本都可以由有限的字符集组成. 
    *   **词汇表小:** 英语字母、数字和符号加起来只有几百个. 
*   **缺点:**
    *   **序列过长:** "hello" 会被分成 5 个 token,导致计算效率极低. 
    *   **破坏语义单元:** 单个字符通常不具备独立的语义,模型需要耗费大量精力去学习如何从字符组合成词. 
### 3.3 子词算法 (Subword Algorithms)
子词算法是目前的主流方法,它完美地平衡了上述方法的优缺点. 其核心思想是：**常见词用单个 token 表示,而罕见词则被拆分成多个有意义的子词单元. **
*   **[字节对编码 (Byte-Pair Encoding, BPE)](./Lecture1-Byte-Pair-Encoding.md):** 这是最经典和最常用的子词算法. 它通过迭代地合并语料库中最高频的相邻 token 对来构建词汇表. 
*   **WordPiece:** 由 Google **[BERT](./Lecture1-BERT.md)** 使用. 与 BPE 类似,但它合并 token 对的标准是看合并后是否能最大化训练数据的似然(likelihood),而不是原始频率. 
*   **Unigram Language Model:** 由 Google T5 和 SentencePiece 库使用. 它从一个大的初始词汇表开始,然后根据概率模型,逐步移除那些对整体数据似然影响最小的 token,直到达到期望的词汇表大小. 
## 4. 现代 Tokenizer 的实现细节
现代 Tokenizer (如 Hugging Face 的 `tokenizers` 库) 通常是一个包含多个阶段的流水线：
1.  **归一化 (Normalization):** 进行文本的初步清理,如转换为小写、处理 Unicode 兼容性字符(NFC/NFD)等. 
2.  **预切分 (Pre-tokenization):** 根据规则(如正则表达式)将文本初步分割成“单词”块. 例如,**[GPT-2](./Lecture1-GPT-4.md)** 的 Tokenizer 会根据空格和标点来切分. 
3.  **模型 (Model):** 在每个预切分的块上应用核心的子词算法(如 **[BPE](./Lecture1-BPE.md)**). 
4.  **后处理 (Post-processing):** 添加特殊的 token,如 `[CLS]`, `[SEP]` 等,以符合特定模型(如 **[BERT](./Lecture1-BERT.md)**)的输入格式要求. 
正如 Andrej Karpathy 所说,Tokenizer 是“**[语言模型](./Lecture1-Language-Models.md)**的第一个层,也是唯一一个用 C++/Rust 实现并且不可训练的层”. 它虽然不起眼,但对模型的性能、效率和行为有着至关重要的影响. 