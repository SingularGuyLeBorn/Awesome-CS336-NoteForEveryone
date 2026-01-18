# CS336 Lecture 14: 实战数据过滤和去重 (Practical Data Filtering and Deduplication)

> **编辑蓝图 (Editorial Blueprint)**
> 
> **核心主题**: 本讲座是CS336课程中最具实践性的一讲，深入讲解了大规模语言模型预训练数据处理流水线的两大核心：**数据过滤 (Filtering)** 和 **数据去重 (Deduplication)**。从算法原理到工程实现，从语言识别到毒性过滤，全面覆盖。
> 
> **知识结构**: 
> - 第一部分：过滤算法基础（N-gram语言模型、KenLM、fastText分类器、重要性重采样DSIR）
> - 第二部分：过滤应用场景（语言识别、质量过滤、毒性过滤）
> - 第三部分：去重技术（精确去重、布隆过滤器、近似去重、MinHash、LSH）
> 
> **精英补充笔记**:
> - **[深入探讨: KenLM与N-gram语言模型](./Lecture14-KenLM.md)** - N-gram回退机制、Kneser-Ney平滑
> - **[深入探讨: MinHash与LSH算法](./Lecture14-MinHash-LSH.md)** - 近似去重的数学原理与实现

---

## 一、过滤算法基础 (Filtering Algorithms Fundamentals)

### 1.1 为什么需要过滤？

预训练数据的质量直接决定模型性能。互联网数据（如Common Crawl）包含大量：
- **低质量内容**: 垃圾邮件、广告文本、重复模板
- **非目标语言**: 多语言混杂
- **有害内容**: 毒性文本、违法信息

过滤的目标是：**从海量噪声数据中高效筛选出高质量、目标语言、安全的文本**。

### 1.2 N-gram 语言模型 (N-gram Language Models)

N-gram模型是过滤流水线的基础工具，用于评估文本质量。

#### 核心思想

给定一个token序列 $x_1, x_2, ..., x_n$，N-gram模型将其概率分解为：

$$P(x_1, ..., x_n) = \prod_{i=1}^{n} P(x_i | x_{i-n+1}, ..., x_{i-1})$$

- **Unigram (n=1)**: $P(x_i)$ - 只看当前词
- **Bigram (n=2)**: $P(x_i | x_{i-1})$ - 看前一个词
- **Trigram (n=3)**: $P(x_i | x_{i-2}, x_{i-1})$ - 看前两个词

#### Python实现：Bigram模型

```python
from collections import defaultdict
import math

class BigramModel:
    """简单的Bigram语言模型"""
    def __init__(self):
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.total_count = 0
    
    def train(self, texts: list[str]):
        """在语料库上训练模型"""
        for text in texts:
            tokens = text.split()
            for i, token in enumerate(tokens):
                self.unigram_counts[token] += 1
                self.total_count += 1
                if i > 0:
                    prev_token = tokens[i - 1]
                    self.bigram_counts[prev_token][token] += 1
    
    def log_prob(self, text: str) -> float:
        """计算文本的对数概率"""
        tokens = text.split()
        log_p = 0.0
        for i, token in enumerate(tokens):
            if i == 0:
                # Unigram概率
                p = (self.unigram_counts[token] + 1) / (self.total_count + len(self.unigram_counts))
            else:
                prev_token = tokens[i - 1]
                # Bigram概率，使用加一平滑
                count = self.bigram_counts[prev_token][token]
                total = sum(self.bigram_counts[prev_token].values())
                p = (count + 1) / (total + len(self.unigram_counts))
            log_p += math.log(p)
        return log_p
```

### 1.3 KenLM：高效N-gram语言模型

**KenLM** 是一个高度优化的N-gram语言模型库，支持:
- **修改Kneser-Ney平滑**: 比简单的加一平滑效果更好
- **回退机制 (Backoff)**: 当高阶n-gram未见过时，回退到低阶
- **压缩存储**: 使用Trie结构和量化技术

#### 安装与使用

```bash
pip install kenlm
# 需要先用lmplz工具训练.arpa模型
```

```python
import kenlm

# 加载预训练的KenLM模型
model = kenlm.Model("wiki_en_5gram.arpa")

# 计算句子的对数概率（以10为底）
text = "The quick brown fox jumps over the lazy dog"
log_prob = model.score(text)
perplexity = model.perplexity(text)

print(f"Log probability: {log_prob}")
print(f"Perplexity: {perplexity}")
```

#### 回退机制详解

当遇到未见过的n-gram时，KenLM使用回退：

$$P(x_i | x_{i-n+1:i-1}) = \begin{cases} 
P_{MLE}(x_i | x_{i-n+1:i-1}) & \text{if count} > 0 \\
\alpha(x_{i-n+1:i-1}) \cdot P(x_i | x_{i-n+2:i-1}) & \text{otherwise}
\end{cases}$$

其中 $\alpha$ 是回退权重，确保概率归一化。

### 1.4 fastText 文本分类器

**fastText** 是Facebook开发的高效文本分类工具，特别适合：
- **语言识别 (Language Identification)**
- **主题分类**
- **质量评分**

#### 核心技术

1. **词袋模型 + N-gram特征**: 捕获词序信息
2. **层次Softmax**: 加速大词表分类
3. **子词表示**: 使用字符n-gram处理OOV词

```python
import fasttext

# 训练语言识别模型
# 训练数据格式: __label__en This is English text
# model = fasttext.train_supervised("train.txt")

# 使用预训练的语言识别模型
lang_model = fasttext.load_model("lid.176.bin")

text = "This is a sample text"
predictions = lang_model.predict(text, k=3)  # 返回top-3预测
# 输出: (('__label__en', '__label__de', '__label__nl'), (0.95, 0.02, 0.01))

label, confidence = predictions[0][0], predictions[1][0]
print(f"Language: {label.replace('__label__', '')}, Confidence: {confidence:.2f}")
```

### 1.5 重要性重采样 (DSIR - Data Selection via Importance Resampling)

**DSIR** 是一种数据选择方法，核心思想是：**让选中的数据分布接近目标分布**。

#### 数学原理

设:
- $P_{raw}(x)$: 原始数据（如Common Crawl）的分布
- $P_{target}(x)$: 目标数据（如Wikipedia高质量文本）的分布

重要性权重:
$$w(x) = \frac{P_{target}(x)}{P_{raw}(x)}$$

对于文本 $x$，以概率 $\min(1, \lambda \cdot w(x))$ 选择它，其中 $\lambda$ 控制选择率。

#### 实现：使用KenLM计算重要性权重

```python
import kenlm
import math
import random

class DSIRFilter:
    """基于DSIR的数据选择器"""
    
    def __init__(self, target_model_path: str, raw_model_path: str):
        self.target_model = kenlm.Model(target_model_path)
        self.raw_model = kenlm.Model(raw_model_path)
    
    def compute_importance_weight(self, text: str) -> float:
        """计算重要性权重 w(x) = P_target(x) / P_raw(x)"""
        # KenLM返回log10概率，需要转换
        log_p_target = self.target_model.score(text)
        log_p_raw = self.raw_model.score(text)
        
        # 转换为对数权重（防止数值溢出）
        log_weight = (log_p_target - log_p_raw) * math.log(10)
        return math.exp(log_weight)
    
    def select(self, texts: list[str], selection_rate: float = 0.1) -> list[str]:
        """根据DSIR选择文本"""
        selected = []
        for text in texts:
            weight = self.compute_importance_weight(text)
            prob = min(1.0, selection_rate * weight)
            if random.random() < prob:
                selected.append(text)
        return selected
```

#### DSIR效果验证

![DSIR实验结果](./images/dsir-results.png)

DSIR在下游任务上的表现显著优于随机采样，证明了重要性重采样的有效性。

---

## 二、过滤应用场景 (Filtering Applications)

### 2.1 语言识别过滤 (Language Identification)

语言识别是最基础的过滤步骤，用于筛选特定语言的文本。

```python
import fasttext

class LanguageFilter:
    """语言过滤器"""
    
    def __init__(self, model_path: str = "lid.176.bin"):
        self.model = fasttext.load_model(model_path)
    
    def filter(self, texts: list[str], 
               target_lang: str = "en", 
               threshold: float = 0.8) -> list[str]:
        """筛选指定语言的文本"""
        filtered = []
        for text in texts:
            predictions = self.model.predict(text)
            lang = predictions[0][0].replace("__label__", "")
            confidence = predictions[1][0]
            
            if lang == target_lang and confidence >= threshold:
                filtered.append(text)
        
        return filtered
```

### 2.2 质量过滤 (Quality Filtering)

质量过滤使用多种信号评估文本质量：

#### 2.2.1 基于规则的过滤

```python
def is_high_quality_rule_based(text: str) -> bool:
    """基于规则的质量过滤"""
    # 检查长度
    word_count = len(text.split())
    if word_count < 50 or word_count > 100000:
        return False
    
    # 检查特殊字符比例
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.7:
        return False
    
    # 检查重复行比例
    lines = text.split('\n')
    unique_lines = set(lines)
    if len(unique_lines) / len(lines) < 0.5:
        return False
    
    # 检查"the"/"be"/"and"等常见词比例（英文质量信号）
    words = text.lower().split()
    common_words = set(["the", "be", "to", "of", "and", "a", "in"])
    common_ratio = sum(1 for w in words if w in common_words) / len(words)
    if common_ratio < 0.02 or common_ratio > 0.3:
        return False
    
    return True
```

#### 2.2.2 基于模型的过滤（如DCLM方法）

DCLM使用fastText分类器区分高质量（如OpenAI的WebText筛选结果）和低质量文本：

```python
class QualityClassifier:
    """基于fastText的质量分类器"""
    
    def __init__(self, model_path: str):
        self.model = fasttext.load_model(model_path)
    
    def is_high_quality(self, text: str, threshold: float = 0.5) -> bool:
        predictions = self.model.predict(text)
        label = predictions[0][0]
        confidence = predictions[1][0]
        
        return label == "__label__hq" and confidence >= threshold
```

### 2.3 毒性过滤 (Toxicity Filtering)

毒性过滤移除有害、攻击性或不当内容。

```python
class ToxicityFilter:
    """毒性内容过滤器"""
    
    def __init__(self, model_path: str, blocklist_path: str = None):
        self.model = fasttext.load_model(model_path)
        self.blocklist = set()
        if blocklist_path:
            with open(blocklist_path, 'r') as f:
                self.blocklist = set(line.strip().lower() for line in f)
    
    def contains_blocklist(self, text: str) -> bool:
        """检查是否包含黑名单词汇"""
        words = set(text.lower().split())
        return bool(words & self.blocklist)
    
    def is_toxic(self, text: str, threshold: float = 0.5) -> bool:
        """使用模型判断是否有毒"""
        predictions = self.model.predict(text)
        label = predictions[0][0]
        confidence = predictions[1][0]
        return label == "__label__toxic" and confidence >= threshold
    
    def filter(self, texts: list[str]) -> list[str]:
        """过滤毒性内容"""
        return [t for t in texts 
                if not self.contains_blocklist(t) and not self.is_toxic(t)]
```

---

## 三、去重技术 (Deduplication Techniques)

### 3.1 为什么需要去重？

研究表明：
- **重复数据损害模型性能**: 模型可能过拟合到重复内容
- **训练效率下降**: 浪费计算资源在重复样本上
- **隐私风险**: 重复内容更容易被记忆和泄露

### 3.2 精确去重 (Exact Deduplication)

精确去重移除完全相同的文档或段落。

#### 3.2.1 基于哈希的精确去重

```python
import hashlib

class ExactDeduplicator:
    """基于哈希的精确去重"""
    
    def __init__(self):
        self.seen_hashes = set()
    
    def get_hash(self, text: str) -> str:
        """计算文本的SHA-256哈希"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def deduplicate(self, texts: list[str]) -> list[str]:
        """去除重复文本"""
        unique_texts = []
        for text in texts:
            h = self.get_hash(text)
            if h not in self.seen_hashes:
                self.seen_hashes.add(h)
                unique_texts.append(text)
        return unique_texts
```

#### 3.2.2 行级去重

```python
def deduplicate_lines(texts: list[str]) -> list[str]:
    """对每篇文档的行进行去重"""
    result = []
    for text in texts:
        lines = text.split('\n')
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        result.append('\n'.join(unique_lines))
    return result
```

### 3.3 布隆过滤器 (Bloom Filter)

布隆过滤器是一种空间高效的概率数据结构，用于测试元素是否在集合中。

#### 特性

- **可能有假阳性 (False Positive)**: 可能错误地认为不在集合中的元素在集合中
- **绝无假阴性 (No False Negative)**: 如果说不在，则确实不在
- **空间效率极高**: 远小于存储实际元素

#### Python实现

```python
import mmh3  # MurmurHash3
from bitarray import bitarray
import math

class BloomFilter:
    """布隆过滤器实现"""
    
    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        # 计算最优位数组大小
        self.size = self._optimal_size(expected_items, false_positive_rate)
        # 计算最优哈希函数数量
        self.num_hashes = self._optimal_num_hashes(self.size, expected_items)
        # 初始化位数组
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)
    
    def _optimal_size(self, n: int, p: float) -> int:
        """计算最优位数组大小: m = -n*ln(p) / (ln(2)^2)"""
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(m)
    
    def _optimal_num_hashes(self, m: int, n: int) -> int:
        """计算最优哈希数量: k = (m/n) * ln(2)"""
        k = (m / n) * math.log(2)
        return int(k)
    
    def _get_hash_positions(self, item: str) -> list[int]:
        """获取元素的所有哈希位置"""
        positions = []
        for seed in range(self.num_hashes):
            hash_value = mmh3.hash(item, seed) % self.size
            positions.append(hash_value)
        return positions
    
    def add(self, item: str):
        """添加元素"""
        for pos in self._get_hash_positions(item):
            self.bit_array[pos] = 1
    
    def contains(self, item: str) -> bool:
        """检查元素是否可能存在"""
        return all(self.bit_array[pos] for pos in self._get_hash_positions(item))
    
    def add_and_check(self, item: str) -> bool:
        """添加元素并返回是否已存在（用于去重）"""
        positions = self._get_hash_positions(item)
        was_present = all(self.bit_array[pos] for pos in positions)
        for pos in positions:
            self.bit_array[pos] = 1
        return was_present
```

#### 使用布隆过滤器进行N-gram去重

```python
def bloom_ngram_dedup(texts: list[str], n: int = 5, 
                       bloom_size: int = 10_000_000) -> list[str]:
    """使用布隆过滤器进行N-gram级别去重"""
    bloom = BloomFilter(expected_items=bloom_size)
    result = []
    
    for text in texts:
        words = text.split()
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        # 计算重复n-gram比例
        duplicate_count = 0
        for ngram in ngrams:
            if bloom.add_and_check(ngram):
                duplicate_count += 1
        
        duplicate_ratio = duplicate_count / len(ngrams) if ngrams else 0
        
        # 如果重复率低于阈值，保留文档
        if duplicate_ratio < 0.5:
            result.append(text)
    
    return result
```

### 3.4 近似去重：MinHash

**MinHash** 用于估计两个集合的Jaccard相似度，是近似去重的核心算法。

#### Jaccard相似度

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

#### MinHash原理

1. 将文档表示为n-gram集合
2. 对集合应用多个随机哈希函数
3. 取每个哈希函数的最小值作为签名
4. 两个文档的签名相同的概率等于其Jaccard相似度

```python
import mmh3
import numpy as np

class MinHash:
    """MinHash签名生成器"""
    
    def __init__(self, num_hashes: int = 128):
        self.num_hashes = num_hashes
        # 使用不同seed生成多个哈希函数
        self.seeds = list(range(num_hashes))
    
    def get_signature(self, document: str, ngram_size: int = 5) -> np.ndarray:
        """计算文档的MinHash签名"""
        # 生成n-gram集合
        words = document.split()
        ngrams = set(' '.join(words[i:i+ngram_size]) 
                     for i in range(len(words) - ngram_size + 1))
        
        # 初始化签名为最大值
        signature = np.full(self.num_hashes, np.iinfo(np.int32).max, dtype=np.int32)
        
        # 对每个n-gram计算所有哈希值，取最小
        for ngram in ngrams:
            for i, seed in enumerate(self.seeds):
                hash_value = mmh3.hash(ngram, seed)
                if hash_value < signature[i]:
                    signature[i] = hash_value
        
        return signature
    
    def estimate_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """估计两个签名的Jaccard相似度"""
        return np.mean(sig1 == sig2)
```

### 3.5 局部敏感哈希 (LSH - Locality Sensitive Hashing)

**LSH** 用于高效查找相似文档，避免两两比较。

#### 分桶策略 (Banding)

将MinHash签名分成 $b$ 个带 (bands)，每带 $r$ 行：
- 如果任意一个带的签名完全相同，则认为是候选对
- 调整 $b$ 和 $r$ 可以控制相似度阈值

```python
class LSH:
    """局部敏感哈希实现"""
    
    def __init__(self, num_hashes: int = 128, num_bands: int = 32):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        # 每个带一个哈希表
        self.hash_tables = [dict() for _ in range(num_bands)]
    
    def add(self, doc_id: str, signature: np.ndarray):
        """添加文档签名到LSH索引"""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(signature[start:end])
            
            if band_signature not in self.hash_tables[band_idx]:
                self.hash_tables[band_idx][band_signature] = []
            self.hash_tables[band_idx][band_signature].append(doc_id)
    
    def find_candidates(self, signature: np.ndarray) -> set:
        """找到候选相似文档"""
        candidates = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_signature = tuple(signature[start:end])
            
            if band_signature in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][band_signature])
        
        return candidates
```

### 3.6 完整的近似去重流水线

```python
class FuzzyDeduplicator:
    """基于MinHash + LSH的近似去重"""
    
    def __init__(self, num_hashes: int = 128, num_bands: int = 32,
                 similarity_threshold: float = 0.8):
        self.minhash = MinHash(num_hashes)
        self.lsh = LSH(num_hashes, num_bands)
        self.threshold = similarity_threshold
        self.signatures = {}
    
    def deduplicate(self, texts: list[str]) -> list[str]:
        """执行近似去重"""
        unique_indices = []
        
        for idx, text in enumerate(texts):
            sig = self.minhash.get_signature(text)
            
            # 找到候选相似文档
            candidates = self.lsh.find_candidates(sig)
            
            is_duplicate = False
            for cand_id in candidates:
                if cand_id in self.signatures:
                    similarity = self.minhash.estimate_similarity(
                        sig, self.signatures[cand_id])
                    if similarity >= self.threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                self.signatures[idx] = sig
                self.lsh.add(idx, sig)
                unique_indices.append(idx)
        
        return [texts[i] for i in unique_indices]
```

---

## 四、完整流水线 (Complete Pipeline)

### 4.1 端到端数据处理流水线

```python
class DataProcessingPipeline:
    """完整的数据过滤和去重流水线"""
    
    def __init__(self):
        self.lang_filter = LanguageFilter()
        self.quality_classifier = QualityClassifier("quality_model.bin")
        self.toxicity_filter = ToxicityFilter("toxicity_model.bin")
        self.exact_dedup = ExactDeduplicator()
        self.fuzzy_dedup = FuzzyDeduplicator()
    
    def process(self, texts: list[str], target_lang: str = "en") -> list[str]:
        """执行完整的处理流水线"""
        # Step 1: 语言过滤
        texts = self.lang_filter.filter(texts, target_lang)
        print(f"After language filter: {len(texts)} documents")
        
        # Step 2: 精确去重
        texts = self.exact_dedup.deduplicate(texts)
        print(f"After exact dedup: {len(texts)} documents")
        
        # Step 3: 质量过滤
        texts = [t for t in texts if self.quality_classifier.is_high_quality(t)]
        print(f"After quality filter: {len(texts)} documents")
        
        # Step 4: 毒性过滤
        texts = self.toxicity_filter.filter(texts)
        print(f"After toxicity filter: {len(texts)} documents")
        
        # Step 5: 近似去重
        texts = self.fuzzy_dedup.deduplicate(texts)
        print(f"After fuzzy dedup: {len(texts)} documents")
        
        return texts
```

---

## 五、关键要点总结 (Key Takeaways)

### 过滤技术总结

| 技术 | 用途 | 优点 | 缺点 |
|------|------|------|------|
| KenLM | 质量评分/DSIR | 快速、准确 | 需要训练语料 |
| fastText | 语言/质量/毒性分类 | 高效、易用 | 需要标注数据 |
| DSIR | 分布匹配采样 | 理论保证 | 需要两个语言模型 |

### 去重技术总结

| 技术 | 复杂度 | 空间 | 特点 |
|------|--------|------|------|
| 哈希精确去重 | O(n) | O(n) | 无损，只处理完全相同 |
| 布隆过滤器 | O(n) | O(1)* | 有假阳性，极省空间 |
| MinHash | O(n) | O(n·k) | 近似，处理相似文档 |
| MinHash + LSH | O(n) 期望 | O(n·k) | 避免两两比较 |

### 最佳实践

1. **流水线顺序**: 先便宜的过滤（规则、语言）→ 再昂贵的（模型质量）→ 最后去重
2. **阈值调优**: 根据下游任务表现调整各过滤阈值
3. **分布式处理**: 对于TB级数据，需要MapReduce/Spark分布式实现
4. **多次迭代**: 可能需要多轮去重达到最佳效果

---

## 参考资料

1. **KenLM**: Heafield, K. (2011). KenLM: Faster and Smaller Language Model Queries
2. **fastText**: Joulin, A. et al. (2016). Bag of Tricks for Efficient Text Classification
3. **DSIR**: Xie et al. (2023). Data Selection for Language Models via Importance Resampling
4. **MinHash**: Broder, A. (1997). On the Resemblance and Containment of Documents
5. **DCLM**: Li et al. (2024). DataComp-LM: In Search of the next Generation of Training Data
