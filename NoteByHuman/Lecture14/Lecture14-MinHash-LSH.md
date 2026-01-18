# 深入探讨: MinHash与LSH算法

本文是Lecture 14的精英补充笔记，深入讲解近似去重的数学原理，包括Jaccard相似度、MinHash签名、局部敏感哈希(LSH)及其理论保证。

---

## 一、问题定义：大规模近似去重

### 1.1 精确去重的局限

精确匹配只能找到**完全相同**的文档。然而：
- 复制粘贴常带有微小修改（日期、标点）
- 同一模板生成的内容高度相似但不相同
- 翻译、改写仍包含大量重复信息

### 1.2 近似去重的目标

找到**高度相似**（如>80%相似）的文档对，并移除其中之一。

**挑战**: N个文档，两两比较需要 $O(N^2)$ 时间。对于数十亿文档，不可行！

---

## 二、Jaccard相似度

### 2.1 定义

两个集合 $A$ 和 $B$ 的Jaccard相似度:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**性质**:
- $J(A, B) \in [0, 1]$
- $J(A, A) = 1$
- $J(A, B) = J(B, A)$

### 2.2 文档表示为集合

将文档转化为**shingles (k-gram)集合**:

```python
def shingling(document: str, k: int = 5) -> set:
    """将文档转为k-shingle集合"""
    words = document.split()
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i+k])
        shingles.add(shingle)
    return shingles

# 示例
doc1 = "the quick brown fox jumps over the lazy dog"
doc2 = "the quick brown fox leaps over the lazy dog"

s1 = shingling(doc1, k=3)  # {"the quick brown", "quick brown fox", ...}
s2 = shingling(doc2, k=3)

jaccard = len(s1 & s2) / len(s1 | s2)  # 计算相似度
```

### 2.3 k值的选择

| k值 | 特点 |
|-----|------|
| k=2-3 | 太小，假阳性高（常见短语匹配） |
| k=5 | 常用默认值 |
| k=9-10 | 更严格，假阴性可能增加 |

---

## 三、MinHash：相似度的压缩估计

### 3.1 核心思想

**目标**: 将大集合压缩成固定长度的**签名**，使得签名的相似度近似原集合的Jaccard相似度。

**关键定理**: 
$$P[h_{min}(A) = h_{min}(B)] = J(A, B)$$

其中 $h_{min}(S) = \arg\min_{x \in S} h(x)$ 是集合 $S$ 中哈希值最小的元素。

### 3.2 定理证明

考虑 $A \cup B$ 中的所有元素，每个都有唯一的哈希值（假设无碰撞）。

令 $x^* = \arg\min_{x \in A \cup B} h(x)$ 是哈希值最小的元素。

- 如果 $x^* \in A \cap B$，则 $h_{min}(A) = h_{min}(B) = h(x^*)$
- 如果 $x^* \in A \setminus B$，则 $h_{min}(A) = h(x^*) \neq h_{min}(B)$
- 如果 $x^* \in B \setminus A$，则 $h_{min}(B) = h(x^*) \neq h_{min}(A)$

由于哈希函数是随机的，$x^*$ 落在 $A \cap B$ 中的概率为:
$$P[x^* \in A \cap B] = \frac{|A \cap B|}{|A \cup B|} = J(A, B)$$

### 3.3 多个哈希函数

使用 $k$ 个独立的哈希函数 $h_1, ..., h_k$，生成长度为 $k$ 的签名:

$$\text{sig}(A) = [h_{1,min}(A), h_{2,min}(A), ..., h_{k,min}(A)]$$

**估计相似度**:
$$\hat{J}(A, B) = \frac{1}{k} \sum_{i=1}^{k} \mathbf{1}[h_{i,min}(A) = h_{i,min}(B)]$$

这是Jaccard相似度的**无偏估计**。

### 3.4 估计的方差

$$\text{Var}[\hat{J}] = \frac{J(1-J)}{k}$$

**含义**: 签名越长 ($k$ 越大)，估计越准确。

**实践中**: $k = 128$ 或 $k = 256$ 是常见选择。

### 3.5 实现优化：单哈希多种子

实际上不需要 $k$ 个完全独立的哈希函数:

```python
import mmh3

def minhash_signature(shingles: set, num_hashes: int = 128) -> list:
    """使用不同种子的MurmurHash生成签名"""
    signature = []
    for seed in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            h = mmh3.hash(shingle, seed)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature
```

---

## 四、局部敏感哈希 (LSH)

### 4.1 问题：N个签名的两两比较

即使有了长度 $k$ 的签名，N个文档仍需 $O(N^2 \cdot k)$ 比较。

**LSH的目标**: 只比较**可能相似**的文档对。

### 4.2 分带技术 (Banding)

将签名分成 $b$ 个带 (bands)，每带 $r$ 行，满足 $b \cdot r = k$。

**候选对规则**: 如果两个文档在**任意一个带**完全相同，则成为候选对。

```python
def lsh_buckets(signature: list, num_bands: int = 32) -> list:
    """将签名分带并哈希到桶中"""
    rows_per_band = len(signature) // num_bands
    buckets = []
    
    for band_idx in range(num_bands):
        start = band_idx * rows_per_band
        end = start + rows_per_band
        band = tuple(signature[start:end])
        bucket_id = hash(band)  # 每个带独立哈希
        buckets.append((band_idx, bucket_id))
    
    return buckets
```

### 4.3 概率分析

设真实Jaccard相似度为 $s$。

**单个MinHash相等的概率**: $s$

**一个带（$r$行）完全相同的概率**: $s^r$

**一个带不完全相同的概率**: $1 - s^r$

**所有$b$个带都不完全相同的概率**: $(1 - s^r)^b$

**至少一个带相同（成为候选对）的概率**:
$$P(\text{candidate}) = 1 - (1 - s^r)^b$$

### 4.4 S曲线特性

这个概率函数呈现**S曲线**特性:

```
候选概率
    1 |              ___________
      |           __/
      |         _/
      |        /
      |      _/
    0 |_____/
      ├────┼────┼────┼────┼────► 相似度s
      0   0.2  0.4  0.6  0.8  1.0
```

**阈值点**（曲线陡峭处）约在:
$$t \approx (1/b)^{1/r}$$

### 4.5 参数选择

| b | r | k=b×r | 阈值≈ | 用途 |
|---|---|-------|-------|------|
| 20 | 5 | 100 | 0.55 | 宽松匹配 |
| 32 | 4 | 128 | 0.60 | 常用设置 |
| 50 | 4 | 200 | 0.50 | 更多候选对 |
| 25 | 10 | 250 | 0.85 | 严格匹配 |

**选择原则**:
- 想要更高阈值 → 增加 $r$
- 想要更多候选对（不漏掉相似文档）→ 增加 $b$

### 4.6 完整LSH流程

```python
class LSH:
    def __init__(self, num_hashes=128, num_bands=32):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        self.hash_tables = [{} for _ in range(num_bands)]
    
    def add(self, doc_id, signature):
        """添加文档到索引"""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_sig = tuple(signature[start:end])
            
            if band_sig not in self.hash_tables[band_idx]:
                self.hash_tables[band_idx][band_sig] = set()
            self.hash_tables[band_idx][band_sig].add(doc_id)
    
    def query(self, signature):
        """查询候选相似文档"""
        candidates = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_sig = tuple(signature[start:end])
            
            if band_sig in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][band_sig])
        
        return candidates
```

---

## 五、误差分析

### 5.1 假阳性与假阴性

| 错误类型 | 定义 | 后果 |
|----------|------|------|
| 假阳性 (FP) | 不相似但成为候选对 | 多做一次精确比较（浪费时间） |
| 假阴性 (FN) | 相似但不成为候选对 | 漏掉真正的重复（质量损失） |

### 5.2 权衡

设目标阈值为 $t$:

**假阳性率**: $P(\text{candidate} | s < t)$ - S曲线在 $s < t$ 区域的积分

**假阴性率**: $P(\text{not candidate} | s \geq t)$ - S曲线在 $s \geq t$ 区域的（1-积分）

**优化目标**:
- 降低FN → 增大 $b$（更多带，更容易匹配）
- 降低FP → 增大 $r$（每带更长，更难匹配）

### 5.3 两阶段过滤

实践中通常采用两阶段方法:

1. **LSH阶段**: 快速找到候选对（允许一些FP，尽量减少FN）
2. **验证阶段**: 对候选对计算精确Jaccard相似度

```python
def deduplicate(documents, threshold=0.8):
    # 1. 生成签名
    signatures = {doc_id: minhash(doc) for doc_id, doc in documents}
    
    # 2. LSH找候选对
    lsh = LSH(num_hashes=128, num_bands=32)
    for doc_id, sig in signatures.items():
        lsh.add(doc_id, sig)
    
    # 3. 验证候选对
    duplicates = set()
    for doc_id, sig in signatures.items():
        candidates = lsh.query(sig)
        for cand_id in candidates:
            if cand_id >= doc_id:  # 避免重复比较
                continue
            # 精确计算相似度
            sim = exact_jaccard(documents[doc_id], documents[cand_id])
            if sim >= threshold:
                duplicates.add(doc_id)  # 标记为重复
                break
    
    # 4. 返回非重复文档
    return [d for doc_id, d in documents if doc_id not in duplicates]
```

---

## 六、大规模实现考虑

### 6.1 分布式MinHash

```
Map阶段:
  - 每个Mapper处理一批文档
  - 计算MinHash签名
  - 按band输出 (band_id, band_sig) → doc_id

Reduce阶段:
  - 每个Reducer处理一个band
  - 相同band_sig的文档是候选对
  - 输出候选对或直接标记重复
```

### 6.2 内存优化

- **在线添加**: 签名计算后立即加入LSH，不需存储所有签名
- **布隆过滤器辅助**: 用布隆过滤器快速判断band是否见过
- **压缩签名**: 使用更少的bits存储MinHash值

### 6.3 多轮去重

单轮LSH可能漏掉一些重复（假阴性）。可以:
1. 多轮使用不同的随机种子
2. 逐步降低阈值进行多轮
3. 使用不同的shingle大小

---

## 参考资料

1. Broder, A. Z. (1997). On the resemblance and containment of documents
2. Leskovec, J., Rajaraman, A., & Ullman, J. (2020). Mining of Massive Datasets, Chapter 3
3. Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: towards removing the curse of dimensionality
4. Gionis, A., Indyk, P., & Motwani, R. (1999). Similarity search in high dimensions via hashing
