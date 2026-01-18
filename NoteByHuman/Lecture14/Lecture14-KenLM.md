# 深入探讨: KenLM与N-gram语言模型

本文是Lecture 14的精英补充笔记，深入讲解N-gram语言模型的数学原理、Kneser-Ney平滑技术，以及KenLM的高效实现。

---

## 一、N-gram语言模型基础

### 1.1 语言模型的目标

语言模型的目标是估计**序列概率** $P(w_1, w_2, ..., w_n)$。

根据链式法则:
$$P(w_1, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1})$$

**问题**: 历史 $w_1, ..., w_{i-1}$ 可能非常长，无法直接估计。

**解决方案**: **马尔可夫假设** - 只考虑前 $n-1$ 个词

$$P(w_i | w_1, ..., w_{i-1}) \approx P(w_i | w_{i-n+1}, ..., w_{i-1})$$

### 1.2 N-gram的阶数选择

| 阶数 | 模型 | 条件 | 优缺点 |
|------|------|------|--------|
| n=1 | Unigram | $P(w_i)$ | 简单但忽略上下文 |
| n=2 | Bigram | $P(w_i \| w_{i-1})$ | 捕捉局部依赖 |
| n=3 | Trigram | $P(w_i \| w_{i-2}, w_{i-1})$ | 更丰富的上下文 |
| n=5 | 5-gram | $P(w_i \| w_{i-4:i-1})$ | KenLM默认，平衡性能和泛化 |

### 1.3 最大似然估计

对于Bigram:
$$P_{MLE}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

其中 $C(\cdot)$ 是计数函数。

**问题**: 未见过的n-gram概率为0！

---

## 二、平滑技术 (Smoothing)

### 2.1 为什么需要平滑？

语料库是有限的，必然有大量未见过的n-gram。

**Zipf's Law**: 词频分布极不均匀
- 少数词出现非常频繁
- 大量词只出现一两次
- 更多词从未出现

### 2.2 Add-One (Laplace) 平滑

最简单的方法:
$$P_{add-1}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$$

其中 $V$ 是词表大小。

**问题**: 给未见n-gram分配了太多概率质量！

### 2.3 Add-k 平滑

改进版:
$$P_{add-k}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + kV}$$

$k < 1$ 可以减少对未见n-gram的过度分配。

### 2.4 Kneser-Ney平滑

**核心思想**: 不只看频率，还要看**上下文多样性**。

#### 绝对折扣 (Absolute Discounting)

首先，从每个计数中减去固定值 $d$:
$$P_{abs}(w_i | w_{i-1}) = \frac{\max(C(w_{i-1}, w_i) - d, 0)}{C(w_{i-1})} + \lambda(w_{i-1}) P_{lower}(w_i)$$

其中 $\lambda(w_{i-1})$ 是归一化因子。

#### Continuation概率

Kneser-Ney的创新：对于低阶分布，使用**continuation概率**而非频率:

$$P_{KN}(w) = \frac{|\{w' : C(w', w) > 0\}|}{|\{(w', w'') : C(w', w'') > 0\}|}$$

**直觉**: 一个词的概率不取决于它出现多少次，而是它能跟多少不同的前缀搭配。

**例子**: "Francisco" 几乎只跟 "San" 搭配
- 高频率，但低continuation概率
- 作为回退，不应该得到高概率

#### Modified Kneser-Ney

使用多个折扣值 $d_1, d_2, d_{3+}$，根据原始计数选择:

$$d = \begin{cases}
0 & \text{if } C = 0 \\
d_1 & \text{if } C = 1 \\
d_2 & \text{if } C = 2 \\
d_{3+} & \text{if } C \geq 3
\end{cases}$$

---

## 三、回退机制 (Backoff)

### 3.1 Katz回退

当高阶n-gram未见过时，回退到低阶:

$$P_{BO}(w_i | w_{i-n+1:i-1}) = \begin{cases}
P^*(w_i | w_{i-n+1:i-1}) & \text{if } C(w_{i-n+1:i}) > k \\
\alpha(w_{i-n+1:i-1}) \cdot P_{BO}(w_i | w_{i-n+2:i-1}) & \text{otherwise}
\end{cases}$$

其中:
- $P^*$ 是折扣后的概率
- $\alpha$ 是回退权重，确保概率归一化

### 3.2 插值 (Interpolation)

另一种方法是始终混合不同阶:

$$P_{interp}(w_i | w_{i-n+1:i-1}) = \lambda_n P_n + \lambda_{n-1} P_{n-1} + ... + \lambda_1 P_1$$

其中 $\sum \lambda_i = 1$。

### 3.3 Kneser-Ney的完整公式

结合回退和continuation概率:

$$P_{KN}(w_i | w_{i-n+1:i-1}) = \frac{\max(C(w_{i-n+1:i}) - d, 0)}{C(w_{i-n+1:i-1})} + \gamma(w_{i-n+1:i-1}) P_{KN}(w_i | w_{i-n+2:i-1})$$

其中低阶 $P_{KN}$ 使用continuation计数。

---

## 四、KenLM实现细节

### 4.1 高效存储：Trie结构

KenLM使用**压缩Trie**存储n-gram:

```
      ROOT
     / | \
   the  a   ...
   / \
  cat dog
  |
 sat
```

每个节点存储:
- **概率**: 量化后的log概率
- **回退权重**: 量化后的回退值
- **指针**: 指向子节点

### 4.2 量化 (Quantization)

为了节省空间，KenLM对概率和回退权重进行量化:

```python
# 伪代码
def quantize(value, bits=8):
    # 将连续值映射到256个离散级别
    min_val, max_val = get_range(all_values)
    level = int((value - min_val) / (max_val - min_val) * (2**bits - 1))
    return level
```

8-bit量化可以将存储减少4倍，精度损失可忽略。

### 4.3 查询流程

```python
def query_kenlm(sentence, model):
    """计算句子的log概率"""
    tokens = tokenize(sentence)
    log_prob = 0.0
    state = model.begin_state()
    
    for token in tokens:
        # 尝试最长匹配
        prob, new_state = model.score(state, token)
        log_prob += prob  # log10概率
        state = new_state
    
    return log_prob

def model.score(state, token):
    """查询单个token的概率"""
    context = get_context_from_state(state)
    
    for order in range(max_order, 0, -1):
        ngram = context[-(order-1):] + [token]
        if ngram in trie:
            prob = trie[ngram].log_prob
            new_state = update_state(context, token)
            return prob, new_state
        else:
            # 回退：乘以回退权重，尝试低阶
            backoff = trie[context[-(order-1):]].backoff
            prob += backoff
            context = context[1:]  # 缩短上下文
    
    # 回退到unigram
    return unigram_prob[token], reset_state()
```

### 4.4 KenLM vs 其他实现

| 特性 | KenLM | SRILM | 自定义Python |
|------|-------|-------|--------------|
| 速度 | 极快 | 快 | 慢 |
| 内存 | 低（压缩） | 中 | 高 |
| 最大阶数 | 无限制 | 通常5-7 | 自定义 |
| 批量查询 | 支持 | 支持 | 需自行实现 |

### 4.5 训练KenLM模型

```bash
# 1. 准备文本数据 (每行一个句子)
# corpus.txt

# 2. 使用lmplz训练
lmplz -o 5 \        # 5-gram
      -S 80% \      # 使用80%内存
      --discount_fallback \
      < corpus.txt \
      > model.arpa

# 3. 转换为二进制格式 (更快加载)
build_binary model.arpa model.binary
```

---

## 五、在数据过滤中的应用

### 5.1 质量评分

使用困惑度 (Perplexity) 评估文本质量:

$$PPL(w_1, ..., w_n) = P(w_1, ..., w_n)^{-1/n}$$

**低困惑度** = 模型预测好 = 文本"正常"
**高困惑度** = 模型预测差 = 文本"异常"

```python
import kenlm

model = kenlm.Model("wiki.binary")

def perplexity_filter(text, threshold=500):
    """基于困惑度过滤低质量文本"""
    ppl = model.perplexity(text)
    return ppl < threshold
```

### 5.2 DSIR中的应用

在DSIR中，使用两个KenLM模型计算重要性权重:

```python
target_model = kenlm.Model("high_quality.binary")  # 如Wikipedia
raw_model = kenlm.Model("raw_data.binary")         # 如Common Crawl

def importance_weight(text):
    log_p_target = target_model.score(text)
    log_p_raw = raw_model.score(text)
    return 10 ** (log_p_target - log_p_raw)  # 转换回概率比
```

---

## 参考资料

1. Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling
2. Kneser, R., & Ney, H. (1995). Improved backing-off for m-gram language modeling
3. Heafield, K. (2011). KenLM: Faster and Smaller Language Model Queries
4. Jurafsky, D., & Martin, J. (2023). Speech and Language Processing, Chapter 3
