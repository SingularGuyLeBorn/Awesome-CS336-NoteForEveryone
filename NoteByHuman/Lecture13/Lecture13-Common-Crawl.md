# 深入探讨: Common Crawl 与网络爬虫

**来源**: CS336 Lecture 13 · **主题**: 预训练数据的基础来源

---

## 1. Common Crawl 是什么?

**Common Crawl** 是一个成立于 2007 年的**非营利组织**, 定期爬取互联网并公开发布数据.

**官方地址**: https://commoncrawl.org/

---

## 2. 基本统计

- **频率**: 大约每月一次爬虫
- **历史**: 2008-2025 年, 约 100 次爬虫
- **规模**: 每次爬虫约 27 亿页面
- **成本**: 2016 年数据显示, 100 台机器运行 10-12 天
- **最新**: 2025 年 4 月

---

## 3. 爬虫技术

### 3.1 基本流程

使用 **Apache Nutch** 开源库:

1. 从**种子 URL** 开始 (数亿个, 而非从一个网站开始)
2. 维护一个**爬虫队列 (Crawl Frontier)**
3. 多台机器并行从队列获取 URL, 下载页面
4. 从页面中提取新链接, 加入队列
5. 类似 BFS 遍历整个网络

### 3.2 爬虫策略

| 策略 | 说明 |
|---|---|
| **选择策略** | 决定下载哪些页面 |
| **礼貌策略** | 尊重 robots.txt, 不过载服务器 |
| **重访策略** | 决定何时重新检查页面变化 |

### 3.3 挑战

- **URL 动态性**: URL 可能很长, 多个 URL 指向相同内容
- **重复内容**: 大量重复需要去重
- **覆盖不完整**: Common Crawl 刻意保守和礼貌, **不是完整的互联网**

> **事实**: 即使是 Wikipedia 的所有文章也不完全在 Common Crawl 中!

---

## 4. 数据格式

### 4.1 WARC (Web ARChive)

- 原始 HTTP 响应
- 通常是 HTML
- 保留完整结构

### 4.2 WET (WARC Extracted Text)

- 从 WARC 转换的纯文本
- **有损过程**: 丢失结构信息

### 4.3 HTML → 文本转换

Common Crawl 提供的 WET 文件已经是文本, 但你也可以从 WARC 自己转换.

**常用工具**:
- **trafilatura**: 高质量提取, 但可能丢弃更多内容
- **jusText**: 保留更多文本, 但可能有噪声
- **resiliparse**: 另一个选择

**DCLM 研究发现**: 使用 trafilatura 比使用 WET 文件高 **4 个百分点**!

---

## 5. robots.txt

网站可以通过 `robots.txt` 文件声明哪些爬虫可以访问:

```
User-agent: GPTBot
Disallow: /

User-agent: Google-Extended
Disallow: /

User-agent: *
Allow: /
```

**注意**:
- 这是**指导性**的, 没有法律强制力
- 许多前沿模型开发者有自己的爬虫 (因为 Common Crawl 覆盖不够)
- 不是所有人都尊重 robots.txt

---

## 6. 敏感内容处理

**问答 (来自课堂)**:

Q: Common Crawl 是否过滤敏感/冒犯内容?

A: 默认非常宽松. "冒犯"是高层语义判断, Common Crawl 不做这种判断. 可能有一些明显违法内容的黑名单, 但细节不清楚.

---

## 7. 与"互联网"的区别

当有人说"语言模型在互联网上训练", 这**不完全正确**:

1. Common Crawl ≠ 互联网
2. Common Crawl 刻意不完整 (礼貌策略)
3. 许多内容在 robots.txt 中被禁止
4. 还有大量过滤和处理

**更准确的说法**: 模型在 Common Crawl 的处理后子集上训练.

---

## 8. 使用注意

1. **版权**: Common Crawl 中的大部分内容都有版权
2. **质量**: 原始 Common Crawl 质量很低, 需要大量过滤
3. **格式**: 优先使用 WARC + 好的转换器, 而非 WET
4. **去重**: 大量重复, 需要去重处理

---

## 参考资料

- [Common Crawl 官网](https://commoncrawl.org/)
- [Common Crawl 博客](https://commoncrawl.org/blog/)
- [Apache Nutch](https://nutch.apache.org/)
