# 🚀✨ CS336-NoteForEveryone ✨🚀

<div align="center">

🔥 **Stanford大语言模型课程全系列笔记** 🔥

📚 从 Transformer 到 RLHF，从 Scaling Law 到 GRPO 📚

🌟 Zero to Hero，一步一个脚印 🌟

</div>

---

## 💭 为什么选择这门课？

> *说实话，最近越来越焦虑。每天刷 arXiv，DeepSeek R1 出来了、Qwen 3 又更新了、o1 的推理能力刷新认知……感觉自己像是站在飞驰的列车外面，眼睁睁看着学界的列车呼啸而过。*
>
> *与其焦虑，不如行动。我决定回归基础，从 Stanford CS336 开始，把大模型的每一个技术细节都吃透。不是为了追热点，而是为了真正理解这场革命背后的原理。*
>
> *这个仓库记录的是我学习的过程——有时候一天只推进几分钟，有时候一个周末泡在里面。不完美，但真实。*

### 🎉 各位久等了！

**五个Assignment也将陆续开源**，让我把垃圾代码重构一遍～ 

*(慢慢来，但一定要快 O(∩_∩)O)*

---

## 📚 课程笔记目录

### 第一部分：模型基础 (Lectures 1-7)
| 课程 | 主题 | 笔记 | 一句话感想 |
|------|------|------|-----------|
| 📖 Lecture 1 | 课程导论 | [笔记](./NoteByHuman/Lecture1/) | 终于开始了，冲！ |
| 🏗️ Lecture 2 | Transformer架构 | [笔记](./NoteByHuman/Lecture2/) | 有点混乱，需要多复盘和实践 |
| ✂️ Lecture 3 | Tokenization分词 | [笔记](./NoteByHuman/Lecture3/) | 老师语速太快，信息密度爆炸 |
| 🧠 Lecture 4 | MoE混合专家模型 | [笔记](./NoteByHuman/Lecture4/) | 接近两个月没更新后重新捡起来 |
| ⚡ Lecture 5 | GPU性能优化 | [笔记](./NoteByHuman/Lecture5/) | 出差途中忙里偷闲，收获满满 |
| 🔧 Lecture 6 | CUDA/PTX深入 | [笔记](./NoteByHuman/Lecture6/) | CUDA太深了，先跳过后面再补 |
| 🌐 Lecture 7 | 分布式训练基础 | [笔记](./NoteByHuman/Lecture7/) | 比Lecture 6好懂一点 |

### 第二部分：并行与扩展 (Lectures 8-11)
| 课程 | 主题 | 笔记 | 一句话感想 |
|------|------|------|-----------|
| 🔀 Lecture 8 | 手撕并行训练 | [笔记](./NoteByHuman/Lecture8/) | 一看到手撕就害怕 |
| 📈 Lecture 9 | 详解Scaling Law | [笔记](./NoteByHuman/Lecture9/) | 原来Chinchilla这么有名 |
| 🏎️ Lecture 10 | 模型推理优化 | [笔记](./NoteByHuman/Lecture10/) | KV Cache终于搞明白了 |
| 🎯 Lecture 11 | 如何用好Scaling Law | [笔记](./NoteByHuman/Lecture11/) | μP原来是这个意思！ |

### 第三部分：评估与数据 (Lectures 12-14)
| 课程 | 主题 | 笔记 | 一句话感想 |
|------|------|------|-----------|
| 📊 Lecture 12 | 模型评估详解 | [笔记](./NoteByHuman/Lecture12/) | MMLU居然也有这么多坑 |
| 📦 Lecture 13 | 训练数据策略 | [笔记](./NoteByHuman/Lecture13/) | Common Crawl是个宝藏 |
| 🧹 Lecture 14 | 实战数据过滤和去重 | [笔记](./NoteByHuman/Lecture14/) | MinHash太优雅了 |

### 第四部分：后训练与RL (Lectures 15-17)
| 课程 | 主题 | 笔记 | 一句话感想 |
|------|------|------|-----------|
| 🎓 Lecture 15 | 详解SFT与RLHF | [笔记](./NoteByHuman/Lecture15/) | InstructGPT原来是这么做的 |
| 🎮 Lecture 16 | 详解大模型RL算法 | [笔记](./NoteByHuman/Lecture16/) | R1/K1.5/Qwen3一次看个够 |
| 🛠️ Lecture 17 | 手把手讲解GRPO | [笔记](./NoteByHuman/Lecture17/) | 代码级别的深入讲解，太香了 |

---

## 🌟 笔记特色

- 🎯 **零信息损失**：完整覆盖课堂所有技术细节
- 🌲 **T型知识结构**：主笔记 + 深度补充笔记
- 💻 **代码集成**：结合课程Python代码讲解
- 📸 **图表丰富**：包含课程slides和手绘图解
- 🇨🇳 **中文原创**：非机翻，适合中文读者

---

## 📂 仓库结构

```
├── NoteByHuman/           # 手工整理的完整笔记
│   ├── Lecture1-17/       # 各课程笔记目录
│   └── */images/          # 笔记配图
├── CS336_Lecture/         # 课程原始材料
│   ├── TxtFile(CN)/       # 中文转录文稿
│   └── TxtFile(EN)/       # 英文转录文稿
├── spring2025-lectures/   # 课程Python代码
└── Prompts/               # 笔记生成提示词
```

---

## ✨ 更新日志

本项目记录自己在学习CS336时的过程，逼自己每天学习，不然实在是太懒了（但是依然进度很慢 悲）😂

- **2026-01-19**: 🎉 完成全部17讲笔记！新增Lectures 14-17（数据过滤去重、SFT/RLHF、RL算法、GRPO实现）
- **2025-10-15**: 又是忙忙碌碌的一个月，断断续续听了四天听完了Lecture7，比Lecture6好懂一点，但是一看到下一节课是手撕并行我就害怕了。看一遍有点印象得了
- **2025-10-01**: 听完了Lecture6，但是感觉这节课完全可以跳过，因为这么点时间对于我一个对CUDA PTX很陌生的人完全做不到入门，老师讲得太细，直接上代码，我跟不上(太菜了呜呜呜)。后面真的要进行性能优化就从Lecture5里整的 [GPU学习资源](./NoteByHuman/Lecture5/Lecture5-GPU学习资源.md) 重新学吧，这节课就把笔记写好听一遍跳过就得了
- **2025-09-30**: 出差途中忙里偷闲写完了Lecture5，依旧花了四个小时，不过这节课收获满满啊，GPU性能优化感觉已经入门了
- **2025-09-27**: 接近两个月没更新，但不是没学习，今天更新了Lecture4-MoE的完整笔记和Lecture5的部分内容(未校正)。老师语速感觉我能适应了
- **2025-07-31**: 七月最后一天，更新了第三课笔记(部分)，这堂课老师语速太快，信息密度太大，80分钟的视频我可能得花240分钟来看+做笔记
- **2025-07-30**: 被台风袭击的一天，开始做Assignment1了，顺便更新了Lecture3的笔记
- **2025-07-28**: 更新完成了第二课笔记，第二课有点混乱，听起来很吃力，需要多复盘和实践
- **2025-07-26**: 新增了AI辅助生成文稿的Prompt（虽然效果不尽人意，但也是一个有趣的尝试🤪）

---

## 🔗 相关资源

- 🎬 [CS336官方课程网站](https://stanford-cs336.github.io/spring2025/)
- 📺 [课程YouTube视频](https://www.youtube.com/@StanfordOnline)

---

## 🌌 写在最后

> **"路漫漫其修远兮，吾将上下而求索。"**
> 
> *The road ahead is long and winding, yet I shall search high and low in pursuit of truth.*
>
> — 屈原《离骚》

---

<div align="center">

💡 **如有问题或建议，欢迎提Issue！**

⭐ **觉得有帮助的话，点个Star吧！** ⭐

</div>
