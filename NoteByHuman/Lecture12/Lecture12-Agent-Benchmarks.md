# 深入探讨: Agent 基准 - SWE-Bench 与 CyBench

**来源**: CS336 Lecture 12 · **主题**: 新兴 Agent 评估范式

---

## 1. Agent 评估的新范式

传统基准评估的是模型对**单个问题**的回答. 但许多实际任务需要:

1. **工具使用**: 运行代码、访问 API、搜索互联网
2. **多轮迭代**: 尝试 → 观察结果 → 调整策略 → 再次尝试
3. **长期规划**: 将复杂任务分解为子任务

这就是 **Agent 基准**的设计初衷.

**Agent = 语言模型 + Agent 脚手架** (决定如何调用 LM 的程序逻辑)

---

## 2. SWE-Bench: 软件工程 Agent

### 2.1 任务描述

给定:
- 一个 Python 代码仓库
- 一个 GitHub Issue 描述 (Bug 报告或功能请求)

任务: 提交一个 **Pull Request** (代码补丁) 来解决 Issue.

### 2.2 数据集

- **来源**: 12 个流行的 Python 仓库 (Django, Flask, requests, etc.)
- **规模**: 2294 个任务
- **评估指标**: 相关的**单元测试是否通过**

### 2.3 典型 Agent 流程

```
1. 阅读 Issue 描述
2. 探索代码库结构
3. 定位相关文件
4. 理解现有代码逻辑
5. 生成修复补丁
6. (可选) 运行测试验证
7. 提交补丁
```

### 2.4 变体

- **SWE-Bench Verified**: OpenAI 修复了原始数据集中的一些错误
- **SWE-Bench Lite**: 较小的子集, 便于快速评估

---

## 3. CyBench: 网络安全 Agent

### 3.1 任务描述

**Capture The Flag (CTF)** 风格的网络安全挑战:

- Agent 可以访问一台服务器
- 目标: 通过各种渗透技术获取 Secret Key (Flag)
- 成功获取 Flag = 任务完成

### 3.2 数据集

- **规模**: 40 个 CTF 任务
- **难度衡量**: 人类团队的**首次解决时间 (First-Solve Time)**
- **范围**: 最简单的几分钟, 最难的需要 24 小时

### 3.3 Agent 架构

CyBench 论文中描述的标准 Agent 循环:

```python
while not (solved or timeout):
    # 1. 思考当前状态
    thought = llm.think(memory, observation)
    
    # 2. 制定计划
    plan = llm.plan(thought)
    
    # 3. 生成命令
    command = llm.generate_command(plan)
    
    # 4. 执行命令
    observation = execute(command)
    
    # 5. 更新记忆
    memory.update(command, observation)
    
    # 6. 检查是否获取 Flag
    if "flag" in observation:
        solved = True
```

### 3.4 当前结果

- 最好的模型: ~20% 解决率
- O3 能解决人类需要 42 分钟的问题
- 最难的 24 小时问题仍未被 Agent 解决

---

## 4. 双重用途争议

CyBench 引发了一个有趣的讨论:

**这是安全评估还是能力评估?**

- **安全视角**: 检测模型是否能被用于恶意黑客攻击
- **能力视角**: 强大的渗透测试 Agent 可以帮助企业发现漏洞

美国 AI 安全研究所使用 CyBench 作为**安全评估**, 但本质上它也是一种**能力评估**.

> **核心洞察**: 能力和危险往往是一枚硬币的两面.

---

## 5. Agent 评估的挑战

### 5.1 可复现性

- Agent 行为依赖于随机性 (温度采样)
- 外部环境可能变化 (API 版本、网络状态)
- 需要多次运行取平均

### 5.2 成本

- 一次 SWE-Bench 评估可能需要数千次 LLM 调用
- 成本可达数百甚至数千美元

### 5.3 评估对象

- 评估的是**模型**还是**脚手架**?
- 不同的 Agent 架构会导致不同结果
- 如何公平比较?

---

## 6. 参考资料

- [SWE-Bench 论文](https://arxiv.org/abs/2310.06770)
- [CyBench 论文](https://arxiv.org/abs/2408.08926)
- [MLE-Bench 论文](https://arxiv.org/abs/2410.07095)
