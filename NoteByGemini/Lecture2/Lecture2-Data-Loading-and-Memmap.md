# 专题笔记: 数据加载与内存映射 (Data Loading & Memmap)

### 1. 核心问题: 大规模数据集的挑战

在训练大型语言模型时,我们面对的数据集通常是海量的. 例如,The Pile 数据集超过 800GB,C4 数据集约 750GB,而 Llama 系列模型的训练数据更是达到了 TB 级别. 将如此庞大的数据集一次性加载到计算机的 RAM 中是完全不可行的,因为典型的服务器 RAM 可能只有几十到几百 GB. 

如何高效地从硬盘读取数据,并将其提供给 GPU 进行训练,成为了一个关键的工程问题. 

### 2. PyTorch 的标准数据加载机制: `Dataset` 和 `DataLoader`

**[PyTorch](./Lecture2-PyTorch.md)** 提供了一套优雅且高效的数据加载工具: 

*   **`torch.utils.data.Dataset`**: 这是一个抽象类,你需要继承它并实现两个核心方法: 
    *   `__len__(self)`: 返回数据集的总样本数. 
    *   `__getitem__(self, idx)`: 根据给定的索引 `idx`,从数据源(如硬盘上的文件)加载并返回一个样本. 

*   **`torch.utils.data.DataLoader`**: 它接收一个 `Dataset` 对象,并在此基础上提供了许多强大的功能: 
    *   **批处理 (Batching)**: 自动将单个样本组合成一个批次(batch). 
    *   **数据打乱 (Shuffling)**: 在每个训练周期(epoch)开始时,随机打乱数据的顺序,这对于模型的泛化至关重要. 
    *   **并行加载 (Parallel Loading)**: 通过设置 `num_workers > 0`,`DataLoader` 会启动多个并行的 Python 进程,在 GPU 正在处理当前批次数据时,在后台预先加载和处理好接下来的几个批次. 这极大地减少了数据加载成为训练瓶颈的可能性,是提升**[MFU](./Lecture2-MFU.md)**的关键技巧之一. 

### 3. 内存映射 (Memory-Mapping): 处理超大文件的利器

即使有了 `DataLoader`,我们仍然面临一个问题: 如何在 `__getitem__` 中高效地访问一个巨大的文件(例如,一个包含数万亿个 token 的二进制文件)的任意部分,而不需要将整个文件读入内存？

答案是**内存映射 (Memory-Mapping)**,在 **[NumPy](./Lecture2-NumPy.md)** 中可以通过 `np.memmap` 实现. 

*   **核心思想**: 内存映射是一种操作系统级别的技术,它并**不会立即将整个文件加载到 RAM 中**. 相反,它会在进程的虚拟地址空间中为文件预留一块连续的区域,使得这个文件看起来就像一个内存中的数组. 
*   **按需加载**: 当你访问这个“虚拟数组”的某个切片时(例如 `data[1000:2048]`),操作系统会自动处理底层的 I/O 操作,只将你需要的这部分数据从硬盘加载到物理 RAM 中. 这个过程对用户是透明的. 
*   **优势**: 
    *   **极低的启动开销**: 几乎可以瞬时“打开”一个 TB 级的文件. 
    *   **随机访问效率高**: 可以像操作内存数组一样,高效地随机访问文件的任何部分. 
    *   **内存高效**: RAM 中只保留了当前需要和最近使用的数据页(由操作系统缓存管理). 

**在 LLM 训练中的应用: **
对于一个巨大的、经过分词和序列化的 token 数据集(通常存为一个巨大的 `np.uint16` 数组文件),我们可以这样做: 

1.  使用 `np.memmap` 以只读模式打开这个文件,得到一个看起来像 NumPy 数组的对象. 
2.  在我们的自定义 `Dataset` 的 `__init__` 方法中加载这个 `memmap` 对象. 
3.  在 `__getitem__` 方法中,根据索引 `idx` 计算出需要的数据切片(例如,从 `idx` 开始,长度为 `sequence_length`),然后直接从 `memmap` 对象中切片. 
4.  操作系统会负责将这一小块数据从硬盘加载到内存,然后 `DataLoader` 的 worker 进程将其转换为 PyTorch **[张量](./Lecture2-Tensors.md)** 并发送给主进程. 

**代码示例 (概念性):**
```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 假设我们有一个巨大的二进制文件 'train.bin',存的是 uint16 类型的 token
# 文件大小可能是 2TB

class LargeTokenDataset(Dataset):
    def __init__(self, data_path, sequence_length):
        super().__init__()
        self.seq_len = sequence_length
        # 使用内存映射打开文件,并不会消耗 2TB 的 RAM
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

    def __len__(self):
        # 返回可以构成的样本总数
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 按需从 memmap 对象中切片,OS 会负责从硬盘读取
        x = torch.from_numpy((self.data[idx:idx+self.seq_len]).astype(np.int64))
        y = torch.from_numpy((self.data[idx+1:idx+self.seq_len+1]).astype(np.int64))
        return x, y

# 使用
dataset = LargeTokenDataset('train.bin', sequence_length=2048)
dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=True)

# 训练循环
for batch_x, batch_y in dataloader:
    # ... 进行训练 ...
    pass
```

通过结合 `DataLoader` 的并行加载和 `memmap` 的高效文件访问,我们可以构建出一个能够流畅地为 GPU 提供海量训练数据的强大数据流水线. 

---
**关联知识点**
*   [PyTorch](./Lecture2-PyTorch.md)
*   [NumPy](./Lecture2-NumPy.md)
*   [MFU (模型FLOPS利用率)](./Lecture2-MFU.md)