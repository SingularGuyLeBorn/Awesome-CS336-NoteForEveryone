# 专题笔记：硬件通信层级 (Hardware Communication Hierarchy)

### 1. 概述
在深度学习中，计算性能往往受限于数据移动的速度。理解从芯片内部到跨节点的硬件通信层级，对于设计高效的并行策略至关重要。这个层级通常遵循“距离越近，速度越快，容量越小”的规律。

### 2. 层级结构 (从小/快 到 大/慢)

| 层级位置 | 存储/连接技术 | 典型带宽 (示例) | 特点 | 优化策略 |
| :--- | :--- | :--- | :--- | :--- |
| **单 GPU 内部** | **L1 缓存 / 共享内存** | 极高 (TB/s 级) | 位于 SM (Streaming Multiprocessor) 内部，速度最快，容量极小。 | 算子融合 (Fusion), 分块 (Tiling) |
| **单 GPU 内部** | **HBM (高带宽内存)** | ~1.5 - 3+ TB/s | GPU 的主显存，容量较大 (如 80GB)，但比 L1 慢。 | 提高算术强度，减少 HBM 读写 |
| **节点内跨 GPU** | **NVLink** | ~900 GB/s (H100 总带宽) | NVIDIA 专有高速互联，绕过 CPU 和 PCIe，直连 GPU。 | **[张量并行](./Lecture8-Tensor-Parallelism.md)** |
| **节点内 (传统)** | **PCIe 总线** | ~128 GB/s (PCIe 5.0 x16) | 通用总线，连接 CPU、GPU、网卡等。通常需要数据经过 CPU 内存。 | 尽量避免作为 GPU 间通信路径 |
| **跨节点** | **NVSwitch / InfiniBand** | 高 (数百 GB/s) | 专为高性能计算设计的网络架构，支持 GPU Direct RDMA。 | **[数据并行](./Lecture8-Data-Parallelism.md)**, **[流水线并行](./Lecture8-Pipeline-Parallelism.md)** |
| **跨节点 (传统)** | **以太网 (Ethernet)** | 低至中 (~12.5 - 100 GB/s) | 通用网络标准，延迟和带宽通常不如专用网络。 | 即使在分布式训练中也应尽量避免成为瓶颈 |

### 3. 对分布式训练的影响
*   **Tensor Parallelism** 需要极高频的通信，因此通常被限制在具有 **NVLink** 互联的单节点内部。
*   **Data Parallelism** 和 **Pipeline Parallelism** 的通信频率相对较低，可以跨越节点进行，利用 InfiniBand 或 NVSwitch 系统。
*   底层通信库如 **[NCCL](./Lecture8-NCCL.md)** 会自动探测硬件拓扑，选择最优的通信路径。