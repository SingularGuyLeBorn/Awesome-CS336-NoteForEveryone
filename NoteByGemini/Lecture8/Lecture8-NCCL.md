# 专题笔记：NCCL (NVIDIA Collective Communications Library)

### 1. 简介
**NCCL** (发音为 "Nickel") 是 NVIDIA 提供的集合通信库，专门针对 NVIDIA GPU 进行了优化。它实现了标准的**[集合通信操作](./Lecture8-Collective-Operations.md)**（如 All-Reduce, All-Gather 等），旨在提供尽可能高的带宽和最低的延迟。

### 2. 核心功能
*   **拓扑感知**：NCCL 在初始化时会自动检测系统的硬件拓扑结构，包括 PCIe、NVLink、NVSwitch 以及网卡（NIC）的连接方式。
*   **路径优化**：基于探测到的拓扑，NCCL 会构建最优的通信环（Ring）或树（Tree）结构，以最大化利用硬件带宽。例如，在同一节点内优先使用 NVLink，跨节点使用 GPU Direct RDMA。
*   **CUDA 内核驱动**：通信操作最终转化为在 GPU 上执行的 CUDA 内核，直接负责数据的发送和接收，最大限度地减少 CPU 的介入。

### 3. 在 Pytorch 中的角色
PyTorch 的 **[`torch.distributed`](./Lecture8-Code-TorchDistributed.md)** 模块支持多种后端。当在 NVIDIA GPU 上进行分布式训练时，`nccl` 是首选且性能最好的后端。开发者通常不需要直接调用 NCCL API，而是通过 PyTorch 的高级 API 使用它。