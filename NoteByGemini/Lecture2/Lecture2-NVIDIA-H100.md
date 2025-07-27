# 专题笔记: NVIDIA H100 GPU

### 1. 概述

NVIDIA H100 Tensor Core GPU 是基于 NVIDIA **Hopper 架构**的高性能计算卡,是当前(截至2025年)用于大规模人工智能(AI)和高性能计算(HPC)工作负载的旗舰级产品. 它是其前代产品 A100 (Ampere 架构) 的重大升级,专为加速万亿参数级别的大型语言模型(LLM)和复杂的科学计算而设计. 

### 2. 关键技术特性

H100 的强大性能源于其架构上的多项创新: 

*   **第四代 Tensor Cores 与 Transformer 引擎**: 
    *   Tensor Cores 是 NVIDIA GPU 中专门用于执行矩阵乘加运算的硬件单元,这是深度学习计算的核心. H100 的 Tensor Cores 性能得到了大幅提升. 
    *   **Transformer 引擎**是 H100 的一项革命性创新. 它能够动态地、智能地在 **[FP8](./Lecture2-FP32-FP16-BF16-FP8.md)** 和 FP16 精度之间进行切换. 通过利用 FP8 的极高吞吐量进行计算,同时保持关键部分的 FP16 精度以维持模型的准确性,Transformer 引擎可以将 LLM 的训练和推理速度提升数倍. 

*   **FP8 数据类型支持**: 
    *   H100 是首款原生支持 **FP8 (8位浮点数)** 数据类型的 GPU. 相比 **[BF16/FP16](./Lecture2-FP32-FP16-BF16-FP8.md)**,FP8 将数据位宽减半,从而使计算吞吐量和内存带宽效率翻倍. 这对于处理巨大的 LLM 模型至关重要. 

*   **高带宽内存 (HBM3)**: 
    *   H100 采用了 HBM3 内存技术,其 PCIe 版本的内存带宽高达 **2TB/s**,而 SXM 版本的带宽更是达到了惊人的 **3.35TB/s**.  高内存带宽意味着 GPU 能以更快的速度喂给计算核心数据,这对于内存密集型的 LLM 来说是性能的关键瓶颈之一. 

*   **第四代 NVLink 和 NVLink Switch 系统**: 
    *   **NVLink** 是 NVIDIA 开发的高速 GPU 间互联技术. H100 的 NVLink 带宽高达 900 GB/s,远超传统的 PCIe Gen5 (128 GB/s). 
    *   **NVLink Switch 系统**允许将多达 256 个 H100 GPU 连接成一个统一的、高速的计算结构,就像一个巨大的单一 GPU 一样. 这极大地简化了**[模型并行](./Lecture2-Model-Parallelism.md)**和数据并行的实现,是训练超大规模模型的关键基础设施. 

*   **第二代多实例 GPU (MIG)**: 
    *   MIG 技术允许将一块 H100 物理 GPU 安全地分割成多达七个独立的、拥有独立计算和内存资源的 GPU 实例. 这使得 GPU 资源可以被更细粒度地分配给不同的用户或任务,提高了 GPU 的利用率,特别是在推理场景下. 

### 3. 性能规格(以 PCIe 版本为例)

*   **工艺**: 台积电 4N 定制工艺,拥有 800 亿个晶体管. 
*   **内存**: 80 GB HBM3. 
*   **内存带宽**: 2 TB/s. 
*   **功耗**: 最大 350W. 
*   **理论峰值算力 (TFLOPS)**: 
    *   FP64: 24 TFLOPS
    *   FP32: 48 TFLOPS
    *   **TF32 Tensor Core**: 800 TFLOPS (带稀疏性) / 400 TFLOPS (稠密)
    *   **BF16/FP16 Tensor Core**: 1600 TFLOPS (带稀疏性) / 800 TFLOPS (稠密)
    *   **FP8 Tensor Core**: 3200 TFLOPS (带稀疏性) / 1600 TFLOPS (稠密)

*注: 稀疏性(Sparsity)是指利用模型权重中的 2:4 结构化稀疏特性来获得性能翻倍的技术,但在通用场景下,通常参考稠密(Dense)计算的性能. *

### 4. 在课程中的意义

在 CS336 课程中,H100 代表了当前可用于训练大模型的顶级硬件. 理解其核心特性,如对 FP8 和 BF16 的支持、高内存带宽以及 MFU 的概念,是进行所有“餐巾纸数学”估算、理解性能瓶颈和设计高效训练策略的基础. 讲座中所有关于性能和内存的计算,都是围绕这类现代 GPU 的规格展开的. 

---
**关联知识点**
*   [FLOPS (浮点运算)](./Lecture2-FLOPS.md)
*   [MFU (模型FLOPS利用率)](./Lecture2-MFU.md)
*   [FP32 / FP16 / BF16 / FP8](./Lecture2-FP32-FP16-BF16-FP8.md)
*   [模型并行 (Model Parallelism)](./Lecture2-Model-Parallelism.md)