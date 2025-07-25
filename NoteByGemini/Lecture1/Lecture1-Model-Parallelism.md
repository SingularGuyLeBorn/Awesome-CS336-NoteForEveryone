# 专题：模型并行 (Model Parallelism)
## 1. 问题背景
随着**[语言模型](./Lecture1-Language-Models.md)**的规模越来越大，单个 GPU 的内存已经无法容纳整个模型的参数。例如，一个 175B（1750亿）参数的 LLaMA 模型，如果使用 FP16（每个参数 2 字节）存储，仅模型权重就需要 350 GB 内存，远超任何单个 GPU 的显存容量（如 A100/H100 的 80 GB）。
**模型并行**就是为了解决这个问题而提出的一系列技术。其核心思想是：**将单个巨大模型的不同部分（参数、计算、激活值）拆分到多个 GPU 上，让它们协同完成训练或推理任务。**
这与数据并行（Data Parallelism）形成对比。数据并行是在每个 GPU 上都保留一份完整的模型副本，然后将数据批次（batch）切分给不同的 GPU。当模型小到单 GPU 能放下时，数据并行是首选；当模型大到单 GPU 放不下时，就必须使用模型并行。
## 2. 主要技术分类
模型并行主要有以下几种形式：
### 2.1 张量并行 (Tensor Parallelism)
张量并行将模型中的单个大矩阵（如 Transformer 中的权重矩阵）切分到多个 GPU 上。例如，一个 `A * B = C` 的矩阵乘法，可以将权重矩阵 `B` 按列切分，分别放到两个 GPU 上。每个 GPU 计算部分结果，然后通过通信（如 All-Reduce）将结果合并。
*   **代表技术:** Megatron-LM。
*   **优点:** 计算和通信可以高度重叠，效率较高。
*   **缺点:** 通信开销随着并行度的增加而显著增加，通常局限于单个节点内的 GPU（通过 NVLink 高速互联）。
### 2.2 流水线并行 (Pipeline Parallelism)
流水线并行将模型的不同层（Layers）分配到不同的 GPU 上。例如，一个 48 层的模型，可以将 1-12 层放在 GPU 0，13-24 层放在 GPU 1，以此类推。数据像流水线一样依次流过这些 GPU。
*   **代表技术:** GPipe, PipeDream。
*   **挑战:** 存在“流水线气泡”（Pipeline Bubble）问题，即在流水线的开始和结束阶段，部分 GPU 处于空闲状态，导致效率下降。
*   **解决方案:** 使用微批次（Micro-batching）技术，将一个大批次切成多个小批次，让它们在流水线中交错执行，从而减少气泡，提高 GPU 利用率。
### 2.3 序列并行 (Sequence Parallelism)
序列并行是在张量并行的基础上，针对长序列场景的优化。它沿着序列长度维度对输入数据进行切分，从而减少每个 GPU 上存储的激活值（Activation）大小，因为激活值的大小与序列长度成正比。
*   **优点:** 能有效训练非常长的序列，这对于处理文档、书籍等长文本至关重要。
*   **应用:** 通常与张量并行结合使用。
### 2.4 专家并行 (Expert Parallelism)
这是针对**[混合专家模型 (MoE)](./Lecture1-Mixture-of-Experts.md)**的并行策略。在 MoE 模型中，每个 token 只会被路由到少数几个“专家”（即 FFN 层）进行计算。因此，可以将不同的专家分配到不同的 GPU 上。
*   **优点:** 可以在计算量几乎不变的情况下，将模型参数量扩大数倍。
*   **通信:** 需要 All-to-All 通信来在不同 GPU 的专家之间传递数据，这是其主要的通信瓶颈。
## 3. 统一框架：3D 并行与 ZeRO
现代的大规模训练框架通常会结合多种并行策略。
*   **3D 并行:** Megatron-LM 提出的方案，它将数据并行、张量并行和流水线并行结合起来，形成一个三维的并行策略，可以在数千个 GPU 上高效训练万亿参数级别的模型。
*   **ZeRO (Zero Redundancy Optimizer):** 由 Microsoft DeepSpeed 提出，它是一种更精细的内存管理策略。它不仅仅切分模型参数，还切分了优化器状态（Optimizer States）和梯度（Gradients），这些在训练中会占用大量内存。
    *   **ZeRO-1:** 切分优化器状态。
    *   **ZeRO-2:** 在 ZeRO-1 基础上，切分梯度。
    *   **ZeRO-3:** 在 ZeRO-2 基础上，切分模型参数本身。FSDP (Fully Sharded Data Parallelism) 是 PyTorch 中对 ZeRO-3 的官方实现。
## 4. 总结
模型并行是实现超大规模语言模型的关键使能技术。通过将张量、流水线、序列乃至专家并行等多种策略有机结合，并辅以像 ZeRO 这样的精细化内存管理方案，研究人员和工程师才得以在庞大的 GPU 集群上训练出如 **[GPT-4](./Lecture1-GPT-4.md)** 这样能力强大的模型。