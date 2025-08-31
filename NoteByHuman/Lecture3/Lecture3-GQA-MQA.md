# 精英笔记:分组查询与多查询注意力 (GQA/MQA)

**分组查询注意力(Grouped-Query Attention, GQA)**和**多查询注意力(Multi-Query Attention, MQA)**是现代大型语言模型中至关重要的架构优化,其目标并非提升模型性能,而是**大幅降低推理成本,提高生成速度**. 要理解GQA/MQA,必须首先区分模型在训练和推理两种模式下的计算瓶颈.

### 1. 瓶颈的根源:自回归推理与KV缓存

- **训练模式**: 在训练时,模型可以一次性处理整批(batch)的长序列. 计算是高度并行的,瓶颈主要在于浮点运算能力(FLOPs). 此时的算术强度(每次内存访问对应的计算量)很高,GPU的计算单元能被充分利用.
- **推理模式 (自回归生成)**: 在生成文本时,模型必须逐个token进行. 流程如下:

  1. 模型处理当前所有上下文,生成下一个token.
  2. 将新生成的token添加到上下文中.
  3. 重复此过程,直到生成结束符.

  为了避免每次都重新计算整个上下文的注意力,模型会使用**KV缓存(KV Cache)**. 即,将过去所有token的键(Key)和值(Value)向量存储起来. 在生成新token时,只需要计算当前token的查询(Query)向量,并让它与缓存中所有的K向量进行交互.

  ![KV Cache Animation](https://storage.googleapis.com/static.a-b-c/project-daedalus/L3-P60.png)

  这个过程暴露了新的瓶颈:随着生成序列变长,KV缓存会变得异常庞大. 在每一步生成中,都需要从GPU内存中完整地读出这个巨大的KV缓存. 这导致**内存带宽(Memory Bandwidth)**成为瓶颈,而不是计算能力. 此时的算术强度极低,GPU的大部分时间都在等待数据,而非计算.

### 2. 多查询注意力 (MQA):根本性的解决方案

标准的多头注意力(Multi-Head Attention, MHA)中,`N`个查询头(Q heads)对应着`N`个独立的键/值头(K/V heads). 这意味着KV缓存的大小与头的数量`N`成正比.

MQA的核心思想是:让所有的查询头共享同一组键/值头.

![MQA Diagram](https://storage.googleapis.com/static.a-b-c/project-daedalus/L3-P62.png)

- **结构**: 仍然有`N`个独立的Q头,但只有一个K头和一个V头.
- **效果**:
  - KV缓存的大小被**急剧压缩**了`N`倍.
  - 推理时需要从内存中读写的数据量大幅减少.
  - 内存带宽瓶颈得到极大缓解,从而显著提升了模型的生成吞吐量(每秒生成的token数).

MQA是一种非常激进的优化,它可能会对模型性能造成轻微的损失,因为所有查询头被迫从相同的K/V“知识库”中提取信息,降低了多样性.

### 3. 分组查询注意力 (GQA):平衡性能与效率

GQA是MHA和MQA之间的一个灵活折中方案.

GQA的核心思想是:将`N`个查询头分成`G`个组,每组内的查询头共享同一组K/V头.

![GQA Diagram](https://storage.googleapis.com/static.a-b-c/project-daedalus/L3-P63.png)

- **结构**: `N`个Q头,`G`个K/V头(`1 < G < N`).
- **特例**:
  - 当`G = N`时,GQA等价于MHA.
  - 当`G = 1`时,GQA等价于MQA.
- **效果**:
  - GQA提供了一个可以在模型性能和推理效率之间进行权衡的“旋钮”.
  - 相比MHA,它通过共享K/V头来压缩KV缓存,提升推理速度.
  - 相比MQA,它保留了多组K/V头,赋予模型更强的表达能力,从而更好地保持了性能.

![GQA Performance vs Speed Graph](https://storage.googleapis.com/static.a-b-c/project-daedalus/L3-P64.png)

上图清晰地展示了这种权衡:

- **时间(右图)**: MQA(橙色)的速度最快,MHA(红色)最慢,GQA(蓝色)居中,且随着分组数(GQA groups)减少而变快.
- **性能(左图)**: MHA的性能最高,MQA最低,GQA则在两者之间.

由于其出色的平衡能力,GQA已经成为现代LLM的标准配置,它使得在保持高性能的同时,也能实现可接受的推理服务成本.
