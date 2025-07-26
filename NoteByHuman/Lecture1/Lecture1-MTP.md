# Multi-Token Predict
｜ 关于MTP的拓展阅读:  [https://zhuanlan.zhihu.com/p/18056041194](https://zhuanlan.zhihu.com/p/18056041194)
多token预测目标,**Multi-token Prediction**. DeepSeek-V3 创新性地采用了 MTP 目标,将预测范围扩展到每个位置的多个后续 token. 这种设计具有双重优势,首先,MTP 目标通过增加训练信号的密度可以提高数据利用效率;其次,它使模型能够提前规划表征,从而更准确地预测后续 token. 如下图所示,该实现方案与先前研究的方法有所不同: 前者使用独立输出头并行预测 D 个额外 token,而 DeepSeek-V3 采用顺序预测方式,并在每个预测层级保持完整的因果关系链 (Casual Chain)
具体实现: 使用了 D 个顺序 MTP Module 来预测 D 个额外的 token. 第k个 MTP 模块由**一个共享嵌入层Emb、一个共享输出头OutHead、一个Transformer模块以及一个投影矩阵组成**,具体公式如下: 
$$
h_i^k = M_k[RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]
$$
$$
h_{1:T-k}^k = TRM_k(h_{1:T-k}^{lk})
$$
$$
P_{i+k+1}^k = OutHead(h_i^k)
$$
最后的MTP训练目标,对于每一个 MTP Module为: 
$$
L_{MTP}^k = \text{CrossEntropy}(P_{2+k:T+1}^k, t_{2+k:T+1}) = -\frac{1}{T} \sum_{i=2+k}^{T+1} \log P_i^k[t_i]
$$
最后综合 MTP 目标的均值乘以系数作为 DeepseekV3的 $L_{main}$ 之外的额外训练目标: 
$$
L_{MTP} = \frac{\lambda}{D} \sum_{k=1}^{D} L_{MTP}^k
$$
