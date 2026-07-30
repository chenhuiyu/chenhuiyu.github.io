---
title: "Semantic ID 不是唯一答案｜生成式推荐 2026·03：ContRec"
date: "2026-07-30"
updated: "2026-07-30"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - ContRec
  - Diffusion
pairKey: "generative-recommendation-2026-03-contrec"
slug: "generative-recommendation-2026-03-contrec-zh"
excerpt: "ContRec 放弃离散 Semantic ID，以连续 Token 和扩散模型生成偏好向量，再检索真实商品。"
series: "generative-recommendation-2026"
seriesOrder: 3
draft: false
---


[CARE](/blog/generative-recommendation-2026-02-care-zh) 在离散 SID 的树上修建了更多桥梁，但 [ContRec](https://arxiv.org/abs/2504.12007) 提出一个更不客气的问题：

> 为什么生成式推荐一定要把商品压成离散 token？

Semantic ID 让 Transformer 能像生成文字一样生成商品，但量化会丢失细节，固定 codebook 也会制造碰撞、层级长度和词表容量之间的权衡。ContRec 选择另一条路线：**在连续空间里生成偏好，再把结果映射回商品库。**

## 30 秒看懂本文

1. **旧方法的问题**：离散量化把连续商品表示切成格子。相近商品可能被分开，不同商品也可能共享 code；更长 SID 提高表达力，却增加解码成本。
2. **ContRec 的答案**：用非量化 tokenizer 保留连续表示，再用条件扩散模型从噪声逐步生成下一商品的偏好向量。
3. **代价转移**：它避开了 codebook，却把复杂度搬到多步去噪、向量检索与实时索引。没有免费午餐，只是厨房换了位置。

![ContRec：从离散地址转向连续偏好](/blog/generative-recommendation-2026/03-contrec/mechanism.svg)

## 连续 Token 到底是什么

对攀岩鞋和防滑粉，离散 SID 可能是：

```text
攀岩鞋  <sport><shoe><climb>
防滑粉  <sport><gear><climb>
```

这种表示清晰，但颜色、硬度、场景和价格等细节最终被压进有限 code。ContRec 的 tokenizer 不做 hard quantization，而是保留连续 embedding \(e_i\in\mathbb{R}^d\)。

用户历史也被编码为条件 \(c_u\)。模型的目标不再是预测若干 code，而是生成一个连续向量 \(\hat e\)，使它落在用户下一步可能喜欢的商品附近。

## 扩散模型怎样生成推荐

训练时，从目标商品向量 \(e_0\) 加噪得到 \(e_t\)：

\[
e_t=\sqrt{\bar\alpha_t}e_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

去噪器接收 noisy item、时间步与用户条件：

\[
\hat\epsilon_\theta=\epsilon_\theta(e_t,t,c_u),
\]

并最小化噪声预测误差：

\[
\mathcal L_{\text{diff}}
=
\mathbb E\left[\lVert \epsilon-\hat\epsilon_\theta\rVert_2^2\right].
\]

推理时从高斯噪声开始，多次反向去噪得到 \(\hat e_0\)，再通过近邻检索返回真实商品：

\[
\hat i=\arg\max_{i\in\mathcal I}\operatorname{sim}(\hat e_0,e_i).
\]

所以 ContRec 仍然不是“凭空创造商品”。它生成的是一个连续查询，最终答案仍受商品库约束。

## LLM 在哪里

论文还利用 LLM 表达用户偏好，把自然语言层面的兴趣概括作为扩散条件之一。这个设计看起来很诱人：语言模型负责解释历史，扩散模型负责在连续商品空间落点。

但需要保持警惕。自然语言偏好描述可能只是辅助条件，并不自动成为忠实解释。要证明一句“用户正在转向室内攀岩”真正影响了向量，需要删除或替换这句条件，观察生成结果是否按预期移动。

## 实验应该怎样读

ContRec 在 LastFM、MovieLens-1M、Amazon Beauty 和 Games 等数据上与序列推荐、扩散推荐、LLM 推荐和 SID 模型比较，论文报告相对 TIGER 的平均提升，并分析扩散步骤、tokenizer 与语言条件。

有三个不能混淆的维度：

- **排名质量**：HR、NDCG 是否提升；
- **生成成本**：去噪步数、模型延迟与 GPU 吞吐；
- **检索成本**：连续向量怎样在实时商品库中做近邻搜索。

减少推理步数可能降低质量，增加步数又可能让它输给一次自回归解码。真正的工业比较必须把模型生成与向量索引的总延迟放在一起。

## 它失败在哪里

连续空间没有非法 SID，但会产生“落在商品之间”的向量。最近邻映射可能返回语义相近却库存不可用、价格不合适或重复曝光的商品。

此外，扩散模型生成一个向量天然适合 next-item，却不直接解决整页商品的顺序、互补与去重。若分别采样多个向量，列表内部可能缺乏协调；若联合生成整页向量，维度与采样成本又会膨胀。

ContRec 最重要的意义不是宣布扩散击败 Transformer，而是证明：

> Semantic ID 是一种工程接口，不是生成式推荐的逻辑定律。

下一篇 [DualFashion](/blog/generative-recommendation-2026-04-dualfashion-zh) 会把连续生成再往前推一步：模型不只生成“应该检索哪个商品”的向量，而是直接生成一件尚不存在的服装及其描述。

[上一篇：CARE](/blog/generative-recommendation-2026-02-care-zh) · [返回第二季专题页](/series/generative-recommendation-2026)
