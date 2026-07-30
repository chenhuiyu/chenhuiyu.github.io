---
title: "Tokenizer 终于能听见推荐损失｜生成式推荐 2026·01：DIGER"
date: "2026-07-30"
updated: "2026-07-30"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - DIGER
  - Semantic ID
pairKey: "generative-recommendation-2026-01-diger"
slug: "generative-recommendation-2026-01-diger-zh"
excerpt: "DIGER 让 Semantic ID 在推荐训练中保持可微，用 Gumbel 探索和不确定性衰减缓解 codebook 塌缩。"
series: "generative-recommendation-2026"
seriesOrder: 1
draft: false
---


第一季停在 [OneReason](/blog/generative-recommendation-20-onereason-zh)：模型已经能生成商品 code、推荐理由，甚至一段“思考过程”。但整个系统仍有一块地基几乎没有被下游任务触碰：

> 商品的 Semantic ID，通常在推荐模型开始训练之前就已经被冻结了。

这像让制砖厂只按“砖块是否还原泥土纹理”优化，而建筑师只能接受现成砖块。砖很漂亮，不代表适合盖房子。[DIGER](https://arxiv.org/abs/2601.19711) 的问题非常直接：**推荐损失能不能反向改变商品 ID 本身？**

## 30 秒看懂本文

1. **旧方法的问题**：RQ-VAE 根据内容重构学习 Semantic ID，推荐器根据点击序列学习下一商品。两个目标各自优化，推荐梯度在 tokenizer 前被截断。
2. **DIGER 的答案**：前向仍选择离散 code，反向却通过 Gumbel-Softmax 的软概率更新整个 codebook，让推荐损失穿过离散选择。
3. **真正的难点**：可微不等于稳定。直接用 STE 很容易在训练早期押注少数 code，造成 codebook collapse。DIGER 用早期探索、后期收敛的不确定性衰减来控制这场“编码淘金热”。

![DIGER：推荐梯度穿过 Semantic ID](/blog/generative-recommendation-2026/01-diger/mechanism.svg)

## 用八件商品看懂目标错位

沿用第一季的玩具商品：网球拍、网球、攀岩鞋、防滑粉、羽毛球拍、羽毛球、泳帽、速干毛巾。

只看内容，网球拍和羽毛球拍最相似；只看行为，攀岩鞋和防滑粉可能更常被连续点击。传统两阶段流程会先问：

```text
“哪两个商品的文本与图片最像？”
```

再让推荐器适应这些固定 ID。DIGER 希望 tokenizer 同时听见另一个问题：

```text
“哪种编码更有利于预测用户下一步会选什么？”
```

因此，内容相似仍是约束，但不再是商品语言的唯一立法者。

## 模型从输入到输出发生了什么

对商品 \(v\) 的第 \(j\) 层 code，编码器先计算它与 \(K\) 个 codebook 向量的相似度：

\[
\ell_{v,j,i}=\operatorname{sim}(r_{v,j},e_i).
\]

然后加入 Gumbel 噪声并得到软分布：

\[
\tilde y_{v,j,i}
=
\frac{\exp((\ell_{v,j,i}+g_{v,j,i})/\tau)}
{\sum_{k=1}^{K}\exp((\ell_{v,j,k}+g_{v,j,k})/\tau)}.
\]

前向传播使用

\[
c_{v,j}=\arg\max_i(\ell_{v,j,i}+g_{v,j,i}),
\]

所以推荐器看到的仍是一串真正的离散 token。反向传播则使用软 embedding

\[
\bar e_{v,j}=\sum_i \tilde y_{v,j,i}e_i,
\]

于是每个候选 code 都能按概率获得梯度，而不是只有被 argmax 选中的那个 code 被更新。

这是一座双层桥：**前向保持离散，反向保持可导。**

## 为什么 Gumbel 噪声不是装饰

如果训练一开始就做确定性 hard assignment，某几个 code 会因为偶然领先而获得更多梯度，继而更常被选择，形成“越热门越热门”的自我强化。最终大量 code 闲置，商品被挤进少数拥堵地址。

DIGER 的 DRIL 先让模型探索多个可能编码，再逐步降低随机性。论文提供两种不确定性衰减：

- **SDUD**：让噪声尺度随生成损失下降，当模型越来越确定时，编码也越来越接近推理期的确定性选择；
- **FrqUD**：只对使用过热的 code 注入更多探索，低频 code 保持稳定，直接围绕 codebook 利用率调度。

关键不是“噪声越大越好”，而是训练早期需要搜索，训练后期必须收口。

## 训练与推理的差别

训练时，推荐损失、重构损失和量化约束共同更新 tokenizer 与生成器。推理时则不再采样 Gumbel 噪声，商品对应稳定的 hard SID，再由自回归模型逐 token 生成。

这也暴露了工程难题：

- codebook 更新后，历史日志里的旧 SID 是否仍兼容；
- 新商品加入时，能否局部编码而不重建全库；
- 线上缓存、索引和模型版本怎样共同迁移；
- 一个更好的离线 codebook，是否会在大规模 beam search 中保持收益。

## 实验究竟证明了什么

论文在 B-Shop、I-Shop 和 Yelp 等公开数据上比较两阶段方法、直接 STE 以及 DIGER 变体，并报告 Recall@10、NDCG@10 与 code 使用稳定性。结果支持两个判断：

1. 让推荐梯度影响 SID，确实比完全冻结 tokenizer 更有潜力；
2. “可微”本身不是答案，探索策略与 codebook 平衡决定训练是否会崩。

但这些实验仍是公开数据上的离线 next-item 任务。它们没有证明 SID 在线频繁变化时的兼容成本，也没有证明端到端重编码能在真实商品库中长期稳定运行。

## 我认为它最重要的贡献

DIGER 改变的是模块边界。Tokenizer 不再是数据预处理脚本，而成为推荐模型的一部分。

这一步很小，也很危险。小在于公式仍围绕 Gumbel-Softmax 与自回归损失；危险在于一旦商品 ID 可以被推荐目标重写，索引、日志、缓存和实验可比性都会被卷入训练闭环。研究问题从“怎样学一个好 ID”升级为：

> 怎样让商品语言持续学习，又不让整个系统每天换一种方言？

下一篇 [CARE](/blog/generative-recommendation-2026-02-care-zh) 会接住另一个问题：即使 ID 已经更合理，逐 token 生成仍可能把第一个小错误放大成整条错误路径。

[返回第二季专题页](/series/generative-recommendation-2026)
