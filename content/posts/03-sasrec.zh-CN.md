---
title: "历史很长，真正相关的可能只有几步｜生成式推荐 03：SASRec"
date: "2026-07-27"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - SASRec
  - Transformer
pairKey: "generative-recommendation-03-sasrec"
slug: "generative-recommendation-03-sasrec-zh"
excerpt: "GRU 把历史逐步压进一个状态；SASRec 用因果自注意力让每个位置直接寻找相关行为，在长程表达与稀疏数据之间取得平衡。"
series: "generative-recommendation"
seriesOrder: 3
draft: false
---

小林最近有 50 条行为。她买过网球拍，看过羽毛球鞋，点开过泳镜，也曾在一个月前购买攀岩鞋。

现在她正在浏览粉袋。

哪一步最重要？

- 最后一次点击通常很有用；
- 一个月前的攀岩鞋可能更有因果联系；
- 中间那些网球与游泳行为也许只是噪声。

[上一篇 GRU4Rec](/blog/generative-recommendation-02-gru4rec-zh) 会按顺序把 50 步压进一个隐藏状态。理论上它能记住很久以前，实际却要让那条攀岩鞋信息经过几十次递归更新才能抵达现在。

[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)，即 SASRec，换了一条路：

> 不要求所有过去先挤进一个向量。让当前预测直接与历史每一步比较，再决定应该关注谁。

这篇 2018 年的论文把 Transformer 的 self-attention 引入序列推荐，也把 Item ID 又向“token”推进了一步。

## 30 秒看懂本文

1. **GRU4Rec 留下的问题**：递归状态难以并行，早期行为到当前输出的路径长，而且所有历史被压进同一向量。
2. **SASRec 的答案**：给 Item ID 加位置向量，用 causal self-attention 让位置 $t$ 只查看 $1,\ldots,t$，并自动分配不同注意力。
3. **最重要的证据**：在 Amazon Beauty/Games、Steam 和 MovieLens-1M 上，SASRec 同时领先稀疏数据上的简单模型与稠密数据上的 RNN/CNN；论文报告相对最强基线平均提升 6.9% Hit Rate 和 9.6% NDCG。

![从递归状态到直接查看历史](/blog/generative-recommendation/03-sasrec/paradigm-shift.svg)

*图 1：RNN 的信息沿时间逐步传递；self-attention 让当前位置用一跳路径访问所有过去位置。*

## 上一篇留下了什么问题

GRU 的状态更新可写成

$$
\mathbf h_t=f(\mathbf x_t,\mathbf h_{t-1}).
$$

这带来三个结构性限制：

1. **信息瓶颈**：$\mathbf h_t$ 要同时保存与未来所有可能请求有关的信息；
2. **最长路径 $O(n)$**：第一步影响第 $n$ 步，需要穿过 $n-1$ 次状态传递；
3. **时间依赖**：同一条序列的第 $t$ 步必须等待第 $t-1$ 步。

在另一端，一阶 Markov Chain 只看最后一次行为，路径短、参数少，在极稀疏数据上反而可能比复杂神经网络更稳。

SASRec 的目标是连接两者：

- 像 RNN 一样，有能力利用长期历史；
- 像 Markov Chain 一样，预测时可以只依赖少数真正相关的行为；
- 又能让序列中的所有训练位置并行计算。

## 用同一条历史理解注意力

假设小林的可见历史是：

```text
I01 网球拍 → I03 攀岩鞋 → I05 羽毛球
```

模型要预测下一项 `I06 泳镜`。SASRec 不会预先规定“只看最后一步”，也不会把三件商品等权平均。当前位置生成 query，每个历史位置提供 key 与 value：

- query 与 `I01` 的 key 匹配较弱，它对当前预测贡献较小；
- `I03` 可能提示最近兴趣从球类转向其他运动；
- `I05` 是最近行为，可能得到中等权重。

![SASRec 如何在同一历史中选择相关行为](/blog/generative-recommendation/03-sasrec/toy-attention.svg)

*图 2：权重仅用于解释机制，并非原论文对某个用户的真实数值。真正的 attention 由训练数据学习，且不同层会产生不同表示。*

有一个关键约束：预测位置 $t+1$ 时，模型不能看到 $t+1$ 之后的商品。否则训练阶段会偷看答案。因此 SASRec 使用 **causal mask**，把所有未来位置的 attention logit 设为负无穷。

这和 decoder-only 语言模型的左到右遮罩是同一个原则。

## 模型从输入到输出发生了什么

设用户序列为

$$
S^u=(S_1^u,S_2^u,\ldots,S_{|S^u|}^u),
$$

最大长度为 $n$，Item embedding 维度为 $d$。

### 1. 截断、padding 与位置

序列过长时只保留最近 $n$ 个行为；不足时在左侧 padding。每件商品查表得到向量，位置也有一个可学习向量：

$$
\mathbf E_t
=
\mathbf M_{S_t^u}
+
\mathbf P_t,
\qquad
\mathbf E\in\mathbb R^{n\times d}.
$$

Item ID 本身没有顺序含义。没有 $\mathbf P_t$，`网球→攀岩` 与 `攀岩→网球` 对 attention 来说只是一组向量。

### 2. Causal self-attention

线性投影得到

$$
Q=\mathbf E W^Q,\quad
K=\mathbf E W^K,\quad
V=\mathbf E W^V.
$$

attention 矩阵大小为 $n\times n$。第 $t$ 行表示当前位置怎样组合此前位置。

### 3. 残差、LayerNorm 与逐位置 FFN

attention 输出经过 dropout、残差连接和 LayerNorm，再经过相同地作用于每个位置的两层前馈网络。原论文默认堆叠两个 self-attention block。

### 4. 用隐藏表示给下一项打分

位置 $t$ 的表示 $\mathbf h_t$ 与候选商品 embedding 做点积：

$$
r_{i,t}=\mathbf h_t^\top\mathbf M_i.
$$

每个位置都以真实的下一件商品为正样本，并采一个未交互商品为负样本，用二元交叉熵训练。

![SASRec 的输入、因果注意力与下一项打分](/blog/generative-recommendation/03-sasrec/mechanism.svg)

*图 3：每个位置只读取它左侧的历史；输出仍然是候选商品分数，Item embedding 同时用于输入与预测层。*

核心 shape：

| 张量 | Shape | 含义 |
|---|---:|---|
| `item_ids` | $[B,n]$ | padding 后的行为序列 |
| `embeddings` | $[B,n,d]$ | Item + position |
| `attention_logits` | $[B,n,n]$ | 每个位置对历史位置的相关性 |
| `hidden` | $[B,n,d]$ | 每个时间步的上下文表示 |
| `positive_scores` | $[B,n]$ | 每步真实下一项分数 |
| `negative_scores` | $[B,n]$ | 每步采样负项分数 |

## 核心公式逐项拆解

SASRec 的中心操作是 scaled dot-product attention：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
+
M_{\text{causal}}
\right)V.
$$

逐项解释：

- $QK^\top$：每个当前位置与每个历史位置的匹配分；
- $\sqrt d$：维度大时点积方差也会变大，除以它避免 softmax 过早饱和；
- $M_{\text{causal}}$：未来位置为 $-\infty$，过去与当前位置为 0；
- softmax：每一行变成对可见历史的归一化权重；
- 乘 $V$：用这些权重聚合历史信息。

与 GRU 的区别不是“有无加权平均”这么简单。

GRU 的第一步信息到最后一步要通过整条递归链；self-attention 直接建立一条边。论文将最大路径长度概括为：

- RNN：$O(n)$；
- self-attention：$O(1)$。

代价是注意力矩阵需要 $O(n^2)$ 空间与

$$
O(n^2d+nd^2)
$$

级别计算。对论文里 $n=50$ 或 $200$ 的序列很可控，对今天数万行为的工业历史则必须使用稀疏注意力、分层压缩或检索。

## 训练和推理分别怎么做

训练时，对用户序列做一次右移：

```text
input : [PAD, I01, I03, I05]
target: [I01, I03, I05, I06]
```

每个非 padding 位置都贡献一个 next-item loss：

```text
for user sequence:
    keep the most recent n items and left-pad
    hidden = SASRec(item_embedding + position_embedding,
                    causal_mask=True)

    for every valid position t:
        positive = the real item at t + 1
        negative = sample one item not in the user sequence
        optimize BCE(score(hidden[t], positive),
                     score(hidden[t], negative))
```

推理时：

1. 截取用户最近 $n$ 个行为；
2. 一次并行计算全部位置表示；
3. 读取最后一个非 padding 位置 $\mathbf h_t$；
4. 与候选商品 embedding 点积；
5. 过滤已交互或不可用商品，取 Top-K。

SASRec 的训练在时间维并行，但推理仍可能需要全目录矩阵乘法。若商品数为 $|\mathcal I|$，最后一步约为 $O(|\mathcal I|d)$，除非接外部召回或近似最近邻索引。

## 实验究竟证明了什么

论文选择了稀疏程度差异很大的四个数据集：

| 数据集 | 用户数 | 商品数 | 人均行为 |
|---|---:|---:|---:|
| Amazon Beauty | 52,024 | 57,289 | 7.6 |
| Amazon Games | 31,013 | 23,715 | 9.3 |
| Steam | 334,730 | 13,047 | 11.0 |
| MovieLens-1M | 6,040 | 3,416 | 163.5 |

这组选择专门检验论文的主张：简单 Markov 方法往往在稀疏数据稳，复杂 RNN/CNN 在稠密数据强，SASRec 能否适应两端。

![SASRec 在四个数据集上的 NDCG@10](/blog/generative-recommendation/03-sasrec/evidence.svg)

*图 4：紫色是 SASRec，灰色是每个数据集上最强的神经基线。数值来自原论文 Table III。*

论文报告 SASRec 相对最强基线平均提升：

- **6.9% Hit Rate**；
- **9.6% NDCG**。

更具体地：

- Beauty NDCG@10 从最强神经基线 0.2556 提升到 **0.3219**；
- Steam 从 0.5595 提升到 **0.6306**；
- MovieLens-1M 从 0.5538 提升到 **0.5905**。

消融也很有价值：

- 去掉位置向量在最稀疏 Beauty 上略好，却在更稠密数据变差，说明顺序价值依赖数据；
- 输入与输出 Item embedding 不共享会在四个数据集上都明显下降；
- 去掉 residual connection、dropout 或所有 attention block 通常严重退化；
- 默认的单头 attention 在当时较小的 $d$ 下不输多头。

论文还报告在单张 GTX 1080 Ti 上，SASRec 模型更新约 1.7 秒/epoch，快于对照的 Caser 与 GRU4Rec+ 设置。这个数字不能直接迁移到现代系统，但支撑了“序列位置可并行”的工程动机。

## 它失败在哪里

### 1. 只能从左向右训练

因果 mask 保证不泄漏未来，这是优点，也意味着位置 $t$ 的表示永远不能利用右侧上下文。训练数据里明明知道完整历史，却只使用一半方向的信息。

### 2. 二次注意力无法无限延长

$n=200$ 时 $n^2$ 只有 4 万；$n=10,000$ 时就是 1 亿。SASRec 证明了 self-attention 在推荐中有效，没有解决超长工业历史。

### 3. Item ID 仍是私有 token

`I03` 与 `I04` 的 embedding 能从共现中学到关系，但 ID 本身没有文本语义。新商品没有交互时，模型无法仅凭“攀岩粉袋”这个标题理解它。

### 4. 负采样和 sampled evaluation 会改变难度

论文训练每个位置采一个负项；评估把真实下一项与 100 个随机负项排序。这比在完整商品库中检索容易，也会让不同论文的 Hit@10 不能直接横比。

### 5. Attention weight 不等于因果解释

权重高表示模型在当前计算中更依赖某个 value，不证明那次行为在现实中“导致”了推荐，更不保证权重就是忠实的人类解释。

### 6. 它仍然只优化下一项

每个位置预测一个 item，列表多样性、跨商品依赖、库存与长期目标仍在模型之外。它是强序列打分器，不是完整生成式系统。

## 下一篇为什么会出现

SASRec 已经让购物记录很像一句话：

- Item ID 是 token；
- position embedding 表示词序；
- causal self-attention 根据上文预测下一个 token。

但训练时我们其实拥有整条用户历史。假设序列是：

```text
网球拍 → 攀岩鞋 → 粉袋 → 攀岩裤
```

若把“粉袋”遮住，左右两侧的“攀岩鞋”和“攀岩裤”都能帮助恢复它。SASRec 因为只能看左边，没有使用右侧证据。

于是一个自然问题出现：

> 既然 Item ID 已经像词，能否把购物历史当句子，像 BERT 一样做完形填空？

下一篇：[BERT4Rec](https://arxiv.org/abs/1904.06690)——把商品当成词，把购物记录当成句子。

## 原始论文

- Wang-Cheng Kang, Julian McAuley. [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). ICDM 2018.

[返回完整路线图 →](/series/generative-recommendation)
