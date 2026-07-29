---
title: "把商品当成词，把购物记录当成句子｜生成式推荐 04：BERT4Rec"
date: "2026-07-28"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - BERT4Rec
  - Transformer
pairKey: "generative-recommendation-04-bert4rec"
slug: "generative-recommendation-04-bert4rec-zh"
excerpt: "SASRec 只从左向右预测下一项；BERT4Rec 随机遮住历史商品，用双向上下文做完形填空，让 Item ID 真正成为可预测的 token。"
series: "generative-recommendation"
seriesOrder: 4
draft: false
---

请补全这段购物记录：

```text
网球拍 → [MASK] → 羽毛球 → [MASK]
```

如果答案分别是“攀岩鞋”和“泳镜”，模型不仅要看遮罩左边，也可以看右边。

[上一篇 SASRec](/blog/generative-recommendation-03-sasrec-zh) 已经把 Item ID 放进 Transformer：商品像 token，行为顺序像句子，模型根据前缀预测下一项。但训练数据里明明保存着完整历史，causal mask 却要求每个位置永远忽略右侧。

[BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690) 提出：

> 不再让每个位置只预测它的下一项。随机遮住历史中的若干商品，让模型同时利用左右行为把它们恢复出来。

Item ID 到这里不再只是一个被打分的数据库主键。它真正成为了一种可以被 mask、被预测、被上下文定义的 token。

## 30 秒看懂本文

1. **SASRec 留下的问题**：左到右训练只使用每个位置左侧的历史；表示能力受到单向约束，一条序列每轮提供的监督也有限。
2. **BERT4Rec 的答案**：随机把比例 $\rho$ 的 Item ID 替换成 `[MASK]`，用双向 Transformer 恢复所有被遮商品，即推荐版 Cloze task。
3. **最重要的证据**：论文用“BERT4Rec (1 mask)”隔离双向结构收益，再用多 mask 版本测额外监督收益；在 Beauty 与 ML-1M 上，两部分都带来提升。

![从 causal next-item prediction 到双向 Cloze](/blog/generative-recommendation/04-bert4rec/paradigm-shift.svg)

*图 1：训练时，SASRec 只看左侧；BERT4Rec 在历史内部双向补空。线上预测仍然只能使用请求发生前的历史。*

## 上一篇留下了什么问题

SASRec 对序列

```text
I01 → I03 → I05 → I06
```

构造的监督近似是：

```text
I01      预测 I03
I01 I03  预测 I05
I01 I03 I05 预测 I06
```

这符合线上因果性，却有两个训练问题。

### 1. 表示是单向的

在完整日志里，`I03` 的左右行为可能共同解释它是什么样的兴趣节点。只看左边，模型无法利用未来发生过的 `I05`、`I06` 帮助学习 `I03` 的上下文表示。

### 2. 深层单向模型可能“目标泄漏”

不能简单地去掉 causal mask。多层 self-attention 中，如果所有位置都能自由互看，用位置 $t$ 的原始 Item ID 预测位置 $t$，模型会沿残差或间接路径看到答案，训练退化成抄写。

BERT4Rec 的解决方式和 BERT 一样：先把目标 token 从输入中拿走，再允许双向查看其余上下文。

## 用完形填空理解 Cloze

假设小林的历史是：

```text
I01 网球拍 → I03 攀岩鞋 → I05 羽毛球 → I06 泳镜
```

一次 corruption 随机选中第二、第四个位置：

```text
输入：[I01, MASK, I05, MASK]
标签：MASK₁=I03, MASK₂=I06
```

![BERT4Rec 的多位置 Cloze 训练例子](/blog/generative-recommendation/04-bert4rec/toy-cloze.svg)

*图 2：只有 mask 位置计算目标 loss；未被遮住的行为共同构成双向上下文。*

这里要区分三个概念：

- **mask 数量**：一条序列每轮遮几个位置；
- **mask 比例 $\rho$**：遮住位置占有效历史的比例；
- **模型方向**：被遮位置可以同时注意左侧和右侧。

多 mask 不只是数据增强。一条长度 $n$ 的序列，传统 next-item 训练能产生约 $n$ 个前缀目标；BERT4Rec 在不同 epoch 使用不同 mask 组合，可以看到更多上下文—目标配对。

但遮得越多不一定越好。$\rho$ 太小，监督稀少；太大，模型缺少足够上下文。论文在不同数据集上单独调节 $\rho$。

## 模型从输入到输出发生了什么

### 1. 构造被污染的序列

对用户历史 $S^u$ 随机选出位置集合 $S_m^u$，用特殊 token `[MASK]` 替换，得到 $S^{u\prime}$。

### 2. 双向 Transformer Encoder

Item embedding 与 position embedding 相加后，经过 $L$ 层双向 self-attention。与 SASRec 不同，这里没有 causal mask：

$$
\mathbf H^L\in\mathbb R^{n\times d}.
$$

每个被遮位置的最终表示都融合左右上下文。

### 3. 只在 mask 位置恢复 Item ID

将 $\mathbf h_m^L$ 送入输出层，得到商品词表上的概率：

$$
P(v\mid S^{u\prime},m).
$$

### 4. 线上在末尾追加 `[MASK]`

真实推荐请求没有未来行为。推理时，把 `[MASK]` 放在历史末尾：

```text
[I01, I03, I05, MASK]
```

读取最后位置的 logits，选出下一项。论文还在训练中加入“只 mask 最后一项”的样本，让训练任务更接近 serving。

![BERT4Rec 训练与线上推理的数据流](/blog/generative-recommendation/04-bert4rec/mechanism.svg)

*图 3：训练随机恢复历史内部多个位置；推理只恢复末尾位置。双向模型没有穿越真实时间。*

若 batch size 为 $B$：

| 张量 | Shape | 含义 |
|---|---:|---|
| corrupted IDs | $[B,n]$ | 含多个 `[MASK]` 的历史 |
| attention | $[B,h,n,n]$ | $h$ 个双向 attention head |
| hidden | $[B,n,d]$ | 所有位置的上下文表示 |
| mask positions | $[M,2]$ | batch 内共 $M$ 个目标坐标 |
| item logits | $[M,|\mathcal I|]$ | 每个 mask 的商品分布 |

实际实现常只 gather 被 mask 的 $M$ 个 hidden，再做输出投影，避免为无监督位置保留不必要 logits。

## 核心公式逐项拆解

BERT4Rec 的 Cloze loss 是所有被遮商品的负对数似然：

$$
\mathcal L
=
-
\sum_{m\in S_m^u}
\log
P\left(v_m^\ast\mid S^{u\prime}\right).
$$

- $S_m^u$：本轮被 mask 的位置集合；
- $S^{u\prime}$：被污染后的输入；
- $v_m^\ast$：位置 $m$ 原本的真实 Item ID；
- $P(v_m^\ast\mid S^{u\prime})$：双向上下文下恢复正确商品的概率。

对一个 batch，通常再对用户和 mask 位置求平均。

这条公式与 SASRec 的差异不在 softmax，而在条件信息：

$$
\text{SASRec:}\quad
P(v_{t+1}\mid v_{\le t}),
$$

$$
\text{BERT4Rec train:}\quad
P(v_m\mid v_{<m},v_{>m}\text{ except masks}).
$$

推理时则回到：

$$
P(v_{n+1}\mid v_{\le n},[\mathrm{MASK}]).
$$

所以“bidirectional”描述的是训练表示，不是系统获得了未来行为。

## 训练和推理分别怎么做

```text
for each user sequence S:
    choose a random mask set M with proportion rho
    S_corrupt = replace positions M with [MASK]

    hidden = bidirectional_transformer(S_corrupt)
    logits = item_head(hidden[M])
    loss = cross_entropy(logits, original_items[M])

    sometimes construct a sample that masks only the final position
```

线上：

```text
history = keep_recent_items(user, max_length=N-1)
input = history + [[MASK]]
hidden = BERT4Rec(input)
scores = item_head(hidden[last_position])
return constrained_top_k(scores)
```

与 SASRec 相比：

- 训练可以在多个随机位置获得监督；
- 每个 mask 的表示利用双向上下文；
- 推理每次需要重新编码整个可见历史，不能像 GRU hidden state 那样简单增量更新；
- 最后仍需在 Item ID 词表或候选集上打分。

## 实验究竟证明了什么

论文在四个数据集上与 POP、BPR-MF、NCF、FPMC、GRU4Rec、Caser、SASRec 等比较，并报告相对最强基线平均提升：

- HR@10：**7.24%**；
- NDCG@10：**11.03%**；
- MRR：**11.46%**。

不过最有解释力的是 Table 3，而不是总表。作者设计三个版本：

1. **SASRec**：单向 causal attention；
2. **BERT4Rec (1 mask)**：每次只 mask 一个位置，主要隔离“双向表示”；
3. **BERT4Rec**：多个 mask，再加入更丰富的 Cloze 监督。

![双向表示与多 mask 分别带来的收益](/blog/generative-recommendation/04-bert4rec/evidence.svg)

*图 4：HR@10，数值来自原论文 Table 3。两级提升分别支持 bidirection 与多位置 Cloze。*

Beauty 上：

$$
0.2653
\rightarrow
0.2940
\rightarrow
0.3025.
$$

MovieLens-1M 上：

$$
0.6629
\rightarrow
0.6869
\rightarrow
0.6970.
$$

第一跳说明即使每轮只 mask 一项，双向上下文也优于单向 SASRec；第二跳说明多位置 Cloze 还能继续提供收益。

其他实验还表明：

- $\rho=0.2$ 比 0.1 在四个数据集都更好，但最佳值依数据而变；
- 隐维度变大不保证持续提升，RNN 更容易在大维度退化；
- 去掉 position-wise FFN、residual、position embedding 或 LayerNorm 均会影响性能；
- 长序列数据对 position 与多 head 更敏感。

## 实验数字为什么不能直接当排行榜

论文使用常见的 leave-one-out：

- 每个用户最后一项测试；
- 倒数第二项验证；
- 测试时把真实商品与 100 个按流行度采样的未交互商品排序。

这种设置适合做受控比较，却不是全目录检索。一个模型在 101 件商品里把真值放进前十，和在百万商品里召回它，不是同一难度。

而且把最后一次历史行为当唯一真值，默认“下一个点击就是正确答案”。如果用户同时可能喜欢多件商品，未点击候选会被错误视为负例。

## 它失败在哪里

### 1. 训练—推理有结构差异

训练时，内部 mask 同时有左右上下文；推理时末尾 mask 只有左侧历史。论文加入末尾 mask 样本缓解差异，但没有消除任务不一致。

### 2. 双向不等于知道未来

线上系统不可能读取尚未发生的点击。BERT4Rec 的优势来自用完整训练序列学习更好的 Item/context 表示，而不是预知用户下一步。

### 3. 计算仍为二次，且难以增量

完整 self-attention 是 $O(n^2d)$。用户每增加一个行为，朴素实现重新编码整个窗口；这与可以缓存 hidden state 的 RNN 或 KV cache 式 decoder 不同。

### 4. 商品词表仍随目录增长

每件商品是一个 atomic token。百万商品意味着巨大输入 embedding 与输出矩阵；上新商品没有交互表示，也没有共享子词可以继承。

### 5. 文本与世界知识仍在门外

模型能从共同出现学到 `I03` 与 `I04` 相关，却不知道它们分别叫“攀岩鞋”和“粉袋”，也不会因为“防滑橡胶”这段描述理解新商品。

### 6. 它恢复的是行为，不是理由

模型擅长补 Item ID，不会回答“为什么推荐”，也不能统一评分、解释、对话或搜索等任务。每种任务仍要单独设计头与数据。

## 下一篇为什么会出现

第一幕到这里完成了四次迁移：

1. **BPR**：从拟合数值到比较顺序；
2. **GRU4Rec**：从静态用户到行为序列；
3. **SASRec**：从递归记忆到 attention；
4. **BERT4Rec**：从下一项预测到 masked token learning。

现在推荐系统已经拥有一套很像 NLP 的形式：

```text
用户历史 = token sequence
Item ID = vocabulary token
训练目标 = next token 或 masked token
Transformer = sequence encoder
```

但它仍然只会说 Item ID。

如果用户问“给我推荐一件适合室内抱石的装备，并解释理由”，我们需要评分、序列预测、检索与解释多个模型协作。一个更大胆的问题随之出现：

> 能不能把推荐、评分、解释和问答全部改写成文字，让一个模型用同一种 text-to-text 目标学习？

下一篇：[P5](https://arxiv.org/abs/2203.13366)——所有推荐任务都能改写成一句话吗？

## 原始论文

- Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, Peng Jiang. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690). CIKM 2019.

[返回完整路线图 →](/series/generative-recommendation)
