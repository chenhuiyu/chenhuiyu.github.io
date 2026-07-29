---
title: "用户不是一个点，而是一段正在发生的故事｜生成式推荐 02：GRU4Rec"
date: "2026-07-26"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - GRU4Rec
pairKey: "generative-recommendation-02-gru4rec"
slug: "generative-recommendation-02-gru4rec-zh"
excerpt: "BPR 把用户压成一个静态向量；GRU4Rec 第一次把一次访问看成会随点击更新的状态，并在每一步预测下一件商品。"
series: "generative-recommendation"
seriesOrder: 2
draft: false
---

小林平时爱打网球，但今晚她连续看了攀岩鞋、粉袋和护指胶带。

如果系统只记得她长期的网球偏好，下一件可能仍然推荐球拍；如果系统能看懂**这一段刚刚发生的行为**，更合理的答案可能是抱石刷或攀岩裤。

[上一篇 BPR](/blog/generative-recommendation-01-bpr-zh) 学会了比较“正商品应该排在负商品前面”，却把每个用户压进一个固定向量 $\mathbf w_u$。`网球 → 攀岩` 与 `攀岩 → 网球` 对它几乎是同一组交互。

[Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)，也就是 GRU4Rec，改变了用户的定义：

> 在 session-based recommendation 中，我们可能根本不知道用户是谁。此刻的“用户表示”，就是这段 session 到目前为止形成的隐藏状态。

推荐系统由此第一次从“静态匹配”认真转向“下一步预测”。

## 30 秒看懂本文

1. **BPR 留下的问题**：静态用户向量忽略行为顺序；匿名访问或新 session 甚至没有可查的用户向量。
2. **GRU4Rec 的答案**：用 GRU 按顺序读取点击，隐藏状态 $\mathbf h_t$ 在每一步更新，并为下一件商品输出分数。
3. **最重要的证据**：在 RSC15 与 VIDEO 两个真实 session 数据集上，单层 GRU 相比强 item-KNN 基线显著提升 Recall@20 与 MRR@20；论文还把 session-parallel mini-batch、batch 内负样本和 ranking loss 做成一套可训练方案。

![从静态用户向量到动态 session 状态](/blog/generative-recommendation/02-gru4rec/paradigm-shift.svg)

*图 1：BPR-MF 用一个长期向量表示用户；GRU4Rec 用随点击变化的隐藏状态表示“她现在在做什么”。*

## 上一篇留下了什么问题

BPR-MF 的打分是

$$
s(u,i)=\mathbf w_u^\top\mathbf h_i.
$$

其中 $\mathbf w_u$ 需要从用户过去的交互中学出来。这在 Netflix 式的长期账号场景里合理，但很多真实请求没有稳定身份：

- 电商访客没有登录；
- 新闻读者换了设备或清理 cookie；
- 用户第一次来到一个小网站；
- 同一用户的两次访问意图完全不同；
- 我们只想预测这次 session 的下一步，而不是长期画像。

工业系统常用 item-to-item 共现处理这种场景：用户刚看攀岩鞋，就推荐经常和攀岩鞋一起出现的商品。它简单、快速、常常很强，却通常只看最后一次点击。

GRU4Rec 的问题意识很准确：

> 能不能保留 item-KNN 对“当前行为”的敏感，同时让模型记住不止最后一件商品？

## 用四步 session 理解监督信号

设小林这次访问依次产生：

```text
I01 网球拍 → I03 攀岩鞋 → I05 羽毛球 → I06 泳镜
```

一条长度为 4 的序列可以右移一位，构造 3 个训练目标：

```text
输入 I01，目标 I03
输入 I03，目标 I05
输入 I05，目标 I06
```

但每个目标并不是只看当前输入。预测 `I06` 时，GRU 的 $\mathbf h_3$ 已经顺序处理过 `I01, I03, I05`。

![一条 session 怎样展开成多个 next-item 目标](/blog/generative-recommendation/02-gru4rec/toy-sequence.svg)

*图 2：标签只是“下一件”，隐藏状态却汇总了从 session 开始到当前位置的历史。*

这里出现了后续生成式推荐需要的两个基本元素：

1. **Item ID 成为序列中的离散符号**；
2. **当前输出以过去的有序行为为条件**。

GRU4Rec 还不是今天所说的“自回归生成商品列表”：常见推理仍会给大量候选商品算分、取 Top-K。但推荐已经开始像语言建模一样，把历史右移后预测下一个 token。

## 模型从输入到输出发生了什么

### 1. 输入：原论文使用 1-of-N

假设商品数为 $N$，当前点击商品 $i_t$ 被表示成长度 $N$ 的 one-hot 向量：

$$
\mathbf x_t\in\{0,1\}^N.
$$

今天的实现通常会先查一个 $d$ 维 item embedding，但原论文报告额外 embedding layer 略差，因此保留了 1-of-N 输入。阅读旧论文时要区分“论文做了什么”和“现代复现通常怎么写”。

### 2. GRU：用门控制记忆

普通 RNN 每一步把当前输入和旧状态混在一起，长序列容易出现梯度消失。GRU 增加两个门：

- **update gate $\mathbf z_t$**：本次写入多少新信息；
- **reset gate $\mathbf r_t$**：构造候选状态时，旧记忆保留多少。

状态更新为

$$
\mathbf h_t
=
(1-\mathbf z_t)\odot\mathbf h_{t-1}
+
\mathbf z_t\odot\tilde{\mathbf h}_t.
$$

如果 $\mathbf z_t$ 接近 0，模型延续旧状态；接近 1，则更多采用新候选状态。

### 3. 输出：下一件商品的偏好分数

隐藏状态经过输出层，为候选商品得到分数 $\hat r_{t,i}$。原始完整输出可以覆盖全部 $N$ 件商品，但大目录下计算与更新成本太高，因此论文只对正确下一项和采样项计算必要分数。

![GRU4Rec 的 session-parallel 训练数据流](/blog/generative-recommendation/02-gru4rec/mechanism.svg)

*图 3：多条 session 同时前进；GRU 更新状态；当前 batch 中其他 session 的目标商品顺便成为负样本。*

若 batch size 为 $B$、隐藏维度为 $d$，可以把核心张量理解为：

| 张量 | Shape | 含义 |
|---|---:|---|
| 当前商品 | 原论文 $[B,N]$ | $B$ 条活跃 session 的 one-hot 输入 |
| hidden state | $[B,d]$ | 每条 session 独立的动态状态 |
| 正确下一项 | $[B]$ | 每条 session 的 shifted target |
| batch 内候选分数 | $[B,B]$ | 每行一个正项，其余列可作负项 |

## 核心公式逐项拆解

GRU 的完整公式包括 update gate、reset gate 与候选状态：

$$
\mathbf z_t
=
\sigma(W_z\mathbf x_t+U_z\mathbf h_{t-1}),
$$

$$
\mathbf r_t
=
\sigma(W_r\mathbf x_t+U_r\mathbf h_{t-1}),
$$

$$
\tilde{\mathbf h}_t
=
\tanh\left(
W\mathbf x_t
+
U(\mathbf r_t\odot\mathbf h_{t-1})
\right),
$$

$$
\mathbf h_t
=
(1-\mathbf z_t)\odot\mathbf h_{t-1}
+
\mathbf z_t\odot\tilde{\mathbf h}_t.
$$

把它映射到小林的行为：

- 看到 `I03 攀岩鞋` 时，reset gate 可以降低与当前意图无关的网球记忆；
- update gate 决定“攀岩”应该把 session 状态改写多少；
- 后续 `I05 羽毛球` 若只是随手点开，模型也可能保留一部分攀岩状态；
- 预测 `I06 泳镜` 时，输出依赖的是累积后的 $\mathbf h_3$，而不是单独的 `I05`。

门不是人类可读的“网球开关”或“攀岩开关”。它们是逐维连续值，解释时不能把一个神经元强行命名成兴趣标签。

## 三个让它真正可训练的工程设计

GRU4Rec 的影响不只来自“把 GRU 放进推荐”。论文真正解决了稀疏、大词表、变长 session 的训练问题。

### 1. Session-parallel mini-batch

传统语言模型可以把固定长度片段堆成 batch，但 session 长度可能从 2 到几百不等。论文的做法是：

1. 取前 $B$ 条 session 的第一个事件；
2. 每条 session 在下一 step 前进一步；
3. 某条 session 结束，就在对应 batch 槽位换入下一条；
4. 换入时把该槽位的 hidden state 清零。

这样既保留完整 session，又能使用高效矩阵运算。

### 2. Batch 内其他目标作为负样本

假设第 $b$ 条 session 的下一项是 $i_b^+$。对这一行来说，其他 session 的 $i_{b'}^+$ 可以当负例。一次输出矩阵运算得到 $B\times B$ 分数，无需额外采样器。

由于热门商品更常出现在 batch 目标中，这种方法也隐式接近 popularity sampling。优点是更容易抽到用户可能知道的商品；缺点是 batch 中可能存在假负例。

### 3. Ranking loss

论文尝试了 BPR 与新设计的 TOP1。BPR 形式是：

$$
\mathcal L_{\text{BPR}}
=
-
\frac1{N_S}
\sum_j
\log\sigma(\hat r_{t,i^+}-\hat r_{t,j}).
$$

TOP1 则近似“负项排在正项前面的相对 rank”，同时用额外项把负样本分数压回 0：

$$
\mathcal L_{\text{TOP1}}
=
\frac1{N_S}\sum_j
\left[
\sigma(\hat r_{t,j}-\hat r_{t,i^+})
+
\sigma(\hat r_{t,j}^2)
\right].
$$

第二项很重要：如果 batch 中一个商品有时是正项、有时充当其他行的负项，所有分数可能一起不断抬高。把负项约束在 0 附近能缓解这种漂移。

## 训练和推理分别怎么做

```text
sort sessions and open B active slots
initialize hidden[B, d] = 0

while sessions remain:
    x = current item from every active session
    y = next item from every active session

    hidden = GRU(x, hidden)
    scores = score(hidden, candidate_items=y)
    loss = TOP1_or_BPR(diagonal positives, off-diagonal negatives)
    update parameters

    advance every session
    replace finished sessions and reset their hidden rows
```

训练时，目标来自已经发生的下一次点击。

线上推理时，流程是：

1. session 开始时将 $\mathbf h_0$ 置零；
2. 每收到一个新点击，就更新一次 $\mathbf h_t$；
3. 对候选商品计算下一项分数；
4. 过滤不可售、重复、违规商品后取 Top-K；
5. 下一个真实点击到来，再继续更新状态。

所以 GRU4Rec 的 hidden state 天然适合增量 serving：不用每次重算整个历史。但如果服务层丢失状态、session 划分错误或用户跨设备，记忆也会断开。

## 实验究竟证明了什么

论文使用两个数据集：

- **RSC15**：约 796 万条训练 session、3163 万次点击、37483 件商品；
- **VIDEO**：约 300 万条 session、1300 万次观看、33 万个视频。

评估逐事件回放 session，使用 Recall@20 和 MRR@20。下图重绘论文 Table 3 中 `TOP1 + 1000 hidden units` 与最强 item-KNN 基线的对比。

![GRU4Rec 与 item-KNN 的原论文结果](/blog/generative-recommendation/02-gru4rec/evidence.svg)

*图 4：GRU4Rec 在两个数据集的 Recall@20 与 MRR@20 上总体领先 item-KNN。数值来自原论文 Table 1 与 Table 3。*

以 RSC15 为例：

- item-KNN Recall@20 为 **0.5065**；
- GRU4Rec（TOP1, 1000 units）为 **0.6206**，相对提升 22.53%；
- MRR@20 从 **0.2048** 提升到 **0.2693**。

但实验也提供了不那么“整齐”的信息：

- 在 VIDEO 上，BPR loss 的 MRR 可能低于 item-KNN；
- cross-entropy 在大 hidden size 上数值不稳定；
- 单层 GRU 优于更深网络，作者推测是 session 普遍较短，但没有证明原因；
- 原论文中额外 item embedding 略差。

真正被支持的结论是：

> 在匿名、短 session 的下一项预测里，顺序状态比只看最后一次点击的强共现基线更有表达力；而训练采样与 loss 对结果至关重要。

## 它失败在哪里

### 1. 所有历史被压进一个向量

$\mathbf h_t$ 必须同时保存短期意图、重复模式与长期线索。序列越长，越早的信息经过更多递归步骤，越容易被稀释。门能缓解，不会消灭瓶颈。

### 2. 时间维无法完全并行

同一 session 的 $\mathbf h_t$ 必须等待 $\mathbf h_{t-1}$。不同 session 可以并行，但一条长序列内部仍要逐步计算，GPU 利用和长依赖学习都会受限。

### 3. In-batch negative 可能是假负例

另一条 session 的正确商品，可能同样适合当前用户。batch 越小，候选覆盖有限；热门商品出现越频繁，也越容易反复被当成负项。

### 4. 仍然需要候选打分

模型预测的是商品分数，不是由多个共享 token 组成的可约束路径。面对百万目录，完整 softmax 或全量打分仍然昂贵。

### 5. Session 边界是一项产品假设

多久不操作算新 session？购物车跨天是否连续？登录前后是否合并？边界错误会清空本该保留的状态，或把两个完全不同的意图硬连起来。

### 6. 论文评估属于 2015 年范式

Recall@20 与 MRR@20 衡量下一个日志事件是否被找回，不等于长期满意度；VIDEO 还只在 3 万热门候选中排序。不能把这些离线结果直接外推到今天的超大工业目录。

## 下一篇为什么会出现

GRU4Rec 让用户从一个静态点变成了一段故事。

但它讲故事的方式像传话：第一步把信息交给第二步，第二步再交给第三步。假设小林过去 100 步里有一条“买了攀岩鞋”，当前正在看粉袋，模型要让这条信息经过几十次状态更新才能影响现在。

我们真正想要的也许不是把所有过去平均压进一个向量，而是：

> 当前要预测什么，就直接回头寻找与它最相关的那几步。

下一篇：[SASRec](https://arxiv.org/abs/1808.09781)——历史很长，真正相关的可能只有几步。

## 原始论文

- Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk. [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939). ICLR 2016.

[返回完整路线图 →](/series/generative-recommendation)
