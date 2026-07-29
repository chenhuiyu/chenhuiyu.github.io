---
title: "推荐模型也存在 Scaling Law 吗？｜生成式推荐 16：HSTU"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - HSTU
  - Scaling Law
pairKey: "generative-recommendation-16-hstu"
slug: "generative-recommendation-16-hstu-zh"
excerpt: "HSTU 把工业推荐重写为用户行为的 sequential transduction，并从注意力、采样、内核和服务四层解决长序列规模化。"
series: "generative-recommendation"
seriesOrder: 16
draft: false
---

到 [ETEGRec](/blog/generative-recommendation-15-etegrec-zh) 为止，我们一直在讨论一个商品怎样变成 token。但真实工业推荐还有另一座更大的山：

```text
数十亿动态 ID
数百到数千类异构特征
每位用户最长 10⁵ 级行为
每天数百亿次新 action
严格的高 QPS、低延迟
```

小数据上把模型换成 Transformer，不会自动解决这些问题。[Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) 的切入点也不是“让 LLM 读商品标题”，而是：

> 把用户 action 本身当作一种生成建模模态，把 ranking 与 retrieval 重写成 sequential transduction，再为这种数据设计 HSTU。

这是系列从学术原型转向工业骨干的分水岭。

## 30 秒看懂本文

1. **Semantic ID 路线留下的问题**：万级目录、短历史和 beam search 结果，不能说明生成式推荐能在十亿用户、动态词表和长序列下工作。
2. **HSTU 的答案**：把异构特征并入一条时间线，用一次用户级生成训练监督多个位置；以 pointwise aggregated attention、时间偏置、门控、Stochastic Length、ragged kernel 与 M-FALCON 共同缩放。
3. **最重要的证据**：HSTU-large 在 Amazon Books 的 NDCG@10 比 SASRec 高 65.8%；8192 长度时比 FlashAttention2 Transformer 快 5.3–15.2 倍；工业 GR 扩到 1.5T 参数并在线提升 ranking 主指标 12.4%。

![DLRM 到 sequential transduction](/blog/generative-recommendation/16-hstu/paradigm-shift.svg)

*图 1：HSTU 的“生成式”首先是数据与训练范式，不等于生成自然语言，也不等于必须使用 Semantic ID。*

## 上一篇留下了什么问题

传统 DLRM 为每个 `(user, candidate)` 构造一个样本：

```text
用户特征 + 行为统计 + 候选特征 + 交叉特征 → 点击/观看概率
```

这套方法可解释、可控，也积累了大量工程经验。但有两个扩展瓶颈：

1. 同一用户的历史会为每个候选重复编码，候选越多，计算近似线性增长；
2. 继续堆交叉网络和 MLP，质量常比计算更早饱和。

学术 sequential recommender 则通常只保留 item ID 序列，丢掉真实系统里重要的 creator、language、location、counter、ratio 等特征。

HSTU 论文的第一步不是选网络，而是重新安排数据。

## 怎样把异构推荐特征变成时间线

类别特征中，用户点击的商品/视频是主时间线。其他变化缓慢的序列，例如关注的创作者、所在城市、加入的社区，只保留连续相同区间的第一个变化点，再合并到主时间线。

连续数值特征更难。历史 CTR、衰减计数等几乎每次互动都变化，完全序列化会爆炸。论文做了一个大胆选择：

> 若原始类别事件和目标候选都已进入序列，一个足够强、target-aware 的序列模型应能重新学习这些聚合，因此可删除大量手工 numerical features。

后面的 MTGR 会直接挑战这条假设。现在先理解它为什么让 scaling 成为可能：模型不再为每个统计量存一份人工摘要，而是从更长原始行为中学习摘要。

## 用八件商品理解 ranking 与 retrieval

把商品 \(\Phi_i\) 和用户动作 \(a_i\) 交错：

```text
I01 网球拍, 点击,
I03 攀岩鞋, 完播,
I05 羽毛球拍, 跳过,
I07 泳帽, 收藏
```

### Ranking：给候选，预测动作

若下一个 token 是候选 `I06 泳镜`，模型预测：

\[
P(a_{i+1}\mid \Phi_0,a_0,\ldots,\Phi_{i+1}).
\]

候选已经进入 attention，所以它能在很早的层与所有历史做 target-aware 交互。输出可以是点击、完播、时长等多任务 logits。

### Retrieval：给历史，预测下一正向内容

在正向互动位置监督下一个内容：

\[
P(\Phi_{i+1}\mid u_i).
\]

若下一次曝光被跳过，目标可记为 \(\varnothing\)，而不是把任意曝光都当成用户真正想要的内容。

![八件商品中的 ranking 与 retrieval](/blog/generative-recommendation/16-hstu/toy-transduction.svg)

*图 2：同一 token 流可以在不同位置发出不同监督，统一的是用户表示骨干，不是强行让所有任务输出同一种东西。*

## 为什么“按用户生成训练”能省一个数量级

点式流训练在每个时间点重新构造历史，长度为 \(n_i\) 的用户会被反复编码。论文写出的总成本包含：

\[
\sum_i n_i(n_i^2d+n_id^2).
\]

取最长序列 \(N\)，可近似成：

\[
O(N^3d+N^2d^2).
\]

生成式训练改为在一次完整序列前向中监督多个位置，encoder 成本被多个 target 摊销。按用户长度以 \(s_u(n_i)\propto1/n_i\) 采样后，成本降为：

\[
O(N^2d+Nd^2).
\]

这不是把 attention 从二次变成线性，而是消除了“同一前缀被重复算 \(N\) 次”的冗余。工业实现可以在一次 request 或 session 结束时发出用户级训练样本。

## HSTU Layer 到底改了什么

一个 HSTU layer 有三步：

\[
U,V,Q,K=\operatorname{Split}\big(\phi_1(f_1(X))\big),
\]

\[
A(X)V=\phi_2\big(QK^\top+\operatorname{rab}^{p,t}\big)V,
\]

\[
Y=f_2\Big(\operatorname{Norm}(AV)\odot U\Big).
\]

其中 \(\phi_1,\phi_2\) 都使用 SiLU，\(\operatorname{rab}^{p,t}\) 同时编码相对位置与真实时间间隔。

![HSTU Layer](/blog/generative-recommendation/16-hstu/mechanism.svg)

*图 3：注意力聚合与逐元素 gate 在一个轻量 block 中合流；HSTU 只保留两个外部线性层。*

### 不对序列做 softmax

Transformer attention 用 softmax 把每行权重归一为 1。如果目标相关历史从 2 条增到 20 条，总质量仍为 1，“相关行为有多少”这条强度信号容易被抹平。

HSTU 对每个 \(QK^\top\) 分数做 pointwise SiLU，不沿序列归一化；之后再 LayerNorm 稳定数值。这样注意力能同时表达相对相关性与相关事件数量。合成 streaming 数据中，Transformer HR@10 为 0.0442，去掉时间偏置的 HSTU 为 0.0893；换回 softmax 后是 0.0617。

### 用门控代替独立 FFN

\(\operatorname{Norm}(AV)\odot U\) 让聚合结果直接与 pointwise gate 交互，类似 SwiGLU/MoE 中的条件计算。标准 Transformer 一层需要更多中间 activation；论文估算 bfloat16 下 HSTU 为 \(14d\)，Transformer 为 \(33d\)，因此同样显存可堆更深。

## 训练与服务的另外三块拼图

### 1. Ragged/Jagged kernel

用户历史长度高度不均。如果全部 pad 到最长序列，大量计算浪费在空 token。HSTU 使用完全 ragged 的 grouped GEMM 和融合 kernel，按真实长度计算。

### 2. Stochastic Length

长历史中存在多时间尺度重复。训练时大多数情况从超长用户中抽子序列，少数情况保留全量。论文的 \(\alpha=1.6\) 设置会把长度 4096 大多缩到 776；64%–84% 的 sparsity 下，主任务 NE 退化不超过 0.002。它是有概率保留长上下文，不是永远截断最近 N 条。

### 3. M-FALCON

ranking 仍要为一个用户处理大量候选。M-FALCON 对候选 micro-batch，缓存与候选无关的历史计算，把 target-aware 代价摊销。论文称在相同预算下服务 285 倍更复杂的 GR，评分 1024/16384 个候选时 QPS 反而是 DLRM 的 1.50/2.99 倍。

## 实验究竟证明了什么

公共数据 full-ranking 结果：

![HSTU 的 NDCG@10](/blog/generative-recommendation/16-hstu/evidence.svg)

*图 4：相对增益在稀疏的 Amazon Books 上最大，扩大 HSTU 后三个数据集都继续提高。*

| 数据集 | SASRec | HSTU | HSTU-large |
| --- | ---: | ---: | ---: |
| MovieLens-1M | 0.1603 | 0.1720 | **0.1893** |
| MovieLens-20M | 0.1621 | 0.1878 | **0.2106** |
| Amazon Books | 0.0156 | 0.0219 | **0.0257** |

工业 one-pass streaming 中，完整 HSTU 的 retrieval log perplexity 为 3.978，Transformer++ 为 4.015；ranking 主任务 NE 为 0.4937，Transformer++ 为 0.4945，越低越好。

论文更引人注意的是 scaling 观察：

- 训练 compute 跨三个数量级时，retrieval HR@100、HR@500 与 ranking NE 都可由对数/幂律形式拟合；
- GR 扩到 1.5T 参数，DLRM 约在 200B 参数附近饱和；
- 最大配置为序列长 8192、embedding 1024、24 个 HSTU layers；
- 工业 ranking 线上主任务相对 DLRM +12.4%，次任务 +4.4%。

这里的 1.5T 包含高基数 embedding 参数，不能与 1.5T dense LLM 的每 token FLOPs直接等同。论文比较的是推荐系统自己的容量—计算—质量关系。

## 它失败在哪里

### 1. 大部分核心证据不可复现

1.5T 模型、私有数据、在线指标与专用 serving kernel 都无法在公共基准完整复现。开源实现能验证 block，却不能复现整个工业结论。

### 2. Scaling law 是经验拟合，不是自然定律

曲线覆盖三个数量级、一个平台的特定任务与资源配置。数据质量、负反馈定义、序列长度和 embedding 规模共同变化；不能推出“任何推荐器只要加算力都会改善”。

### 3. Pointwise attention 的抗噪性有代价

softmax 自动限制总质量，也天然抑制大量小噪声；不归一的 pointwise 聚合要依赖 LayerNorm、数据处理与 kernel 稳定训练。长历史并非所有事件都有用。

### 4. 删除 numerical features 是强假设

理论上序列可以重新学习 CTR、曝光次数和交叉统计；有限长度、有限模型和非平稳分布下，显式统计可能仍然更高效。MTGR 会用线上系统证明这一点。

### 5. “统一”仍不是一个最终列表目标

ranking 预测动作，retrieval 预测内容；它们共享 sequential transduction 表示，但业务上的召回、排序、策略与列表约束没有在一个输出 session 中完全合并。

## 下一篇为什么会出现

HSTU 证明工业推荐骨干可以随计算扩展，并把许多任务放进同一条行为时间线。但实际系统仍可能保留 retrieval、pre-ranking、ranking 的级联：

```text
统一 backbone ≠ 统一最终决策
```

[下一篇 OneRec](/blog/generative-recommendation-17-onerec-zh) 会再向前一步：用一个 encoder–decoder 直接生成 5–10 个视频的 session，以 Sparse MoE 扩容量，再用 reward model 与 DPO 把最终列表向观看时长和互动偏好对齐。

---

**本篇批判性结论**：HSTU 的核心不是一个 attention 公式，而是“重写问题后才有 scaling”。数据按用户聚合、任务按序列转导、计算按真实长度稀疏、候选成本按缓存摊销——四层同时成立，万亿参数才有意义。
