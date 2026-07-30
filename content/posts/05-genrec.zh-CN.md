---
title: "下一件商品不够，模型要生成整页｜生成式推荐 2026·05：GenRec"
date: "2026-07-30"
updated: "2026-07-30"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - GenRec
  - Reinforcement Learning
pairKey: "generative-recommendation-2026-05-genrec"
slug: "generative-recommendation-2026-05-genrec-zh"
excerpt: "GenRec 以页面为训练和生成单位，用 Token Merger 压缩长历史，并以 GRPO-SR 对齐页面级偏好。"
series: "generative-recommendation-2026"
seriesOrder: 5
draft: false
---


前四篇主要讨论商品怎样表示、怎样推理、怎样生成。[GenRec](https://arxiv.org/abs/2604.14878) 把镜头拉回真实 App：

> 用户下拉一次，系统需要的是一整页，而不是一个“下一商品”。

同一份历史可以对应很多合理页面。若训练数据把每个点击商品都当作唯一答案，模型会把正常的多解问题误学成互相冲突的标签。GenRec 因此把页面本身变成监督单位。

## 30 秒看懂本文

1. **旧方法的问题**：next-item 训练忽略列表内部的顺序、互补、重复与多样性；长历史展开成多 token SID 又会吞噬上下文与算力。
2. **GenRec 的答案**：Page-wise Generation 一次预测整页商品；Token Merger 压缩历史 token；GRPO-SR 用页面级奖励对齐相关性、有效性和偏好。
3. **最重要的证据**：论文报告大规模线上部署与一个月 A/B 结果，说明生成式推荐已不只是公开数据集上的 next-item 实验。但私有数据、奖励模型和完整基线仍限制外部复核。

![GenRec：从 next item 到整页生成](/blog/generative-recommendation-2026/05-genrec/mechanism.svg)

## 为什么 point-wise 监督会互相打架

假设用户打开运动页面，真实曝光与点击形成：

```text
攀岩鞋 → 防滑粉 → 速干毛巾 → 泳帽
```

如果我们分别构造四个训练样本，相同历史会被要求“唯一正确地”预测四个目标。页面级监督则保留一个事实：这些商品共同组成一次展示，顺序与共现也携带策略信息。

页面生成可以写成：

\[
p(S_u\mid H_u)
=
\prod_{j=1}^{|S_u|}
p(i_j\mid H_u,i_{<j}),
\]

其中 \(S_u\) 是完整页面。后一个商品显式依赖前面已经生成的商品，模型因此有机会学习去重、互补和整页节奏。

## Token Merger 在压缩什么

若每个商品有多级 SID，数百次历史行为会膨胀成上千 token。GenRec 使用 Token Merger 将一件商品或一段局部历史的多个 token 汇聚为更紧凑的表示。

重要的是，压缩主要减少**编码历史**的长度，而页面输出仍需逐 token 解码。它缓解上下文与 attention 成本，却没有消除生成整页的解码负担。

这提醒我们读“token 减半”时要问：减的是 prompt、KV cache，还是最终输出步数？

## 为什么还需要强化学习

SFT 学习的是日志中“曾经出现过的页面”，不等于知道一个新页面是否好。GenRec 用 GRPO-SR 对多个候选页面做组内比较，奖励可以抽象为：

\[
R(S)=R_{\text{valid}}
+
\mathbb 1[R_{\text{valid}}>0]
\left(
\alpha R_{\text{relevance}}
+\beta R_{\text{preference}}
+\gamma R_{\text{diversity}}
\right).
\]

Validity gate 很关键。如果页面含非法商品、重复过多或格式错误，模型不能靠其他高分项把坏页面“平均成好页面”。这是一种防 reward hacking 的硬门槛。

不过奖励定义也把平台价值观写进模型。相关性、长尾曝光、成交和用户满意度的权重不同，会生成完全不同的页面。

## 线上结果应该怎样读

论文报告点击、成交与长尾商品等线上增益，并称模型完成部署。工业证据的价值很高，但不能把所有提升都归因于某一个模块。

一个完整系统通常同时改变：

- 训练样本与页面定义；
- tokenizer 与多模态特征；
- 模型规模；
- RL 奖励；
- beam、缓存和过滤；
- 流量策略与候选库存。

因此最可信的阅读方式，是区分整体方案 A/B 与模块消融。前者证明“整套系统能工作”，后者才帮助判断哪个想法可迁移。

## 它失败在哪里

页面生成仍面临三个硬问题：

1. **多样答案**：日志只记录一个曝光页面，无法说明其他页面是坏答案；
2. **反馈偏差**：点击来自旧策略曝光，模型可能继承并放大旧策略；
3. **解码约束**：库存、去重、合规和实时状态必须在生成中或生成后强制处理。

GenRec 证明页面级生成已经进入生产，但它仍主要服务内容推荐。广告系统还多出预算、出价、预估收入和竞争机制。

下一篇 [GR4AD](/blog/generative-recommendation-2026-06-gr4ad-zh) 会把生成模型送进广告主链路，那里每个 token 都可能直接改变收入。

[上一篇：DualFashion](/blog/generative-recommendation-2026-04-dualfashion-zh) · [返回第二季专题页](/series/generative-recommendation-2026)
