---
title: "生成越深，偏差可能越大｜生成式推荐 2026·02：CARE"
date: "2026-07-30"
updated: "2026-07-30"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - CARE
  - Recommendation Reasoning
pairKey: "generative-recommendation-2026-02-care"
slug: "generative-recommendation-2026-02-care-zh"
excerpt: "CARE 把自回归 Semantic ID 解码解释成级联排序，用渐进历史和查询锚定的并行推理控制前缀偏差。"
series: "generative-recommendation-2026"
seriesOrder: 2
draft: false
---


[DIGER](/blog/generative-recommendation-2026-01-diger-zh) 让商品 ID 能被推荐损失修改，但它没有改变自回归解码的基本事实：

> 第一个 token 一旦选错，后面的 token 只能在错误前缀划出的狭窄世界里继续认真工作。

[CARE](https://arxiv.org/abs/2602.03692) 把这件事解释成**级联排序**。每生成一个 SID token，模型都在缩小候选集合。越早的决策拥有越大的“封路权”，热门前缀和偶然错误也因此会逐层放大。

## 30 秒看懂本文

1. **旧方法的问题**：逐 token 生成不是若干独立分类，而是一串不可逆的候选裁剪。早期前缀偏向热门类别，后续就很难抵达长尾商品。
2. **CARE 的答案**：逐步开放用户历史，让不同粒度的证据分别参与；同时使用由当前 query 锚定的并行推理分支，避免所有计算只沿单一路径滚落。
3. **它所谓的 reasoning**：不是输出一段自然语言思维链，而是改变信息可见性与计算图。评价它应看准确率、多样性、偏差和开销，而不是文字是否像“思考”。

![CARE：把自回归解码看成级联排序](/blog/generative-recommendation-2026/02-care/mechanism.svg)

## 用商品树理解前缀封路

假设第一级 token 表示大类：

```text
运动
├── 球拍
├── 鞋
├── 穿戴
└── 配件
```

第二级再区分网球、羽毛球、攀岩或游泳，第三级才落到具体商品。

用户最近购买泳帽，但长期历史里球拍点击很多。如果第一个 token 被热门历史推向“球拍”，后面即使识别到“游泳”，也无法生成速干毛巾，因为它已经不在当前前缀的可达子树中。

传统排序中，第一阶段召回漏掉的商品不会被精排救回。自回归 SID 也是同一逻辑，只是每一层 token 都充当了一次微型召回。

## CARE 的两块核心结构

### 1. 渐进式历史编码

CARE 不把完整历史一次性塞给每个解码位置，而是分阶段开放信息。短期历史先回答当前意图，中期与长期历史随后补充稳定偏好和重复模式。

可以把第 \(s\) 阶段的状态写成：

\[
h_s=f_\theta(H_{u}^{(s)},q),
\]

其中 \(H_u^{(s)}\) 是当前阶段可见的历史子集，\(q\) 是当前推荐请求。模型不必在第一步同时消化所有证据，也减少长期热门兴趣压倒近期转向的机会。

### 2. Query-anchored parallel reasoning

普通自回归模型的第 \(j\) 个状态高度依赖前缀 \(c_{<j}\)：

\[
p(c_j\mid H_u,c_{<j}).
\]

CARE 增加由 query 直接锚定的并行分支。每条分支都能重新查看请求与历史，而不是只继承上一 token 的单一结论。最终融合多个分支状态，再决定 SID。

这不是把 beam size 调大。Beam search 仍在同一前缀概率树里竞争；并行 reasoning 的目的，是让中间表示拥有不完全相同的证据路径。

## 为什么准确率与多样性会冲突

热门 code 拥有更多训练样本，早期 token 更容易选择它们。这样通常能提升短期命中率，却会压缩可达商品空间，使长尾与多样性变差。

CARE 因此不只报告 Recall/NDCG，也检查 DivR、ORR 等多样性或偏差指标。这里最重要的阅读方式不是找一个“总冠军分数”，而是看：

- 准确率提升是否伴随更严重的热门集中；
- 多样性改善是否只是牺牲命中率换来的；
- 不同数据集的商品树与流行度分布是否一致；
- 计算增加是否足以支持线上解码。

论文报告其模块带来的推理时间增量相对有限，说明结构化并行计算可能比生成长篇自然语言 CoT 更接近生产约束。

## 它没有证明什么

CARE 讨论流行度偏差，不等于已经解决推荐公平性。供应商曝光、敏感属性、价格歧视和平台激励都比热门度更复杂。

同样，隐藏状态里的“reasoning branch”也不是可解释证据。除非我们做反事实干预，例如移除某段历史后结果按预期改变，否则不能声称这些分支忠实描述了模型因果过程。

更大的问题是：CARE 仍然接受离散 SID 与树状前缀。它努力让模型在树里走得更稳，却没有回答这棵树是否必须存在。

下一篇 [ContRec](/blog/generative-recommendation-2026-03-contrec-zh) 会直接拆掉这棵树：不再逐 token 生成离散地址，而是在连续空间里生成用户偏好向量。

[上一篇：DIGER](/blog/generative-recommendation-2026-01-diger-zh) · [返回第二季专题页](/series/generative-recommendation-2026)
