---
title: "HSTU 规模化：Ragged、候选摊销与多模态内容表示"
date: "2026-09-10"
updated: "2026-09-10"
category: "HSTU"
language: "zh-CN"
tags: ["hstu", "Hands-on", "Model Learning"]
pairKey: "hstu-scale"
slug: "hstu-scale-zh"
excerpt: "把长短序列浪费、候选相关计算和内容 embedding 分开分析，连接推荐与多模态理解。"
series: "hstu"
seriesOrder: 4
draft: false
---

![HSTU 规模化：Ragged、候选摊销与多模态内容表示](/learning/hstu-scale-zh.svg)

## 从零开始的完整讲解

### 大规模首先暴露的是长度不均与重复计算

用户历史可能从几条到几千条。如果一个 batch 为最长用户补齐，短历史会携带大量 padding。另一方面，同一用户的一批候选常共享几乎全部历史，逐候选重复编码历史会浪费计算。Ragged execution 和候选摊销分别针对这些不同冗余。

不要把“把 padding 去掉”与“attention 从平方变线性”混为一谈。Ragged 能减少无效位置，但每条有效序列内部的连接仍取决于模型定义和内核。

### 手算 padding 的浪费与 offset

三条历史长度为 2、5、1，稠密存储有 `3*5=15` 个位置，有效位置只有 8 个。Ragged 可以把它们连续放在 values 数组中，再用 offsets `[0,2,7,8]` 标记每条起止。第三条序列对应 `values[7:8]`。

```python
sequences = [[1,2],[3,4,5,6,7],[8]]
values, offsets = [], [0]
for sequence in sequences:
    values.extend(sequence)
    offsets.append(len(values))
print(values, offsets)
assert all(values[offsets[i]:offsets[i+1]] == s for i,s in enumerate(sequences))
```

这只是存储验证。Attention 内核还必须保证不同用户之间不互相读取，不能把拼接后的 values 当成一条长历史。Empty sequence、超长序列和 offset 类型也需要检查。

### 候选摊销取决于模型是否允许共享

如果历史编码 h 与候选无关，可以一次算 h，再批量与多个候选表示打分，避免重复历史前向。若模型从早期就让候选与历史相互作用，共享范围就受限。HSTU 相关系统中的具体候选组织与 attention mask 要按模型实现分析，不能声称所有排序器都能直接缓存同一份用户向量。

估算时把历史编码、候选相关计算、最终打分和通信分开。例如历史部分占总成本 80%，即使完全复用，剩余 20% 仍会随候选数增长。真实收益还受内存读取和 batch 布局影响。

### 多模态内容特征怎样连接冷启动

新物品没有足够交互，但通常已有标题、图片或视频。内容编码器可以提供语义表示，与 ID embedding 通过投影、拼接或门控融合。这样为新物品提供了一条信息通路，但不保证内容相近就有相同点击或购买率。

热门度、价格、可用性和用户意图都可能影响偏好。CLIP 向量擅长某些语义匹配，不一定天然校准到推荐目标。应比较 ID-only、content-only 和融合版本，并按新物品与老物品分别评估。

### 特征版本比一个漂亮公式更容易造成线上事故

训练使用的内容 encoder、图片预处理和向量归一化，必须与线上索引相容。如果换 encoder 却只更新部分商品，向量空间可能不再一致；同维度并不代表可直接比较。保存 encoder revision、生成时间和 schema，并制定重新计算与切换策略。

同样，训练时的物品属性应符合当时可用版本，避免未来内容泄漏。内容理解管线、序列训练与服务检索需要共同维护这份接口，而不是各自认为“只是一个 float 数组”。

### 把规模化结果拆成质量、效率与可靠性

效率看有效 token/s、显存、padding 比例、候选数与延迟；质量看时间切分后的排名及冷启动；可靠性看空历史、删除物品、缺失内容向量和版本切换。任何一个维度都不能用另一个代替。

浏览器实验先验证 offsets、表示与打分的小规模逻辑；真正 ragged kernel 和多机吞吐需要硬件测试。先把用户边界和候选依赖说清楚，再讨论加速，才能避免得到一个跑得更快却把用户历史串在一起的系统。

## 不同长度不是边角问题

如果 batch 的真实长度是 `[10,100,1000]`，padding 到 1000 会分配 3000 个 token 位置，只有 1110 个有效。Dense attention 的潜在成对工作是 $3\times1000^2$，而逐序列只需 $10^2+100^2+1000^2$；这只是工作量比较，不代表 kernel 能获得同倍加速。

Ragged / jagged execution 用拼接后的 values 和 offsets 表示变长序列，避免对大量 padding 做无效工作。Kernel 需要处理边界、分组与负载均衡。Stochastic Length 则在训练时采样序列长度，是改变训练计算的方法；不能与仅改变内存布局的 ragged 混为一谈。

## 候选越多，越要找可复用的历史

Target-aware ranking 要对多个候选输出预测。如果每个候选都重算用户历史，会重复大量工作。M-FALCON 思路涉及 microbatch candidates 和缓存候选无关计算，同时保持 candidate-specific 部分正确。只有 mask 和计算依赖允许共享时才能缓存；不能把候选已经影响过的 hidden states 当通用用户表示复用。

```python
lengths = [10,100,1000]
print('padded token positions:', len(lengths)*max(lengths))
print('ragged token positions:', sum(lengths))
print('pair-work ratio:', len(lengths)*max(lengths)**2/sum(n*n for n in lengths))
```

这段给出理论工作量，不测 GPU。实际需 profiler 验证 sparse kernel、launch、cache 和 memory traffic。

## 多模态表示在哪里进入？

冷启动商品可能没有行为，但已有标题、图片或视频。内容 encoder 可生成特征，映射到推荐模型使用的空间。CLIP 相似性与用户偏好不是同一目标：两张视觉相近的图，用户行为价值可能差异很大。因此应比较 ID-only、content-only 和融合模型，按新商品、长尾与语言切片评估。

内容特征也有时间版本：如果 encoder 用未来数据更新，离线回放不能直接使用新特征冒充当时可用信息。Serving 还要计入 encoder 离线/在线计算、feature store 新鲜度和缺失回退。

## 自测

**问题：** 加入多模态向量后，新商品指标提高，是否足以证明整体上线？

<details><summary>展开答案</summary>还需检查总体质量、成熟商品退化、特征延迟与缺失、成本及线上实验。改进一个切片是证据的一部分，不是全局结论。</details>

## 原始资料与继续阅读

- [HSTU original paper](https://arxiv.org/abs/2402.17152)
- [Official generative-recommenders implementation](https://github.com/meta-recsys/generative-recommenders)
- [Official HSTU benchmark example](https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/README.md)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
