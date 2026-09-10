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
