---
title: "HSTU 的起点：推荐是在预测什么序列？"
date: "2026-09-10"
updated: "2026-09-10"
category: "HSTU"
language: "zh-CN"
tags: ["hstu", "Hands-on", "Model Learning"]
pairKey: "hstu-sequence"
slug: "hstu-sequence-zh"
excerpt: "区分内容、行为、目标与候选，理解 sequential transduction 和数据泄漏的边界。"
series: "hstu"
seriesOrder: 1
draft: false
---

![HSTU 的起点：推荐是在预测什么序列？](/learning/hstu-sequence-zh.svg)

## 先别急着替换 Transformer

用户看到内容，产生点击、跳过、停留或购买等行为。HSTU 所在的生成式推荐框架把这类时间过程作为建模对象。“生成式”不意味着一定输出自然语言，也不意味着必须生成 Semantic ID。Retrieval 可以预测下一个内容表示，ranking 则可以在候选条件下预测行为。

本路线是现有生成式推荐第 16 章的代码延伸；[原专题概览](/blog/generative-recommendation-16-hstu-zh) 介绍论文背景，这里重点放在数据到张量的转换。

## 监督的时间边界

假设历史为 `(网球,点击,t1)`、`(攀岩,长观看,t2)`，t3 到来一个游泳视频候选。预测 t3 行为时，能读取前两次真实行为和当前候选内容，不能读取 t3 已完成的观看时长。时间戳、停留统计或聚合特征如果使用未来更新值，也会泄漏。

离线 next-item 教学任务常用 `items[:-1]` 预测 `items[1:]`；这与工业 content/action 交织输入不是完全相同格式。Notebook 明确使用前者以便在 CPU 上训练，并不会声称复现论文工业系统。

## 从列表到 batch

```python
import torch
sequences = [[1, 3, 4, 2], [2, 4, 1]]
inputs = torch.tensor([[1,3,4], [2,4,0]])
targets = torch.tensor([[3,4,2], [4,1,0]])
valid = inputs.ne(0)
print(inputs.shape, valid.sum(1))
```

这里 0 是保留 padding ID，不能作为普通商品参与 top-k。真实特征空间可包含 item、creator、类型、时间与内容 embedding；不同来源应注明更新时刻与缺失策略。未经训练的特征拼接不自动成为更好的模型。

## 评价之前先定义候选

在全部商品中检索，与从 100 个抽样负例中挑一个，难度完全不同。负例采样分布也会影响结果，尤其 popular-item 偏置。记录是否过滤历史已见商品；如果业务允许复购，机械过滤可能删掉正确目标。切分应遵守全局时间和特征可用性，而不仅每用户最后一条。

## 自测

**问题：** 只用预测时刻之前的行为序列，是否保证无泄漏？

<details><summary>展开答案</summary>不保证。商品 embedding、流行度、用户统计与候选集也可能由未来数据产生。需要逐字段检查 point-in-time 可用性。</details>

## 原始资料与继续阅读

- [HSTU original paper](https://arxiv.org/abs/2402.17152)
- [Official generative-recommenders implementation](https://github.com/meta-recsys/generative-recommenders)
- [SASRec baseline](https://arxiv.org/abs/1808.09781)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
