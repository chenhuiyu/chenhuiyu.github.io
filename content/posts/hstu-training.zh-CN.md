---
title: "训练一个微型 HSTU 推荐器：数据、Loss 与 Top-k"
date: "2026-09-10"
updated: "2026-09-10"
category: "HSTU"
language: "zh-CN"
tags: ["hstu", "Hands-on", "Model Learning"]
pairKey: "hstu-training"
slug: "hstu-training-zh"
excerpt: "从零训练可执行的 PyTorch 教学模型，检查因果性、padding、梯度与推荐输出。"
series: "hstu"
seriesOrder: 3
draft: false
---

![训练一个微型 HSTU 推荐器：数据、Loss 与 Top-k](/learning/hstu-training-zh.svg)

## 小模型也要形成闭环

本章 Notebook 不依赖私有日志或付费 GPU：生成两种可控行为模式，训练一个包含 U/Q/K/V、SiLU 聚合、门控和残差的单头模型，输出下一商品的 logits。它使用合成数据和简化时间结构，用于理解机制，不用于推断线上效果。

输入 `[B,T]` 商品 IDs 经 embedding 到 `[B,T,D]`，HSTU-inspired block 输出同形状表示，再乘输出头得到 `[B,T,V]`。所有有效位置都可以监督 next-item，而非只训练最后一个位置。目标 0 是 padding，loss 用 ignore_index=0 排除它。

## 一次 backward 不代表学到了

先固定 seed，训练少量确定性序列，观察 loss 是否下降。打印梯度范数检查有效参数是否更新，比较训练前后 top-k。再用不同起点的保留序列测试，记录真实指标。合成规则简单，良好结果只说明在这个受控任务上学会了模式。

```python
# 与 Notebook 中定义的 model / inputs / targets 一起使用。
logits = model(inputs)
loss = torch.nn.functional.cross_entropy(
    logits.flatten(0,1), targets.flatten(), ignore_index=0)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

本段是训练步骤，不是独立脚本。完整模型定义、数据、优化器、训练循环、曲线与断言都在 Notebook。

## 三个不能省略的正确性检查

第一，改变未来 token 后，较早位置的输出应不变，验证因果遮罩。第二，把 padding embedding 换成随机非零值，合法输出仍应不受被 mask 的 padding 影响。第三，在评估 top-k 前屏蔽保留 ID 0，避免把 padding 当推荐。

还要把“训练 loss 下降”与“验证指标提高”分开，不能用同一批数据证明泛化。若只在训练集上跑通，则称为 implementation sanity check。Notebook 的 held-out 合成模式也不是现实世界独立测试集。

## 自测

**问题：** 训练 loss 很低但推荐总是 ID 0，先查什么？

<details><summary>展开答案</summary>检查 padding 是否参与监督、有效标签位置、输出候选过滤和最后有效位置索引。也检查是否取了 batch 的最后 padding 位置，而不是序列真实末尾。</details>

## 本次真实运行结果

![3538 参数的教学模型实际训练曲线；合成任务的收敛不代表工业推荐效果。](/learning/hstu-training-actual.png)

3538 参数的教学模型实际训练曲线；合成任务的收敛不代表工业推荐效果。

## 原始资料与继续阅读

- [Official generative-recommenders implementation](https://github.com/meta-recsys/generative-recommenders)
- [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
