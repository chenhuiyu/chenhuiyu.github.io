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

## 从零开始的完整讲解

### 先把完整训练循环拆成可检查的五步

给定历史，编码器产生用户状态 h；候选物品各有表示 e；打分函数产生候选 logits；loss 比较 logits 与真实下一物品；反向传播更新可训练参数。下一轮用更新后的参数重复。训练曲线应该由这些更新产生，而不是为了展示效果手工设定。

模型输出维度取决于候选集合。对全量小词表，可以一次输出 K 个分数；真实大目录通常需要召回与候选打分的组合。这里先用小 K，是为了让每个分数与标签都能打印出来。

### 从一个用户向量推导打分头梯度

设 `z_j=sum_d h_d*W_dj`，再对 z 做 softmax。交叉熵对 z 的梯度是 `p_j-y_j`，所以对 `W_dj` 的梯度是 `h_d*(p_j-y_j)`。浏览器 Notebook 逐项实现这条式子，更新 3×3 的打分矩阵，共 9 个参数。

这个版本的聚合器和 embedding 固定，因此 loss 下降只能归因于打分头学习，不能声称 HSTU 全部模块都被训练了。完整 PyTorch Notebook 则允许梯度流过更完整的教学模型；两种设置帮助区分“计算表示”和“训练表示”。

### 为什么先做 overfit-a-tiny-batch

复杂模型训练失败时，先让它在极小数据集上降低 loss，是检查前向、标签和梯度是否连通的有效方法。如果连几条确定规则都学不会，可能存在错位标签、参数未加入优化器、错误 mask 或学习率问题。

但小样本过拟合成功只证明有能力记住这些样本，不代表泛化。合成循环数据尤其容易：最后一个物品可能已经决定下一物品，长序列模块未必被真正需要。必须通过改变规则或消融历史检验模型是否利用了顺序。

### 训练 loss 与排名指标不是同一件事

Cross-entropy 关心正确项获得多少概率。HR@k 只看正确项是否进入前 k；NDCG 还考虑排名位置。只有一个相关目标时，若它排第 r，折损增益可写为 `1/log2(r+1)`，在前 k 外记为零。两个模型可能 HR@10 一样，但正确项概率与平均名次不同。

多目标相关性、不同候选数和过滤规则会改变指标定义。报告数字前先写候选集合及相关性标签。浏览器的三物品任务即使得到很高命中率，也不意味着百万商品检索达到同样水平。

### 用四个小测试避免“训练得很好”的假象

第一，改变 padding 不应改变有效历史的预测。第二，修改未来 token 不应影响过去位置的输出。第三，确认 target 没有直接包含在当前输入特征里。第四，测试数据应来自明确隔离的时间或生成规则，而不是训练样本的复制。

这些测试检查不同风险，不能用“loss 单调下降”代替。训练过程中还应监控非有限值、梯度范数与有效样本数，避免整批标签被屏蔽却仍输出一个看似正常的平均值。

### 怎样从玩具模型走到真实推荐任务

先扩充物品数与不确定性，再加入动作、时间及多模态内容向量。每增加一类特征，都保留仅靠热门度、最后一个物品和简单 pooling 的基线。若复杂模型没有明显超过它们，优先分析数据和目标，而不是盲目增加层数。

保存随机种子、模型结构、候选规则与评价代码。线上还要考虑曝光偏差、延迟反馈和交互闭环，离线排名提升并不自动转化为用户收益。把可复现的小实验作为起点，才有条件解释更大系统中的变化。

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
