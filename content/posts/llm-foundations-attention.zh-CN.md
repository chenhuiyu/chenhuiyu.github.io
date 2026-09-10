---
title: "亲手算一遍 Attention：Q、K、V 与因果遮罩"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "zh-CN"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-attention"
slug: "llm-foundations-attention-zh"
excerpt: "用四个 token 看懂打分、归一化和加权聚合，理解为什么遮罩必须放在 softmax 之前。"
series: "llm-foundations"
seriesOrder: 2
draft: false
---

![亲手算一遍 Attention：Q、K、V 与因果遮罩](/learning/llm-foundations-attention-zh.svg)

## 一次带权查找

当前 token 想从上下文读取信息，用 query 表示“我要什么”，key 表示“我有什么”，value 表示“真正交出去的信息”。这是理解投影角色的比喻，不意味着每一维对应可解释概念。三个矩阵通常都由输入乘不同的可学习权重得到。

设 $Q,K\in\mathbb R^{T\times d_k}$，$V\in\mathbb R^{T\times d_v}$。先算 $S=QK^\top/\sqrt{d_k}$，形状是 `[T,T]`，再逐行归一化，最后 $O=AV$，形状是 `[T,d_v]`。$A_{ij}$ 表示位置 i 从位置 j 读取 value 的权重，不能直接解释成“词 j 导致了答案”。

## 为什么除以根号维度？

如果 q、k 各维近似独立、均值为零、方差为一，点积方差随 $d_k$ 增长。除以 $\sqrt{d_k}$ 让分数尺度更稳定，避免 softmax 过早饱和。这是初始化与数值稳定的动机，不是任意训练后分布都严格满足的定理。

```python
scores = q @ k.transpose(-2, -1) / q.shape[-1]**0.5
future = torch.ones(T, T, dtype=torch.bool).triu(1)
scores = scores.masked_fill(future, float('-inf'))
weights = scores.softmax(dim=-1)
out = weights @ v
```

上面的片段依赖已定义的 PyTorch 张量 q/k/v 与 T，完整可执行版本在 Notebook。若 softmax 后再把未来权重归零却不重新归一化，每行和会小于 1，计算已不是相同的注意力。

## 遮罩与温度实验

下方实验固定四组二维 Q/K；先猜第一行在因果模式下是什么，再关闭遮罩。只有一个合法 key 时，第一行权重必为 `[1,0,0,0]`，与温度无关。温度升高后，分布在合法位置之间趋于均匀。它不应向未来位置泄漏。

这里显示完整矩阵便于理解；真实长上下文不会希望把每层每头的 $T^2$ 矩阵都存下来。FlashAttention 分块重排内存访问，保留 exact attention 的数学目标，不是简单删掉低权重元素。

## 自测

**问题：** 给每行的所有合法 score 都加 100，输出会改变吗？

<details><summary>展开答案</summary>数学上不会。softmax 对同一行的统一平移不变。实现通常先减去最大值以免 exp 溢出。给不同列加不同偏置则会改变权重。</details>

## 原始资料与继续阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
