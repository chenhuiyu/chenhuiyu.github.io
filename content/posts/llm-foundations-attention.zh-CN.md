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

## 从零开始的完整讲解

### 为什么每个词需要读取别的词

先想一个没有 attention 的模型：它把每个 token 单独变成向量，却不允许位置之间交换信息。“她把书放进包里，因为它很重”中的“它”究竟指什么，必须依赖上下文。Attention 提供一种可学习的信息读取操作：每个位置提出一个查询，对可见位置打分，再汇总它们携带的内容。它不保证模型解决了指代问题，只提供了表达这种依赖关系的计算工具。

Q、K、V 不是三份新的输入文本，而是同一个输入表示经过三组参数投影后的结果。对于输入 `X: [T,d_model]`，`W_Q: [d_model,d_k]`，乘法 `X @ W_Q` 产生 `Q: [T,d_k]`。K 用相同的 key 维度，才能和 Q 做点积；V 的最后一维可以不同，因为它负责承载被汇总的信息。

### 把一行注意力完整算出来

假设当前 query 是 `[1,0]`，有两个 key `[1,0]` 与 `[0,1]`，对应 value `[10,0]` 与 `[0,20]`。点积得到 `[1,0]`，除以根号 2 后约为 `[0.707,0]`。指数约为 `[2.028,1]`，归一化得到权重 `[0.670,0.330]`。输出就是 `0.670*[10,0]+0.330*[0,20]`，约为 `[6.70,6.60]`。

注意，输出不是某个 key，也不是权重本身；它是 value 的加权和。为了查看中间步骤，可以在下方代码单元运行：

```python
import math
scores = [1/math.sqrt(2), 0.0]
z = [math.exp(s-max(scores)) for s in scores]
a = [v/sum(z) for v in z]
values = [[10,0], [0,20]]
out = [sum(a[j]*values[j][d] for j in range(2)) for d in range(2)]
print('weights =', a, 'output =', out)
```

把第一个 value 从 `[10,0]` 改成 `[100,0]`，权重完全不变，输出却会变化。这一操作能帮助你把“从哪里读取”和“读到什么”区分开。

### 因果性是一条信息边界

自回归训练在位置 t 预测下一个 token，不能提前读取未来答案。对四个位置，第一行只有一个合法 key，第二行两个，直到最后一行四个。矩阵中的未来位置在 softmax 之前被置为负无穷，指数后正好为零。

这里有一个工程陷阱：若某行所有位置都被遮住，直接对全负无穷做 softmax 会产生未定义的数值。实现需要保证有效 query 至少有合法 key，或显式处理全遮罩行。Padding mask 与 causal mask 合并时尤其容易出现这个问题。另一个陷阱是把不同库中“True 表示允许”与“True 表示屏蔽”的约定混用。

### Multi-head 不是简单重复同一次计算

多个头使用不同投影，在不同表示子空间中读取上下文。设 `d_model=512`、8 个头，每头 key dimension 常取 64。每个头产生一份输出，再沿特征维拼接，通过输出投影回到模型宽度。不能说某个头必然对应语法、某个头必然对应情感；这种角色需要实验观察，训练并没有人工写好分工。

标准稠密 attention 的分数矩阵有 `T*T` 个元素。序列从 1,024 增加到 4,096，长度增加 4 倍，矩阵元素增加 16 倍。这个数量关系解释了长上下文的压力；但具体延迟还受 batch、内存访问和内核实现影响，不能简单断言端到端服务也慢 16 倍。

### 阅读热力图时应该追问什么

每一行是一个 query 的读取分布，每一列是一个 key 位置。先核对轴方向、mask 和行和，再讨论“模型关注哪里”。强权重不等于因果重要性：value 的幅度、输出投影、残差路径以及后续层都会影响最终答案。若想验证某个 token 是否影响决策，需要替换、删除或干预实验，并注意这种操作也可能改变输入分布。

下方 Notebook 可以修改输入再画最后一行的真实权重。建议先做预测：统一给所有合法分数加 10 会怎样？把温度提高会怎样？先写下预期，再运行代码，才能区分理解了机制还是只看了一张漂亮的图。

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
