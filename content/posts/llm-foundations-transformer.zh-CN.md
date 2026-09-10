---
title: "Transformer 不只有 Attention：残差、归一化、FFN 与位置"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "zh-CN"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-transformer"
slug: "llm-foundations-transformer-zh"
excerpt: "沿着一个 decoder block 的数据流，辨认哪些运算混合 token，哪些只变换特征。"
series: "llm-foundations"
seriesOrder: 3
draft: false
---

![Transformer 不只有 Attention：残差、归一化、FFN 与位置](/learning/llm-foundations-transformer-zh.svg)

## 两种不同的信息变换

Attention 在不同 token 之间传信息；FFN 对每个位置独立使用同一组权重，改变特征表示。只把 Transformer 理解成“注意力堆叠”，就会漏掉大量参数与计算。一个常见 pre-norm block 是：$h=x+\mathrm{Attn}(\mathrm{Norm}(x))$，$y=h+\mathrm{FFN}(\mathrm{Norm}(h))$。不同模型可能有不同 norm、偏置和残差安排。

残差让每一层学习增量，提供更直接的梯度路径，但不是保证训练稳定的魔法；初始化、学习率和深度同样重要。LayerNorm 会减均值并缩放，RMSNorm 主要按均方根缩放，两者不能混称。

## FFN 为什么往往先变宽？

一个普通 FFN 先把 D 投影到更宽的 M，非线性激活后再投影回 D，使残差能相加。SwiGLU 则增加门控分支：$\mathrm{FFN}(x)=W_o[\mathrm{SiLU}(W_gx)\odot W_ux]$。这里用列向量记号；PyTorch 的存储和乘法方向需要具体核对。

给定 `[B,T,D]`，扩展后是 `[B,T,M]`。用两层线性层的 FFN 权重规模约为 $2DM$ ，三投影门控 FFN 约为 $3DM$ ，不能在相同 M 下声称参数不变。

## 没有位置会怎样？

如果没有位置或非对称遮罩，自注意力对排列是等变的：输入重排，输出跟着重排。模型需要知道顺序。RoPE 在 Q/K 的二维子空间中旋转角度，让点积携带相对位置信息；它不是把位置向量简单加到 embedding，也不是自动获得无限长上下文。

```python
import torch
x = torch.randn(2, 5, 8)
norm = torch.nn.LayerNorm(8)
ffn = torch.nn.Sequential(torch.nn.Linear(8, 32),
                          torch.nn.GELU(), torch.nn.Linear(32, 8))
y = x + ffn(norm(x))  # 仅演示 FFN 残差支路
assert y.shape == x.shape
```

## 从结构走向实验

在真实模型 Notebook 中打印层数、attention heads、hidden size 与每层 hidden states 的 shape。把模型切到 eval 模式，重复输入检查结果，而不是把训练时 dropout 造成的变化误认为采样随机性。

**问题：** FFN 不跨 token，为什么最后能理解上下文？

<details><summary>展开答案</summary>它接收的是已由 attention 混入上下文的表示。层间交替让跨位置混合与位置内非线性变换共同工作。</details>

## 原始资料与继续阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
