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

## 从零开始的完整讲解

### 先画清楚一个 block 的输入与输出

一个 Transformer block 接收 `[B,T,d]`，通常也输出 `[B,T,d]`：样本数、位置数和模型宽度保持不变，变化的是每个位置包含的信息。因为形状一致，多个 block 可以逐层堆叠。读架构图时不要把“层数增加”误解为“句子变长”；增加的是对同一组位置反复加工的次数。

可以把一个常见 pre-norm block 写成两个子步骤：`h = x + Attention(Norm(x))`，然后 `y = h + FFN(Norm(h))`。这不是所有 Transformer 的唯一排列，post-norm、并行残差等变体都存在。先掌握一种布局，再查看具体模型配置中的差异。

### 残差为什么是相加，而不是覆盖

如果每一层都必须重新生成整个表示，深层优化会很困难。残差让模块学习“在已有表示上增加什么修正”。假设输入向量 `[1,2]`，模块输出修正 `[0.1,-0.2]`，相加后得到 `[1.1,1.8]`。即使当前模块输出接近零，原始信号仍能经过这条路径。

从导数看，`y=x+f(x)` 的局部 Jacobian 是 `I+J_f`。恒等项提供直接的梯度路径，但不能因此保证所有深层模型都没有梯度问题。学习率、初始化、归一化位置与精度仍会决定稳定性。

### 归一化处理的是哪根轴

LayerNorm 通常在每个位置的特征维度上计算均值和方差，不是把整段文字混在一起归一化。例如一个位置的 `[1,2,3]`，均值为 2，减均值后是 `[-1,0,1]`，再除以标准差，并应用可学习的缩放和偏移。RMSNorm 则根据均方根缩放，通常不做减均值操作。

这说明归一化不会代替 attention：它通常没有读取别的位置，不能凭空产生跨词上下文。它的作用是管理特征尺度，使后续模块面对更可控的输入分布。实现里的 epsilon 防止除零，不能随意删掉。

### FFN 看似逐位置，为什么仍然重要

Attention 负责在位置之间搬运信息；FFN 对每个位置已经汇总的特征做非线性变换。最简单的结构是先升维、过激活，再降维。设模型宽度 512、中间宽度 2,048，第一矩阵有 `512*2048` 参数，第二矩阵同样多，总计约 210 万，不含 bias。这解释了为什么只关注 attention 会漏掉大量参数和计算。

如果去掉中间非线性，两次线性变换可以合并成一个矩阵，表达能力就受限。门控 FFN 则用一条支路决定另一条支路的信息通过程度；SwiGLU 常见形式涉及 SiLU 门控，但它仍不是“为每个词选择一个专家”的 MoE 路由。

### 位置信息为什么不能省

仅靠内容点积，模型难以区分“狗追猫”和“猫追狗”的顺序结构。绝对位置 embedding 直接给不同位置加上不同向量；RoPE 则旋转 query/key 的成对特征，使点积包含相对位置关系。它不是把时间戳简单拼到输入字符串中。

长上下文扩展不仅需要能给更远的位置算出数字，还要考虑训练中是否见过这些距离、旋转频率如何处理、模型能否从中间位置检索信息。配置接受更长输入，不等于远距离理解已得到验证。

### 如何调试一个层，而不是盲目堆更多层

在模型输入输出 Notebook 中打印 embedding 输出、每层 hidden state 和最终 logits 的形状。比较同一 token 在不同上下文中的向量，确认“查表向量”与“上下文向量”的区别。再检查残差两侧形状相等，归一化轴是最后一维，FFN 回到原始宽度。

若训练出现 NaN，先找第一次出现非有限值的层，检查激活范围、loss 与梯度，不要只把学习率无限调小。若增加层数后效果不升，可能是数据、目标或优化不足，不能仅凭参数规模判断原因。

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
