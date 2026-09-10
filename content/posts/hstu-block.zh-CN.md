---
title: "逐张量拆解 HSTU：SiLU 聚合、时间偏置与门控"
date: "2026-09-10"
updated: "2026-09-10"
category: "HSTU"
language: "zh-CN"
tags: ["hstu", "Hands-on", "Model Learning"]
pairKey: "hstu-block"
slug: "hstu-block-zh"
excerpt: "比较 softmax attention 与 pointwise aggregation，辨认 U/Q/K/V、norm 与 residual 的各自作用。"
series: "hstu"
seriesOrder: 2
draft: false
---

![逐张量拆解 HSTU：SiLU 聚合、时间偏置与门控](/learning/hstu-block-zh.svg)

## 相似的外形，不同的聚合

标准 attention 对一行 scores 做 softmax，权重总和为 1。HSTU 的核心变化包含 pointwise activated aggregation，用 SiLU 等逐点函数处理带位置/时间偏置的 scores，然后聚合 V，再归一化与门控。它没有序列维度的 softmax，因此不能把矩阵直接解释为概率分布。

本章的单头教学式为 $A_{ij}=\mathbf1_{j\le i}\,\mathrm{SiLU}(q_i^\top k_j+b_{ij})/N$，$o_i=W_o[U_i\odot\mathrm{Norm}(\sum_jA_{ij}V_j)]+x_i$。N 在实验里是固定序列长度；真实实现的缩放、head layout、bias 和归一化请对照官方代码。

## 四个投影分别流向哪里？

输入 `[B,T,D]` 经过 norm 与投影产生 U/V/Q/K，Q/K 控制匹配，V 被聚合，U 负责输出门控。Q/K 的 head dimension 可以与 U/V 不同。融合投影可减少多个小算子的开销，但 shape 拆分必须与权重布局一致。

```python
import torch
import torch.nn.functional as F
q, k, v = [torch.randn(1,4,8) for _ in range(3)]
mask = torch.ones(4,4,dtype=torch.bool).tril()
a = F.silu(q @ k.transpose(-2,-1)) / 4
a = a.masked_fill(~mask, 0)
y = a @ v
print(a.sum(-1))  # 不要求等于 1
```

因为 SiLU 在部分负输入上为负，聚合可能带负系数。不要计算“attention entropy”后把它当概率熵，除非先定义有意义的新度量。

## 时间偏置不等同于固定遗忘

相对位置与真实经过时间可以区分“连续三个动作”和“跨三个月的三个动作”。真实论文与实现中的 bias 可学习且有分桶等细节。下方的 $-\alpha(i-j)$ 只是可解释的教学偏置，不代表论文的完整参数化。

增加 α 后更早位置可能变负；结合 norm 与门控，最终影响不一定单调。要通过输出和 ablation 检查，而非仅凭热力图颜色猜推荐结果。

## 自测

**问题：** 把 softmax 换成 SiLU，就完整实现 HSTU 了吗？

<details><summary>展开答案</summary>没有。还涉及数据组织、投影、输出门控、归一化、位置/时间偏置及系统优化。Notebook 标注为 HSTU-inspired 教学模型，不冒充完整复现。</details>

## 原始资料与继续阅读

- [HSTU original paper](https://arxiv.org/abs/2402.17152)
- [Official generative-recommenders implementation](https://github.com/meta-recsys/generative-recommenders)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
