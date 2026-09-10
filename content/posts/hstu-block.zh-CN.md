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

## 从零开始的完整讲解

### 先把 block 看成“历史输入到新表示”的函数

HSTU 是 Hierarchical Sequential Transduction Unit。读论文时可以先把它当作一个接收行为序列表示、输出更新后序列表示的模块，再逐项理解其投影、聚合、归一化与门控。它不是仅仅把 Transformer 的 softmax 换一个函数就完全复现出来。

输入 X 常带 batch、sequence 和 feature 轴。实际高效实现可能使用 ragged 表示，不一定以 `[B,T,d]` 稠密存储；为了手算，本章先用单条固定长度序列。逻辑张量关系与物理存储布局是两层不同的问题。

### U、V、Q、K 分别参与哪一步

可学习投影把输入分为几条支路。Q/K 用于计算位置之间的匹配，V 提供聚合内容，U 参与门控。分支可能通过一次较大线性投影后再切片得到，以利于实现；它们仍是不同的参数子空间。

可以先追踪一种示意流程：构造包含时间或位置偏置的 QK 分数，经过 SiLU 类变换并施加合法位置约束，聚合 V，然后归一化、与 U 相关的分支逐元素相乘，投影并加残差。具体尺度、归一化与投影布局以论文和实现为准，本页玩具函数有明确省略。

### SiLU 权重为什么不应该画成概率

SiLU 定义为 `x*sigmoid(x)`。当 x=-1 时约为 -0.269，当 x=0 时为 0，当 x=1 时约为 0.731。它不像 softmax 那样把一行转换成非负且和为 1 的分布。聚合系数可以为负，行和也没有概率含义。

例如 V1=[1,0]、V2=[0,1]，系数为 [-0.269,0.731]，聚合结果就在这两个向量连线形成的凸组合范围之外。负系数可以理解为某个特征方向被减去，但不能直接说模型“不喜欢第一个物品”；后续门控和投影仍会改变输出。

### Mask 的实现位置与 softmax attention 不同

普通 softmax 常把非法 score 设为负无穷，再归一化。对 SiLU，如果直接代入负无穷，数值上可能涉及负无穷与零相乘，产生 NaN。不能机械复制 softmax 的 mask 代码。实现可以在非线性后将非法聚合系数置零，或使用经过验证的等价处理。

还要区分真实历史、padding、当前预测位置和候选位置。一个张量 shape 正确，不表示所有信息流都合法。完整因果性测试应改变未来输入并验证过去输出不变，而不是只看 mask 长得像三角形。

### 时间偏置提供了什么额外信息

同一个物品的相似度可以相同，但距离当前时间一分钟和一个月的行为含义不同。相对时间或位置偏置让匹配函数感知这种距离。它不是假定所有旧行为都无用，而是为训练提供一种可学习的差异表达。

时间分桶、单位与截断方式会影响偏置。若训练用秒、线上误用毫秒，数值可能落入完全不同区间。数据接口的错误有时比 block 公式错误更难被察觉。

### 实验应该验证哪三件事

第一，检查 SiLU 系数确实允许负数且不要求行和为 1。第二，改变时间偏置，观察聚合结果，而不只看分数。第三，检查因果与 padding 边界。浏览器 Notebook 为便于理解固定 embedding 和部分结构，只训练后续打分头；完整 PyTorch Notebook 提供更完整但仍是教学级的 HSTU-inspired block 与端到端训练。

这两种实验都不是论文中完整生产推荐系统的复现。明确省略了什么，反而能让你把每个机制与输出变化对应起来。

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
