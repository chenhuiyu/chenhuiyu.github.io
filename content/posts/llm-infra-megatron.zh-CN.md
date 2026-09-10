---
title: "Megatron-LM：TP、PP、DP 与序列分片究竟切哪里？"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "zh-CN"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-megatron"
slug: "llm-infra-megatron-zh"
excerpt: "通过一层矩阵乘拆开 tensor parallelism，计算通信量，并理解 pipeline bubble 与 context parallelism。"
series: "llm-infra"
seriesOrder: 5
draft: false
---

![Megatron-LM：TP、PP、DP 与序列分片究竟切哪里？](/learning/llm-infra-megatron-zh.svg)

## 从零开始的完整讲解

### 先区分“分数据”与“分模型”

Data parallel 让多个设备持有模型副本，处理不同样本，之后同步梯度。它增加数据处理能力，但每个设备仍可能需要完整模型状态。Tensor parallel 切开层内矩阵，pipeline parallel 把不同层放到不同设备；它们解决的是单设备放不下或单层计算需要协作的问题。

所有并行方式都需要说明三件事：每个 rank 持有什么，前向产生什么局部结果，哪一步通信把结果拼回逻辑上的完整计算。只说“用了 8 张卡”无法解释这些关系。

### TP 的两种切法，通信并不一样

考虑 `Y=XW`。若沿 W 的输出维度分片，`W=[W1,W2]`，每个设备计算 `XW1` 或 `XW2`，完整输出沿特征维拼接。这常称为 column-parallel，名称依赖矩阵书写约定。

若沿输入维度切，X 也相应拆成 `[X1,X2]`，结果是 `X1W1+X2W2`。局部结果形状相同，但都不完整，需要求和。下方 Notebook 用真实数字验证这个等式。把应该相加的结果拼接，会得到错误形状或错误含义，不能仅靠代码不报错判断正确性。

在两层 FFN 中，可以让第一层输出分片直接供给第二层的输入分片，减少中间不必要的收集。但非线性、门控和 attention 的具体布局会影响设计，不能对任意网络都套同一个通信模板。

### PP 的气泡从哪里来

将 8 层模型分为两段，设备 A 负责前 4 层、B 负责后 4 层。第一批数据先经过 A，B 暂时等待；最后一批离开 A 后，A 也可能等待 B 完成。没有有效工作的这些时间叫 pipeline bubble。

把大 batch 拆成多个 microbatch，可以让 A 处理后一个时，B 处理前一个。某些理想化调度下，气泡占比随阶段数增加而增大、随 microbatch 数增加而减小，但实际还受层计算不均、通信和前反向调度影响。增加 microbatch 并不是完全免费的：它影响有效 batch、激活存储和调度复杂度。

### Sequence parallel 与 context parallel 不要只按名字猜

这些术语都涉及序列维度，但覆盖的算子范围、与 TP 的组合关系和 KV 通信方式不同。Megatron 中 sequence parallel 常与 TP 配合，对部分激活与操作做序列分片；context parallel 则面向更广的长序列处理，需要正确交换或访问 attention 所需的上下文信息。阅读实现时画出每个算子的输入布局，比把两个术语都翻译成“切序列”更准确。

### 总卡数如何分配才有意义

在简单正交组合中，总设备数可以写成 `DP*TP*PP`；加入 context 或 expert parallel 后，分组与重叠关系需要按具体实现确认。并行度不只是一个乘法题：TP 高频通信通常对设备互联敏感，PP 对阶段平衡敏感，DP 对梯度同步敏感。

两台机器各 8 卡和一台 16 卡的通信拓扑可能不同。若单层跨慢链路频繁 all-reduce，即使每张卡分到的计算变少，端到端仍可能变慢。

### 建立从代数到时间线的验证链

先用小矩阵验证分片结果等价，再用小模型比较单设备与分布式 loss、梯度。允许合理浮点误差，但不能忽略系统性偏差。最后看每个 rank 的计算与 collective 时间线，检查是否有一个慢 rank 拖住所有人。

吞吐报告应同时注明有效 batch、sequence length、checkpointing 与收敛表现。更高的 samples/s 若来自更短序列或不同 batch，不是同一训练任务的公平加速。

## 先把四种切分分开

Data parallelism 让不同设备处理不同样本，最终同步梯度。Tensor parallelism 把一层的矩阵运算切开。Pipeline parallelism 把不同层放到不同 stage，让 microbatches 流过。Context parallelism 沿序列维度分摊长上下文并交换 attention 所需信息。Megatron 的 sequence parallelism 常与 TP 组合，分摊某些激活和操作；它不等同于完整的长上下文 attention 切分。

## 一层线性层可以怎样切？

用行向量记号 $Y=XW$，X 是 `[B,T,D]`，W 是 `[D,M]`。Column parallel 把 W 沿输出 M 切开，每卡得到一部分输出特征。下一层若使用 row parallel，把输入维对应的权重切片，各卡产生部分和，再用 collective 合并。可以避免每层都收集完整中间激活，但需要正确安排通信。

```python
import torch
x, w = torch.randn(3, 8), torch.randn(8, 12)
column_parts = [x @ p for p in w.chunk(2, dim=1)]
assert torch.allclose(torch.cat(column_parts, dim=1), x @ w)
row_parts = [a @ b for a,b in zip(x.chunk(2,1), w.chunk(2,0))]
assert torch.allclose(sum(row_parts), x @ w, atol=1e-6)
```

单进程模拟验证数学分解，并不测量分布式性能。

## 通信也要写进账本

理想 ring all-reduce 每 rank 传输量约为 $2(P-1)S/P$，S 是被规约张量 bytes。这个式子忽略启动延迟和拓扑；小张量往往受延迟影响，大张量更看带宽。不能仅用峰值 NVLink 带宽估计多节点网络成本。

PP 在简化 flush schedule 下 bubble 比例近似 $(p-1)/(m+p-1)$，p 为 stages、m 为 microbatches。增加 m 可减少 bubble，但改变 batch 构成、激活占用和调度。Interleaving、1F1B、重计算等要按实际 schedule 分析，不能套一个公式解释所有情况。

## 自测

**问题：** TP 从 2 增加到 8，单层计算分到更多卡，延迟是否必然下降？

<details><summary>展开答案</summary>不一定。更小 GEMM 的效率可能下降，collective 次数和通信延迟可能主导。先确认内存是否必须分片，再在目标拓扑测 strong scaling。</details>

## 原始资料与继续阅读

- [Megatron Core features](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/index.html)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [Sequence parallelism](https://arxiv.org/abs/2205.05198)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
