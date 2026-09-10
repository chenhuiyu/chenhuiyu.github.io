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
