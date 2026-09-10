---
title: "DeepSpeed ZeRO：把训练状态逐层拆账"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "zh-CN"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-deepspeed"
slug: "llm-infra-deepspeed-zh"
excerpt: "从 Adam 的权重、梯度与优化器状态出发，算清 ZeRO-1/2/3 的收益与通信代价。"
series: "llm-infra"
seriesOrder: 6
draft: false
---

![DeepSpeed ZeRO：把训练状态逐层拆账](/learning/llm-infra-deepspeed-zh.svg)

## 参数量为什么不能直接换算训练显存？

以一种常见 mixed-precision Adam 配置为例：低精度权重 2 bytes、低精度梯度 2 bytes、FP32 master weights 4 bytes、两个 FP32 moments 共 8 bytes，每参数合计 16 bytes。不是所有 BF16 训练都保存 master weights，也不是所有梯度都是 2 bytes；计算前必须明确实现。

7B 参数按这个假设仅持久训练状态就约 104.3 GiB，尚未算 activation、attention workspace 和通信缓冲。推理权重只有 14 GB 左右，不能据此断言同显存能完整训练。

## ZeRO 分阶段消除什么冗余？

| 阶段 | 被 data-parallel ranks 分摊的状态 | 本地仍保留 |
|---|---|---|
| ZeRO-1 | optimizer states，包括本例 master weights | 权重、梯度 |
| ZeRO-2 | 再分摊梯度 | 权重 |
| ZeRO-3 | 再分摊参数 | 当前计算需要的临时 gather |

在相同 P 个 ranks 下，本例每 rank 持久 bytes 从 $16N$ 变为 $4N+12N/P$，再到 $2N+14N/P$，最后约 $16N/P$。下方 Python 可改 N、P 验算。实际峰值可能远高于持久状态下界，因为计算前会收集当前层参数。

## 与 TP 和重计算有什么不同？

ZeRO 聚焦 data-parallel replicas 的冗余；TP 把单个算子并行化，两者可以组合。Activation checkpointing 则减少保存的前向激活，在反向时重新计算，用算力换内存。CPU/NVMe offload 把存储压力搬到更慢介质，还会增加搬运，不能直接把容量收益当吞吐收益。

配置 stage=3 不是最后一步。需要查看 parameter gather、梯度 reduce-scatter 与计算是否重叠；microbatch 太小可能让通信主导。对于无法装下的单层，也可能仍需 TP、分块或其他策略。

## 自测

**问题：** 8 卡 ZeRO-3 的状态下界除以 8，峰值也一定恰好除以 8 吗？

<details><summary>展开答案</summary>不是。激活、临时 gather、通信 buffer 与未分片对象未必等比例缩小。应记录实际峰值和最大单层参数，而不是只看静态公式。</details>

## 原始资料与继续阅读

- [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [ZeRO paper](https://arxiv.org/abs/1910.02054)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
