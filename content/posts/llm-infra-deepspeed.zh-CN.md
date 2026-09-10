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

## 从零开始的完整讲解

### 为什么参数只有十几 GB，训练却装不下

推理主要需要权重和运行时状态，训练还需要梯度与优化器状态。混合精度 Adam 的一种常见账本是：每参数 2 bytes 低精度权重、2 bytes 梯度、4 bytes master weight、两个 4-byte moment，共 16 bytes。具体优化器和实现可不同，因此这个数值是明确约定下的计算，不是所有训练的定律。

以 7B 为例，持久训练状态约 112 GB，即约 104.3 GiB，还未包含 activation。看到“模型权重只有 14 GB”就选择 16 GB 显卡训练，漏掉的正是这些状态。

### ZeRO 是逐步消除副本冗余

在普通 data parallel 中，每个 rank 常保存相同的优化器状态、梯度和参数。ZeRO-1 先分片优化器状态，ZeRO-2 再分片梯度，ZeRO-3 连参数也分片。它保持逻辑上的数据并行训练，却改变每个设备长期持有什么。

假设有 N 个 rank，前面的账本下，Stage 1 的每 rank 状态约为 `4P+12P/N` bytes；Stage 2 为 `2P+14P/N`；Stage 3 为 `16P/N`。这些式子没有包含临时收集参数、激活、通信 buffer 与内存碎片，因此不能拿 Stage 3 的数值直接当峰值显存。

### 分片后，算某一层时参数从哪里来

参数分片并不意味着一个设备能凭 1/N 的参数独立算完整层。计算需要的权重会按调度被收集或访问，使用后可以释放；反向还要通信以形成归属正确的梯度分片。省内存与增加通信、调度的关系，是理解 ZeRO-3 的核心。

若一个层本身很大，临时收集这一层仍可能产生显存峰值。prefetch 能尝试隐藏通信，但过度预取又会占用更多内存。配置需要围绕实际峰值和时间线调整，而不是把所有“更积极”的选项都打开。

### Activation checkpointing 解决另一笔账

前向中的中间激活需要留给反向使用。Checkpointing 少存一部分激活，反向时重算，拿计算换内存。它不等于分片 optimizer，也不一定按相同比例减少参数状态。长序列训练里，即使 ZeRO 已经明显减小持久状态，激活仍可能成为主要内存来源。

例如优化前参数状态占 60 GiB、激活占 20 GiB；把参数状态降到 10 GiB 后，激活就成了大头。瓶颈转移很正常，不代表 ZeRO 没有效果。

### CPU/NVMe offload 是容量交换，不是魔法

把某些状态放到 CPU 内存或 NVMe 可以降低 GPU 常驻需求，但每一步可能增加数据迁移或 CPU 更新成本。主存容量、PCIe 带宽、NUMA 布局和存储性能都可能影响结果。能运行一个更大模型与更快训练同一个模型，是两个不同目标。

比较前先确认你需要容量、吞吐还是两者兼顾。若模型已能稳定放入 GPU，额外 offload 未必值得；若没有它完全无法运行，则即便更慢也可能有价值。

### 用 Notebook 建立自己的显存预测

改变参数量、rank 数和 stage，先记录持久状态理论值，再给激活与临时缓冲留独立预算。真实训练时收集峰值 allocated/reserved memory，并结合不同实现指标的含义解释差异。不要用 OOM 前一次打印的静态值否认瞬时峰值。

最后验证 loss 和梯度更新仍符合基线。省下内存但忘记有效 token 的归一化、梯度累积或学习率规则，会得到“能跑却训练了另一件事”的系统。

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
