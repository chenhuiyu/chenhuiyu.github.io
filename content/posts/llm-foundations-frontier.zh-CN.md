---
title: "走向前沿：MoE、长上下文、推理时计算与验证"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "zh-CN"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-frontier"
slug: "llm-foundations-frontier-zh"
excerpt: "用计算、内存、通信和验证四个问题理解前沿设计，避免只背模型名和排行榜。"
series: "llm-foundations"
seriesOrder: 8
draft: false
---

![走向前沿：MoE、长上下文、推理时计算与验证](/learning/llm-foundations-frontier-zh.svg)

## 前沿是几个瓶颈的重新分配

本章不把“SOTA”当成永久冠军。模型能力、任务定义、成本预算和软件版本都在变化。更持久的读法是：它减少了什么，代价转移到了哪里，哪些实验能反驳它？读报告时分别记录训练计算量、推理激活参数、总参数、上下文长度与实际吞吐。

## MoE：少算一些专家，增加路由问题

Mixture-of-Experts 通常用 router 给每个 token 选择部分 FFN experts。总参数决定存储需求，active parameters 更接近每 token 的计算，但不等于实际 FLOPs 或延迟。负载不均、跨卡 all-to-all 和小矩阵效率都可能吞掉计算收益。Expert parallelism 不等于普通 tensor parallelism。

例如 8 个专家每次选 2 个，仅专家 FFN 部分激活 1/4，并不说明整个模型计算下降到 1/4，因为 attention、router、共享专家及通信依然存在。报告同时需要路由负载分布和 token drop / capacity 策略。

## 长上下文：装得下，不代表用得好

GQA 减少 KV heads，MLA 以低秩 latent 缓存等设计压缩缓存；这两种架构不能套用同一条 KV 公式。RoPE 扩展处理位置外推，检索/压缩处理证据选择，稀疏 attention 改变可读范围。它们作用于不同层面。长上下文测试应控制证据位置、干扰文本、所需推理跳数，并记录 TTFT 和缓存成本。

## 推理时计算：多想之后必须能验证

多次采样、搜索、过程推理与工具使用，把部分预算从训练移到测试时。数学、代码等任务可以用可验证反馈训练或筛选，但 verifier 有覆盖范围：通过测试并不保证没有未覆盖 bug。把更多生成 tokens 与更高 pass@k 分开报告，不能拿 k=64 对比别人 k=1 后只讲模型更聪明。

```python
# 完整枚举的 toy pass@k：独立同分布假设下的示意，非真实估计器。
p = 0.2
for k in [1, 4, 16]:
    print(k, 1 - (1-p)**k)
```

实际样本相关、预算不同，且使用有限样本估计 pass@k 时需要正确无偏估计器。本式只解释为什么增加尝试次数本身会提高“至少成功一次”。

## 自测

**问题：** 模型总参数翻倍、active parameters 不变，推理内存也不变吗？

<details><summary>展开答案</summary>不一定。更多 experts 的权重仍需要存储、分片或搬运。active parameters 描述一次路由参与计算的部分，不能替代总权重存储和通信预算。</details>

## 原始资料与继续阅读

- [DeepSeek-V2 / MLA and MoE](https://arxiv.org/abs/2405.04434)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
