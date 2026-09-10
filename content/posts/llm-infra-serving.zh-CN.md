---
title: "推理为什么慢：Prefill、Decode、KV cache 与排队"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "zh-CN"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-serving"
slug: "llm-infra-serving-zh"
excerpt: "先建立延迟与显存账本，再讨论 continuous batching、分页和 prefill/decode 分离。"
series: "llm-infra"
seriesOrder: 1
draft: false
---

![推理为什么慢：Prefill、Decode、KV cache 与排队](/learning/llm-infra-serving-zh.svg)

## 从用户看到的延迟开始

一次请求的首 token 延迟 TTFT 包含排队、预处理、prefill 和首步输出相关开销；后续 token 间隔通常用 ITL 或 TPOT 描述。总延迟近似为 TTFT 加后续 decode 时间，TPOT 的分母惯例要记录清楚。吞吐高并不保证单用户响应快，尤其在接近容量极限时排队会变长。

Prefill 对大量 token 做矩阵乘，常有较高算术强度；小 batch decode 每步生成很少 token，常受权重和 KV 读取限制。但这不是绝对分类：batch、上下文长度、并行方式与硬件都会改变瓶颈。

## KV 的账要用 KV heads 算

普通 MHA/GQA/MQA 的缓存近似为 $2BTLH_{kv}D_hs$ bytes。2 代表 K 和 V，s 是每元素字节数。32 层、8 KV heads、head dimension 128、4096 tokens、4 个并发、FP16 的缓存是 2 GiB。这里不包括权重、workspace、激活和分页浪费。MLA 等架构需要另建模型。

下方计算器让你实际改变并发、长度与 head 数。它输出架构预算，不是某块 GPU 的实测，也不应该拿整卡显存全部分给 KV。量化 cache 还可能需要 scale 元数据和不同 kernel 支持。

## 三种优化分别解决什么？

Continuous batching 在迭代边界接纳新请求、移除已结束请求，减少等待最长序列的浪费。分页把逻辑 token blocks 映射到物理 cache blocks，降低预留和碎片成本。Prefill/decode 分离把两种执行阶段放到不同资源池，代价是 KV 传输、路由和容量配比复杂化。三者不是互斥选项，也不是无条件加速。

## 动手与自测

运行 Notebook 的真实 matmul benchmark，记录 warm-up 后的多次时间；再用框架请求日志比较相同输入/输出长度下不同并发。至少保存 p50/p95 TTFT、ITL、完成吞吐、错误率和 GPU 峰值内存。请求失败不能从分母里悄悄删除。

**问题：** 提高并发后 tokens/s 上升，但 p95 TTFT 翻倍，是不是优化成功？

<details><summary>展开答案</summary>要看服务目标。可能提高了设备利用率但恶化排队。应比较满足既定延迟和错误率约束的 goodput，而不是只有裸吞吐。</details>

## 原始资料与继续阅读

- [vLLM architecture](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [SGLang PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
