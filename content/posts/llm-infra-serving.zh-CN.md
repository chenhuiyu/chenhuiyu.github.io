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

## 从零开始的完整讲解

### 请求进入服务后，时间花在哪里

用户点发送后，服务可能先排队、分词、组 batch，再执行 prompt 的 prefill，输出第一个 token，随后多次 decode，最后把结果传回客户端。用户看到的等待时间包含这些步骤，不只是 GPU kernel 的执行时间。

TTFT 是从请求开始到第一个 token 的时间；ITL 描述相邻输出 token 之间的间隔；端到端延迟包含整段回答。若输出 N 个 token，粗略关系是 `总时间 ≈ TTFT + (N-1)*平均 ITL`，但必须统一网络、首 token 和流式缓冲的计时边界。框架内部计时和用户端计时不是同一指标。

### Prefill 和 decode 为什么需要不同的思考方式

Prefill 一次处理多个已知 prompt token，矩阵乘法有更多并行工作。Decode 需要先决定上一步 token 才能继续下一步，单条请求每次只有少量新位置，往往更难充分复用权重。批量 decode 可以提高复用，但增加并发也会消耗更多 KV 显存并改变排队行为。

常说 prefill 偏计算、decode 偏带宽，是理解典型场景的起点，不是所有设备和长度下的定律。长上下文 attention、MoE 通信、低 batch 的 kernel launch 都可能成为实际主导因素。

### 用两笔账排除不可能的配置

第一笔是静态权重：7B 参数若每个 2 bytes，仅权重约 14 GB，约 13.0 GiB。第二笔是运行时状态：KV、激活、通信缓冲与分配器余量。不能把“权重放得下”直接当成“8 个长请求也跑得下”。

对固定结构，KV 通常近似正比于所有在途序列的累计 token 数。下方 Notebook 可以改变序列长度与并发，看到线性变化。若某服务允许每条请求最多 32k，不意味着可以同时容纳任意数量的 32k 请求。

### 排队可以让一个快 kernel 变成慢服务

假设平均每秒到达 10 个请求，平均在系统中停留 2 秒，在稳定、定义一致的观测条件下，Little's Law 给出平均在途请求数约 20。它是流量、驻留时间和并发的关系，不是容量保证。随着到达率逼近服务能力，排队可能迅速增长。

闭环压测让每个客户端收到回答才发下一个请求；开放式压测按既定到达率发送。前者可能在服务变慢时自动降低压力，掩盖过载。比较结果时要说明采用哪种模式，并记录超时、拒绝和失败请求，不能只统计成功的快请求。

### 吞吐率提升可能牺牲哪一类用户

增大 batch 可能提高总 token/s，但新来的短请求可能需要等待长 prefill。Chunked prefill 把长输入拆成调度片段，帮助平衡首 token 与持续输出的体验；具体大小需要根据 workload 实测。单纯追求最高 GPU 利用率，会忽略交互产品对尾部延迟的要求。

为实际应用定义 SLO，例如大多数短问答应在某时限内开始输出，同时回答质量不能下降。这里的数值应来自产品需求，不应照抄其他公司的 benchmark。

### 一个值得执行的性能实验

固定模型、精度、硬件、输入/输出长度分布，逐步改变并发。每档记录请求吞吐、输出 token/s、p50/p95 TTFT、ITL、显存峰值及错误率。画出质量合格且满足延迟约束的区域。下方浏览器计时只测本机 Python 循环，帮助理解预热和重复测量；它不是任何 GPU 推理框架的性能排名。

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
