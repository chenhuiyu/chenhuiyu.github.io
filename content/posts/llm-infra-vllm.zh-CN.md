---
title: "vLLM：从 PagedAttention 到可测量的服务"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "zh-CN"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-vllm"
slug: "llm-infra-vllm-zh"
excerpt: "把 scheduler、KV block manager、model runner 和 API 服务串起来，设计一个公平的吞吐实验。"
series: "llm-infra"
seriesOrder: 2
draft: false
---

![vLLM：从 PagedAttention 到可测量的服务](/learning/llm-infra-vllm-zh.svg)

## 从零开始的完整讲解

### 先理解传统 KV 分配浪费在哪里

生成开始时，服务通常不知道最终输出有多长。若为每条请求预留最大连续空间，短回答会留下大片未使用显存；若频繁扩大连续缓冲，又可能涉及移动与碎片。显存不足限制并发，进而影响批量计算的效率。

PagedAttention 的关键是把逻辑序列和物理存储分开。读者可以借用操作系统分页的类比：逻辑上第 0、1、2 块连续，物理上可以放在不相邻位置，由 block table 记录映射。GPU attention 内核按照映射读取 KV，模型看到的上下文顺序仍然正确。

### 手算一个分页例子

假设每块容纳 16 个 token，一条 35-token 序列需要 3 块，总容量 48，最后一块空 13 个位置。另一条 17-token 序列需要 2 块，总容量 32。分页没有消灭所有浪费，而是将尾部未使用空间控制在块粒度，并提高分配的灵活性。

```python
block = 16
lengths = [35,17,64]
for length in lengths:
    blocks = (length+block-1)//block
    print(length, blocks, 'unused slots:', blocks*block-length)
```

这个代码只计算存储槽位，不测实际内核，也没有模拟前缀共享。块越小可能减少尾部浪费，但元数据与访问管理也更复杂，不能凭这一个公式就推导最佳 block size。

### Continuous batching 改变的是调度时机

传统静态 batch 可能等一组请求全部结束再处理下一组。若一条需要 10 个 token，另一条需要 500 个，短请求结束后的位置可能没有及时被新请求利用。Continuous batching 在迭代之间调整活跃请求集合，完成者退出，新请求进入。

它不意味着所有请求都同时完成，也不意味着没有排队。Scheduler 仍需受 token budget、KV 容量和优先级等约束。理解 vLLM 应把内存管理与调度联系起来，而不是只记住 PagedAttention 一个名词。

### 前缀缓存为什么要求严格匹配

两条请求若使用完全相同的前缀，在匹配条件满足时可以复用已计算的前缀状态。自然语言“意思一样”并不足够：不同 token 序列通常不能直接共享同一份 KV。模型权重、位置处理及其他影响计算的条件也必须相容。

例如每次把动态时间戳放在 system prompt 开头，就可能破坏后面大量文本的共同前缀。优化 prompt 布局需要保持语义与权限隔离，不能为了命中率把不同用户的私有内容错误地合并。

### 从启动成功到有效比较

启动模型服务后，先发送一个固定短请求，确认 tokenizer、模板、精度和输出合理，再测长请求与并发。保存框架版本、模型 revision、GPU、并行配置和请求分布。若比较另一框架，采样设置、最大生成长度和结束条件也要一致，否则更多 token/s 可能只是回答更短或质量不同。

正式压测至少分冷启动、无缓存稳态与高复用前缀三种情况。命中缓存的结果不能冒充所有请求的速度。若性能突然下降，检查实际长度分布、缓存利用率、抢占或重计算迹象，再进一步看时间线。

### 本章实验的边界

浏览器 Notebook 能验证 KV 字节数、分页算术和 TP 代数，不能在没有 GPU 的网页里运行完整 vLLM 服务。文章保留官方启动入口和服务器侧思路，将可手算的机制与需要设备的性能测试分开。学会先预测内存变化，再用真实部署指标验证，是比复制一个启动命令更可迁移的能力。

## 不要把一个 kernel 当成整个框架

PagedAttention 是重要起点，但服务还需要接收请求、管理生命周期、调度每轮 token、处理采样和流式输出。vLLM 的架构文档把入口、核心调度与设备执行分开描述。理解层次后才能定位：CPU 预处理慢、等待 cache 资源、GPU 算子慢，是三类不同问题。

## 逻辑连续，物理可以不连续

一条序列可以由多个固定大小逻辑 blocks 构成，block table 指向实际存储。设 block size=16，一条 33-token 序列需 3 blocks，最后一个 block 只用 1 个位置。分页没有消除浪费，只把浪费限制在更细粒度，同时增加寻址与管理成本。

```python
lengths = [17, 33, 64]
block = 16
allocated = [((n+block-1)//block)*block for n in lengths]
print(allocated)  # [32, 48, 64]
print('slack tokens:', sum(allocated)-sum(lengths))  # 30
```

这里是假设每条序列独立分配、不共享前缀的容量计算。前缀缓存需要匹配 token 序列及相关上下文；不是语义相似的 prompt 都可以共享 KV。

## 跑通一个服务，再改变一个变量

下面是单 GPU 基础启动形态，需要先按官方文档安装与硬件兼容的 vLLM 版本。记录版本、模型 revision、dtype 和驱动。示例模型名称不表示性能推荐。

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000
```

先用短请求检查响应与生成停止，再测固定长度分布，逐步增加并发。Prefix cache 的收益必须分别测 cold 与 warm；如果只用相同 prompt 重放，会夸大真实业务命中率。不能同时更改 batch、量化和模型版本后把差异全算给分页。

## 自测

**问题：** 把 block size 缩到 1，是否一定最省钱？

<details><summary>展开答案</summary>内部碎片会减少，但元数据、寻址、kernel 访问和管理成本可能增加。最优选择依赖实现和请求长度分布，需要测量。</details>

## 原始资料与继续阅读

- [vLLM architecture](https://docs.vllm.ai/en/latest/design/arch_overview/)
- [vLLM quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)
- [PagedAttention paper](https://arxiv.org/abs/2309.06180)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
