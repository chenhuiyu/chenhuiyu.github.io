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
