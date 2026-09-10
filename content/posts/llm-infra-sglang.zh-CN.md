---
title: "SGLang：共享前缀、结构化生成与 PD 分离"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Infrastructure"
language: "zh-CN"
tags: ["llm-infra", "Hands-on", "Model Learning"]
pairKey: "llm-infra-sglang"
slug: "llm-infra-sglang-zh"
excerpt: "从重复系统提示出发理解 RadixAttention，再计算把 KV 跨机器传输的代价。"
series: "llm-infra"
seriesOrder: 3
draft: false
---

![SGLang：共享前缀、结构化生成与 PD 分离](/learning/llm-infra-sglang-zh.svg)

## 相同开头为什么值得复用？

多轮对话、few-shot 分类和 agent 工具调用经常共享开头。若 token、模型与位置等条件一致，prefix 的 K/V 可以复用。Radix tree 压缩共享前缀，按分叉组织后续部分；它与存储层的分页是不同抽象。RadixAttention 的关键不是把相似句子聚类，而是复用精确匹配的计算。

假设请求分别是 `[A,B,C,X]` 和 `[A,B,C,Y]`，共有前三个 token。第二个请求可能节约前三个位置的 prefill，但第四个位置与后续 decode 仍需计算。如果开头加入变化的时间戳，后面再长的公共文本也未必能作为连续 prefix 命中。

## 结构化输出有边界

约束解码可以限制某一步允许生成的 token，使输出满足语法或 schema。合法 JSON 不等于字段事实正确，工具名合法不等于调用合理。评估时分别检查语法通过率、schema 通过率、语义正确率与额外延迟。约束很强时，采样分布被改变，也需要检查质量。

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 --port 30000
```

安装和硬件支持按官方文档核对。使用相同模型与请求集比较框架，而不是拿默认配置对默认配置宣称总冠军。

## PD 分离不是免费午餐

Prefill worker 生成 KV 后传给 decode worker。传输下界约为 bytes/bandwidth，还要计入调度、协议、布局转换和同步。在教学假设中，2 GiB cache 经有效 25 GiB/s 链路传输至少需 80 ms。这个数字不是 NIC 标称速率推导出的真实延迟，更不代表可以忽略连接与竞争开销。

如果原本 prefill 很短、网络慢、复用差，分离可能更慢。长输入、高并发与不同阶段争抢资源时才更值得验证。当前 SGLang 文档包含 Mooncake、NIXL 等传输路径，适用条件与配置应以所用版本为准。

## 自测

**问题：** 两个系统提示意思完全一样但标点不同，prefix cache 一定命中吗？

<details><summary>展开答案</summary>不一定，通常依赖 token 级精确前缀及运行上下文。语义相似不是缓存正确性的依据。</details>

## 原始资料与继续阅读

- [SGLang paper](https://arxiv.org/abs/2312.07104)
- [SGLang PD disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)
- [SGLang request tutorial](https://docs.sglang.io/docs/basic_usage/send_request)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
