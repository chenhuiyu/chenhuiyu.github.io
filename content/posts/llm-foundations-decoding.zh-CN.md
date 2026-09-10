---
title: "从 logits 到一句话：采样、KV cache 与停止条件"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "zh-CN"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-decoding"
slug: "llm-foundations-decoding-zh"
excerpt: "亲手计算 softmax、温度与 top-p，分清生成策略和模型分布，也理解缓存如何避免重复计算。"
series: "llm-foundations"
seriesOrder: 5
draft: false
---

![从 logits 到一句话：采样、KV cache 与停止条件](/learning/llm-foundations-decoding-zh.svg)

## 模型给分数，解码器做选择

最后一个位置的 logits 是词表上的未归一化分数。Greedy 取最大值；随机采样先得到概率再抽样。温度使用 $p_i\propto\exp(z_i/T)$，通常 T 越低分布越集中。Greedy 是独立模式，代码不应直接用 T=0 做除法。Top-k 保留 k 个候选；top-p 按概率降序取最小前缀，使累计概率达到阈值，再重新归一化。

`[0.6,0.25,0.15]` 在 top-p=0.8 下保留前两个，总质量 0.85，新的概率约为 `[0.706,0.294]`。保留集合依赖当前步分布，不是固定词表子集。温度与截断的顺序也会影响结果。

## Prefill 与 decode 做了不同的工作

Prefill 一次处理已有 prompt，生成各层的 K/V；decode 每步只产生一个新位置的 Q/K/V，并读取缓存。单序列每层 KV 通常是 `[Hkv,T,Dh]`，batch 与 layout 因实现而异。缓存避免重算旧位置投影，但新 query 仍需要访问上下文，标准 dense attention 的每步读取量仍随 T 增长。

缓存不是答案缓存，也不是把所有历史 hidden states 都保存下来。修改旧 prompt、位置编号或模型权重后，旧 KV 不一定还能复用。分页管理和前缀共享属于系统层实现。

## 一个可验证的缓存实验

在 Notebook 中对同一 prompt 先跑完整前向，再取前缀的 `past_key_values`，仅把最后一个 token 输入第二次前向。比较最后位置的 logits，允许浮点误差。对带 padding 的真实 batch 还需要正确传 attention mask、position IDs 或 cache position；不要把单序列示例机械推广。

停止也属于解码器职责：EOS、最大新 token 数、stop string 与工具调用协议不是同一件事。记录是否因为长度截断停止，否则“答案不完整”可能是配置问题。

## 自测

**问题：** 改变温度后答案更正确，说明模型学到了新知识吗？

<details><summary>展开答案</summary>没有。权重不变，只是从已有分布选择输出的策略改变。需要多次采样、固定评估预算和正确性指标，才能判断这种变化是否稳定有益。</details>

## 原始资料与继续阅读

- [Transformers: GPT-2 model and outputs](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [The generation API](https://huggingface.co/docs/transformers/main_classes/text_generation)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
