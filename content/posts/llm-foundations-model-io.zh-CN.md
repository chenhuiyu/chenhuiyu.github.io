---
title: "真实模型实验：token、hidden states、attention 与 embedding"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "zh-CN"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-model-io"
slug: "llm-foundations-model-io-zh"
excerpt: "浏览器加载 MiniLM，Notebook 加载 GPT-2；用实际张量理解 encoder、decoder 和 pooling 的差别。"
series: "llm-foundations"
seriesOrder: 6
draft: false
---

![真实模型实验：token、hidden states、attention 与 embedding](/learning/llm-foundations-model-io-zh.svg)

## 先决定要观察什么

“加载模型”不是一个按钮就解释完的概念。本章拆成三层：tokenizer 把输入变成什么；forward 返回哪些张量；任务后处理怎样把张量变成有意义的结果。浏览器使用 `Xenova/all-MiniLM-L6-v2`，这是适合轻量语义相似度实验的 encoder。它不会自回归聊天。

下方按钮下载实际 ONNX 权重，在浏览器 CPU/WASM 中运行。不会用预置相似度替代失败结果。初次需要下载运行库和模型；访问失败或超时会明确显示错误。模型主要面向英文，中文界面不意味着模型具有同等中文检索能力。

## 从 token 表示到句子向量

模型先给出每个 token 的表示，再做带 mask 的 mean pooling，最后 L2 归一化。设有效 token 指示为 m，句向量为 $e=\sum_t m_th_t/\sum_t m_t$，然后除以 $\|e\|_2$。归一化向量的点积等于余弦相似度，取值范围是 -1 到 1，不是置信概率。

实验展示真实 token IDs、padding mask、输出 `[2,384]` 和前 32 个维度。单个 embedding 维度通常没有稳定可命名的语义；条形图用于观察数值，不要称作模型“关注哪里”。

## 更深入：打开 Notebook

配套 Notebook 加载 `distilgpt2` 的真实权重，输出 `[B,T,V]` logits、各层 hidden states 与 `[B,H,T,T]` attention，并验证缓存和完整前向的一致性。使用 eager attention 以显式返回权重；高性能 attention backend 不一定返回相同可观察对象。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained(
    'distilgpt2', attn_implementation='eager').eval()
inputs = tok('The cat is', return_tensors='pt')
# 完整 Notebook 包含 no_grad、可视化和缓存一致性检查。
```

## 三个值得真的尝试的反例

比较 “The cat is sleeping” 与近义改写；再把否定词加入第二句；最后换成领域术语或中文。词汇重叠高可能导致相似度仍然高，即使逻辑含义相反。把输入拉长超过 128 tokens，检查截断是否删除关键信息。实验记录应同时保存输入、有效长度与模型版本。

**问题：** 两个句向量余弦为 0.9，是否表示它们有 90% 概率互相蕴含？

<details><summary>展开答案</summary>不是。余弦衡量表示空间的几何接近程度。逻辑蕴含需要相应任务和验证集，不能从未经校准的相似度推导概率。</details>

## 本次真实运行结果

![实际 DistilGPT2 前向得到的第 0 层第 0 个 attention head。上三角为零，因果遮罩有效。](/learning/llm-foundations-model-io-actual.png)

实际 DistilGPT2 前向得到的第 0 层第 0 个 attention head。上三角为零，因果遮罩有效。

## 原始资料与继续阅读

- [MiniLM ONNX model](https://huggingface.co/Xenova/all-MiniLM-L6-v2)
- [Transformers.js 3.8.1](https://huggingface.co/docs/transformers.js/v3.8.1/pipelines)
- [Transformers: GPT-2 model and outputs](https://huggingface.co/docs/transformers/model_doc/gpt2)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
