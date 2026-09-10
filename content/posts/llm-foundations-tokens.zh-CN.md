---
title: "文字怎样变成模型能读的张量？"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "zh-CN"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-tokens"
slug: "llm-foundations-tokens-zh"
excerpt: "从词表索引、embedding 查表和 padding 入手，建立 B、T、D、V 四个维度的直觉。"
series: "llm-foundations"
seriesOrder: 1
draft: false
---

![文字怎样变成模型能读的张量？](/learning/llm-foundations-tokens-zh.svg)

## 从一句话开始

“猫在睡觉”不是模型直接接收的对象。Tokenizer 先把字符串映射成整数序列，再用 embedding 矩阵把整数变成向量。Token 可能是词、子词、字节片段或特殊符号；中文字符数、英文单词数与 token 数都不是固定比例。分词是离散预处理，embedding 才是可学习参数。

把词表想成一本有编号的字典，但不要把编号当大小关系。ID 120 和 121 相邻，并不意味着含义相似；训练改变的是对应向量的位置，而不是整数本身。不同模型的 tokenizer 和 embedding 必须配套。

## 跟着 shape 走一遍

设批大小 $B=2$，最长输入 $T=5$，隐藏维度 $D=8$，词表大小 $V=100$。输入 IDs 是 `[2,5]` 的整数，查表矩阵是 `[100,8]`，输出是 `[2,5,8]` 的浮点数。每个位置的向量随后会融合上下文，最初的查表向量和最后一层的上下文表示不是同一种东西。

```python
import torch
embedding = torch.nn.Embedding(100, 8)
ids = torch.tensor([[4, 12, 9, 0, 0], [6, 7, 8, 9, 10]])
mask = ids.ne(0)  # 本例约定 0 是 padding；真实模型读取 tokenizer 配置
x = embedding(ids)
assert x.shape == (2, 5, 8)
print(mask.sum(dim=1))  # tensor([3, 5])
```

## Padding 为什么会污染平均值？

补齐让不同长度样本能组成矩形张量，但补齐位置不是真实文本。假设三个真实位置的标量表示为 `[1,2,3]`，两个 padding 为 `[0,0]`，直接平均是 1.2，正确的 masked mean 是 2。真实 padding embedding 甚至不一定为零。Attention mask 告诉模型哪些位置有效，loss mask 则决定哪些位置参与监督，二者用途不同。

分类或检索任务常把 `[B,T,D]` 池化成 `[B,D]`；生成任务需要投影到 `[B,T,V]`。看到模型输出时先问：这是 token 表示、句子表示，还是词表 logits？不要仅凭最后一个维度很大就称它为 embedding。

## 动手与自测

打开本路线“真实模型输入输出”实验，输入同一句话的大小写、标点和中文版本，记录有效 token 数。再比较 padding 前后的归一化表示。训练语料和模型能力决定语义质量，分词成功不意味着理解成功。

**问题：** ID 全部加一，然后保持模型权重不变，是否只是重命名？

<details><summary>展开答案</summary>不是。模型会查到另一行 embedding。只有同时对 tokenizer 的映射与所有相关词表参数做一致置换，才可能保留行为。</details>

## 原始资料与继续阅读

- [Transformers: GPT-2 model and outputs](https://huggingface.co/docs/transformers/model_doc/gpt2)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
