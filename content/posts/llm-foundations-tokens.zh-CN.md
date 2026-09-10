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

## 从零开始的完整讲解

### 先把“模型读文字”拆成三个动作

假设你输入“猫坐在垫子上”。电脑保存的是编码后的字节，模型做的是数字运算，中间需要一个明确的转换接口。第一步 tokenizer 把字节或字符串切成词表允许的片段；第二步把片段映射为整数 ID；第三步从 embedding 表里按 ID 取出向量。**片段、编号、向量是三个不同对象**。编号 100 比编号 20 大，不表示它更重要、更接近某个意思，编号只相当于查字典的页码。

向量可以先理解成一列数，矩阵是一张数字表，张量是允许更多轴的数字容器。比如 `[2, 5, 8]` 不是三个 token ID，而可能是一个张量的 shape：2 句话、每句补齐到 5 个位置、每个位置 8 个数。看到 shape 时先写出每根轴的名字，比立即背矩阵公式更有用。

### 一个能手算的 embedding lookup

设词表为 `['<pad>', '猫', '坐', '垫子']`，embedding 表有 4 行、每行 3 个数。输入“猫 坐”在这个人为分词规则下变成 `[1,2]`。模型执行 `E[[1,2]]`，取出第 1、2 行，得到 `[2,3]` 的矩阵。这里的第一个 2 表示两个位置，不是 batch size。加入一条样本的 batch 轴后，形状才是 `[1,2,3]`。

```python
vocab = {'<pad>': 0, '猫': 1, '坐': 2, '垫子': 3}
E = [[0,0,0], [0.2,0.5,-0.1], [0.8,-0.2,0.3], [-0.1,0.4,0.9]]
ids = [vocab[word] for word in ['猫','坐']]
x = [E[i] for i in ids]
print(ids, x)
```

这是可直接复制到下方 Notebook 执行的标准 Python。真实模型的词表可能很大、向量可能很长，但查表关系完全一样。Embedding 在训练中会更新；初始向量不是人工写好的词义描述。相同 ID 的初始 embedding 相同，但经过上下文层后，“苹果手机”和“吃苹果”中的表示可以不同。

### 为什么不能简单按空格分词

中文没有天然空格边界，英文又存在复合词、拼写变化、数字与程序代码。子词分词在词表大小和序列长度之间折中：常见片段使用一个 ID，罕见字符串拆得更细。BPE 的基本思想是从较小单元出发，根据训练语料中的统计逐步合并；实际 tokenizer 还可能包含预切分、字节映射和特殊符号规则。

因此不能假定“一个汉字等于一个 token”，也不能拿模型 A 的 token ID 给模型 B 使用。模型与 tokenizer 是共同保存的接口契约。对英文训练较多的分词器，同样语义的中文可能需要更多 token，这会影响上下文预算和延迟，但不能仅据此判断模型中文能力。

### Padding 与 truncation：形状正确不代表内容正确

两句话长度分别为 3、5，组成普通稠密 batch 后通常补齐到 `[2,5]`。短句的两个补位不是用户真正输入的内容，attention mask 需要将其排除；训练时 loss mask 也要避免把补位当成预测目标。这两种 mask 负责不同事情，不能互相替代。

Truncation 是直接丢弃超过上限的 token。如果文档的重要结论在尾部，程序依然能正常返回张量，却已经失去了答案依据。调试时同时打印原始字符串、ID、反解文本、mask 和截断后的长度。只看“没有报错”不能发现这种语义错误。

### 把本章知识用于实验

下方浏览器 Notebook 的第一格故意用字符词表，使每一步都能检查。把文本中的一个字改掉，再运行第一格，你会看到词表及 ID 可能改变；因此后续模型参数也必须重新初始化。它不是预训练模型 tokenizer 的替代品。真实 tokenizer 与权重配套加载的实验在“模型输入输出”一章。

检查自己是否真正理解：如果词表大小是 10,000，hidden dimension 是 256，embedding 有多少参数？答案是 2,560,000，和这一次输入多少个 token 无关。序列长度改变的是查表结果与后续计算规模，不会临时新增模型参数。

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
