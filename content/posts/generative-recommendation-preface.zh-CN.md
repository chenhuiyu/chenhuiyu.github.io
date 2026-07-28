---
title: "生成式推荐到底在生成什么？"
date: "2026-07-24"
updated: "2026-07-24"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - LLM
pairKey: "generative-recommendation-preface"
slug: "generative-recommendation-preface-zh"
excerpt: "在进入 20 篇论文之前，先分清推荐系统究竟在生成文本、商品 ID、整张列表，还是一个看起来像推理的过程。"
series: "generative-recommendation"
seriesOrder: 0
draft: false
---

推荐系统原本并不会“回答问题”。

它更像一台安静的打分机：给定用户 $u$ 和商品 $i$，学习一个分数
$s(u,i)$，再把分数最高的商品排在前面。

$$
\hat{i} = \arg\max_i s(u,i)
$$

今天的生成式推荐却开始做一件看起来完全不同的事：它可以逐 token
生成商品 ID、一次写出整张推荐列表、解释为什么推荐，甚至在输出答案前先生成一段“思考”。

这个系列只追问一个问题：

> 推荐系统从“给所有商品打分”，是怎样一步步变成“直接说出答案”的？

## 同名但不同宗

VAE、GAN 和 Diffusion 很早就被用于推荐，也常被称为“生成式推荐”。它们会生成潜变量、用户表示或数据分布，但并不是这套故事的直接主线。

这里关注的是另一条更具体的路线：**把推荐重新表述成一个自回归生成问题**。模型不再只计算候选商品的分数，而是学习
$P(\text{output tokens}\mid\text{user history})$。

真正关键的区别不是模型能不能生成，而是它究竟生成什么：

- **语言答案**：评分、解释、偏好判断都写成 text-to-text。
- **商品标识**：把 Item ID 编码为一串可解码 token。
- **完整列表**：后一个推荐结果可以依赖前面已经生成的结果。
- **推荐理由与推理过程**：模型不仅输出“选什么”，还尝试表达“为什么”。

## 故事从一台打分机开始

前四篇是这场变化的“前世”。

BPR 建立隐式反馈下的 pairwise 排序范式；GRU4Rec 发现用户不是一个静态向量，而是一段按顺序发生的行为；SASRec 用注意力寻找真正相关的历史；BERT4Rec 再把 Item ID 当成 token，把行为序列当成句子。

到这里，推荐系统虽然还没有真正“生成商品”，但语言模型式的表示方法已经进入了推荐。

## 两条支线在这里分开

第一条是 **LLM 支线**。

P5 尝试把不同推荐任务统一成 text-to-text；M6-Rec 把这种统一推向工业 foundation model；TALLRec、ReLLa 和 LLaRA 则逐步暴露问题：通用 LLM 会说话，不代表它天然理解协同偏好；上下文放得下，也不代表模型知道哪些历史真正相关。

第二条是 **Semantic ID 与生成式检索支线**。

DSI 先证明检索可以不查索引，而是直接生成文档 ID。TIGER、GPTRec、LC-Rec、LETTER 和 ETEGRec 随后把问题推进到推荐系统最核心的部件：怎样给每件商品设计一串既有语义、又懂协同行为、还不会发生 codebook 塌缩的 token。

## 最后，它们在工业系统里汇合

HSTU 把超长用户行为重写为 sequential transduction；OneRec 尝试用一个模型统一召回与排序；MTGR 提醒我们，漂亮的生成范式仍要面对传统交叉特征、延迟和成本；OneRec-Think 与 OneReason 则把问题推向更危险也更有趣的地方：

> 推荐模型看起来会思考，是否真的意味着它理解了偏好？

这不是一个已经有结论的问题，也正是整个系列最后要留下的开放问题。

## 我们会怎样读这 20 篇论文

每篇文章都分三层：

1. **零基础层**：用同一个玩具用户和八件商品解释上一代模型遇到的困难。
2. **工程层**：明确输入输出、tensor shape、训练数据、推理路径和复杂度。
3. **论文层**：只保留一个核心公式、一张关键实验、一个重要 ablation，以及论文没有证明什么。

固定的玩具行为序列是：

```text
网球 → 攀岩 → 羽毛球 → 游泳
```

读者会看到同一份历史如何依次被打分模型、RNN、Transformer、LLM、Semantic ID tokenizer 和生成式推荐模型处理。

下一篇，我们回到一切开始的地方：**BPR，以及推荐系统为什么最初只需要学会“谁应该排在谁前面”。**

[查看完整的 20 篇路线图 →](/series/generative-recommendation)
