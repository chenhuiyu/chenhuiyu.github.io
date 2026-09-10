---
title: "Embedding、RAG 与评估：会检索不等于会回答"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "zh-CN"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-rag-eval"
slug: "llm-foundations-rag-eval-zh"
excerpt: "把检索召回、重排、上下文构造和答案归因分开测量，用失败类型推动系统改进。"
series: "llm-foundations"
seriesOrder: 7
draft: false
---

![Embedding、RAG 与评估：会检索不等于会回答](/learning/llm-foundations-rag-eval-zh.svg)

## 一个完整 RAG 请求

用户问“这个服务的超时是多少？”系统先把问题转成向量，从文档块里召回候选，再用 reranker 判断相关性，把选中文本交给生成模型，最后给出有出处的答案。不同环节可能失败：文档没收录、切块割裂条件、召回偏离、重排遗漏、上下文截断，或生成模型忽略证据。

把所有失败都称为 hallucination 会让修复失去方向。先记录每一步输入输出，再问正确证据首次在哪里消失。RAG 的目标是给模型提供可更新依据，而不是保证一切回答正确。

## 向量检索的形状与边界

文档矩阵为 `[N,D]`，query 为 `[B,D]`，点积得 `[B,N]`。归一化后点积等于 cosine，但 index 的距离度量必须与训练和归一化方式一致。ANN 用近似搜索换延迟与内存，Recall@k 还受索引参数影响。训练 embedding 用的相似性目标可能和实际任务不一致。

```python
import torch
import torch.nn.functional as F
docs = F.normalize(torch.tensor([[1.,0.],[0.,1.],[1.,1.]]), dim=-1)
query = F.normalize(torch.tensor([[.9,.1]]), dim=-1)
scores = query @ docs.T
print(scores.topk(2, dim=-1).indices)
```

这是检索几何示例，不是真实语义模型。真实 embedding 请使用前一章。

## 给每层一个可失败的指标

| 层 | 指标 | 典型诊断 |
|---|---|---|
| 语料 | 证据覆盖率 | 答案是否存在于快照中 |
| 召回 | Recall@k | 正确块能否进入候选 |
| 重排 | MRR / NDCG | 正确块是否排在前面 |
| 生成 | 正确性、引用支持率 | 引用是否真的支持该主张 |
| 服务 | p95 延迟、失败率 | 用户能否稳定拿到结果 |

比较系统时固定语料快照、问题集、token 预算与延迟约束。不能把更多上下文带来的收益归因于一个新检索器。LLM judge 应校准到人工样本，并检查位置偏差、长度偏好及与被测模型的相关错误。

## 自测

**问题：** Recall@20 上升，但最终正确率下降，可能吗？

<details><summary>展开答案</summary>可能。更多噪声可能挤掉上下文预算、放大冲突或让生成器选错证据。还可能是重排或 prompt 变化。按阶段记录命中与失败，才能定位。</details>

## 原始资料与继续阅读

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [BEIR](https://arxiv.org/abs/2104.08663)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
