---
title: "从预训练到对齐：模型到底在优化什么？"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "zh-CN"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-training"
slug: "llm-foundations-training-zh"
excerpt: "区分 next-token loss、SFT、LoRA 和偏好优化，动手检查 shifted labels 与有效监督位置。"
series: "llm-foundations"
seriesOrder: 4
draft: false
---

![从预训练到对齐：模型到底在优化什么？](/learning/llm-foundations-training-zh.svg)

## 预测下一个 token，为什么能并行训练？

自回归分解是 $p(x)=\prod_t p(x_t\mid x_{<t})$。训练时完整文本已经存在，因果遮罩阻止位置 t 看未来，所以一次前向可以同时计算多个位置的 next-token loss。推理时未来 token 尚未产生，才必须逐步生成。训练并行和生成串行并不矛盾。

输入 `[A,B,C]` 对应目标 `[B,C,D]`。交叉熵是正确目标的负对数概率，loss 较低只说明对数据分布拟合更好，不直接证明事实正确、指令遵循或推理可靠。

## SFT 改变数据，LoRA 改变可训练参数

SFT 通常用指令与答案组成序列，可只在 assistant tokens 上计算 loss。Padding、用户问题与系统提示是否参与监督，要明确记录。LoRA 用低秩增量 $\Delta W=BA$ 更新一部分线性层，是参数化方法，不是另一种任务目标；SFT 和 LoRA 可以同时使用。

若原矩阵是 `[d_out,d_in]`，rank r 的增量参数量是 $r(d_{in}+d_{out})$。节约可训练参数和优化器状态，不代表原始模型权重无需加载，也不代表激活内存消失。

```python
import torch
import torch.nn.functional as F
logits = torch.randn(1, 3, 5, requires_grad=True)
labels = torch.tensor([[1, 2, -100]])
loss = F.cross_entropy(logits.reshape(-1, 5), labels.reshape(-1),
                       ignore_index=-100)
loss.backward()
assert torch.all(logits.grad[0, 2] == 0)
```

此例 logits 已与 labels 对齐；调用 Hugging Face causal LM 的内置 loss 时通常传未 shift 的 labels，让模型内部处理，不能重复 shift。

## 偏好优化增加什么信号？

DPO 比较同一 prompt 下的 chosen/rejected 答案，相对于固定 reference policy 调整对数概率比。它依赖偏好数据质量与适用假设，不等同于“无奖励函数的万能 RL”。RLHF / RLVR 则需要进一步区分学习的奖励模型与可程序验证的奖励。奖励变高可能来自利用评分漏洞，必须用独立任务验证。

## 自测与诊断

**问题：** 训练 loss 一直下降，验证集却变差，该先加训练步数吗？

<details><summary>展开答案</summary>先检查过拟合、重复样本、标签错位、训练/验证模板差异与泄漏。增加步数可能扩大问题。固定小批样本做 overfit sanity check，并在独立验证集比较，是不同目的的两项实验。</details>

## 原始资料与继续阅读

- [Transformers: GPT-2 model and outputs](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [DPO](https://arxiv.org/abs/2305.18290)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
