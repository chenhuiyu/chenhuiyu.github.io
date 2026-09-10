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

## 从零开始的完整讲解

### 模型怎样知道自己“错了”

训练开始前，模型只是一组待调整的数字。我们给它输入，得到对下一个 token 的概率，再用真实文本中的下一个 token 当标签。假设正确答案是“猫”，模型给三个候选的概率为 `[0.1,0.7,0.2]`，而“猫”对应第二项，单个位置的损失为 `-log(0.7)`，约 0.357。若它只给正确答案 0.1，损失约 2.303，惩罚更大。

Loss 是一个可优化的标量，不是“智能分数”。它告诉优化器在指定训练目标下怎样移动参数。文本预测做得更好可以带来很多能力，但不意味着每个答案都真实，也不意味着模型满足所有用户偏好。

### 从概率推到最有用的一条梯度

设 logits 为 z，softmax 概率为 p，标签的 one-hot 向量为 y。交叉熵对每个 logit 的梯度为 `p-y`。如果正确类别当前概率 0.7，该项梯度就是 -0.3；梯度下降会提高这个 logit。错误类别概率越大，其正梯度越大，更新会压低它。参数的梯度再通过链式法则从 logits 向前传播。

下方浏览器 Notebook 训练 bigram 语言模型，明确写出了 `grad[source][j] += p[j]-(j==target)`。这是实际梯度更新，不是预先画好的下降曲线。它只学相邻字符关系，但足以观察学习率、训练轮数与 loss 的联系。

### 标签如何错开一位

输入序列“我 爱 猫”用于预测“爱 猫 EOS”。在一次前向中，每个可监督位置都能产生预测；因果遮罩保证它不能看到未来。框架可能在模型内部移动 labels，也可能要求调用者自己构造，不能同时移动两次。

SFT 中，用户问题和系统提示往往只是条件，主要对 assistant 的回答计算 loss。若把用户部分也作为目标，训练优化的问题已经改变。将标签设为 ignore index 可以屏蔽某些位置的 loss，但不会自动阻止其他位置通过 attention 读取它们。

### Batch、梯度累积与学习率

一次更新可以使用多个样本。计算各样本梯度后求平均，再更新参数，能降低单个样本带来的波动。显存放不下大 batch 时，可以分几个 microbatch 前向、反向，累积梯度后统一 step。若每个 microbatch 的 loss 已按自身 token 数求平均，长度差异很大时简单再平均，会改变每个 token 的权重；应该明确采用按样本还是按有效 token 归一化。

学习率太大可能让 loss 震荡或发散，太小则进展缓慢。不要把所有 loss 下降都当成好事：训练集下降但验证集上升，是过拟合的常见信号。数据泄漏会让验证集也显得很好，因此拆分规则是训练流程的一部分。

### 预训练、SFT、LoRA 与 DPO 不在同一分类轴上

预训练和 SFT 描述数据及目标的阶段；LoRA 描述更新参数的方式。它冻结原权重，通过低秩增量调整某些线性层，因此也可以用于 SFT。若原矩阵为 `[4096,4096]`，rank 8 的两矩阵合计 `4096*8+8*4096=65,536` 参数，远小于原来的 16,777,216。可训练参数少不意味着前向不需要原模型权重。

DPO 使用同一 prompt 下的偏好对，通过策略模型与参考模型的相对 log probability 来优化偏好。它不等于“把好答案再做一次普通 SFT”，也不保证学到的偏好覆盖所有场景。比较不同方法时，需要控制数据来源、训练预算和评测协议，不能把数据改进全部归因于算法。

### 最小可复现实验应该留下什么

保存 tokenizer、模型配置、随机种子、数据版本、拆分方式、优化器和学习率设置。评估时除了平均 loss，还要看目标任务正确率、不同长度与语言子集、输出格式和失效案例。能够解释“这个 loss 如何从这些标签算出来”，是比背诵训练名词更扎实的起点。

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
