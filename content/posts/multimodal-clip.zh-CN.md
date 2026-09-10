---
title: "CLIP：让图像和文字在同一空间相遇"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "zh-CN"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-clip"
slug: "multimodal-clip-zh"
excerpt: "亲手构造 B×B 相似度矩阵，理解对比学习、负样本、温度和零样本分类的局限。"
series: "multimodal"
seriesOrder: 2
draft: false
---

![CLIP：让图像和文字在同一空间相遇](/learning/multimodal-clip-zh.svg)

## 两个 encoder，一次配对任务

图像 encoder 与文本 encoder 各输出向量，投影到同一维度并归一化。一个 batch 有 B 个图文配对，矩阵 $S_{ij}=u_i^\top v_j/\tau$ 是 `[B,B]`。对角线是匹配对，其余位置通常作为 batch negatives。图找文与文找图分别做交叉熵，平均成对称目标。

一个重要陷阱：非对角线不一定语义错误。两张不同猫图可能都匹配“a cat”。False negatives 和重复 caption 会影响学习；batch 组织与数据清理是目标的一部分，不是只要公式写对就完成。

## 温度改变的是竞争强度

温度小会放大相似度差异，使分布更尖锐，梯度也随之变化。把全部相似度都乘以一个很大的数，训练可能更容易过度自信。训练中可学习的 logit scale 与推理时人为改阈值不是同一个动作。

```python
import torch
import torch.nn.functional as F
image = F.normalize(torch.randn(4, 8), dim=-1)
text = F.normalize(torch.randn(4, 8), dim=-1)
s = image @ text.T / .07
y = torch.arange(4)
loss = (F.cross_entropy(s,y)+F.cross_entropy(s.T,y))/2
print(loss.item())
```

随机向量只能检查 shape 和 loss 实现。配套 Notebook 加载真实 CLIP，并对自己提供的图片和候选描述运行前向。

## 零样本分类为什么依赖候选集合？

把“a photo of a cat”等类别描述编码，与图像比较，softmax 后排序。概率是在这些候选标签之间归一化的相对得分；即使图片根本不属于任何一个类别，也会给出最大值。增加一个相似标签会改变归一化分母。Prompt wording、语言和类别粒度也影响排序。

内容理解还包括关系、数数、OCR、时间顺序与细粒度属性，整体图文相似并不能证明这些能力。应使用反事实描述，例如“人推车”与“车撞人”，检查是否仅靠对象词匹配。

## 自测

**问题：** 只有一个候选标签时，softmax 得 1，意味着模型确信吗？

<details><summary>展开答案</summary>不是。单元素 softmax 永远是 1。需要原始分数、合适的拒识方法和独立验证集来判断未知类别。</details>

## 原始资料与继续阅读

- [CLIP](https://arxiv.org/abs/2103.00020)
- [Transformers CLIP](https://huggingface.co/docs/transformers/model_doc/clip)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
