---
title: "图像怎样变成 token：Patch、位置与分辨率"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "zh-CN"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-patches"
slug: "multimodal-patches-zh"
excerpt: "从 RGB 张量到 ViT patch embedding，计算视觉 token 数，理解缩放与裁剪可能丢失的内容。"
series: "multimodal"
seriesOrder: 1
draft: false
---

![图像怎样变成 token：Patch、位置与分辨率](/learning/multimodal-patches-zh.svg)

## 从零开始的完整讲解

### 图像对模型而言，首先是一组数字

一张 RGB 图片有高、宽和颜色通道。若尺寸为 224×224，每个像素有红绿蓝三个数，常见输入形状是 `[B,3,224,224]`，也有库使用通道在最后的布局。不能仅凭四维就断定轴含义，要看预处理接口。图像显示正常，不意味着送入模型的通道顺序、范围和归一化都正确。

像素经常从整数 0～255 转为浮点，再按 checkpoint 的均值和标准差处理。不同模型的 preprocessing 是模型接口的一部分，不能随意套用另一个模型的 resize、crop 或 normalization。

### 从图片切出一张 token 表

取 patch 边长 16，将 224×224 划成 14×14 个小块，共 196 块。每块含 `16*16*3=768` 个数；展平后是 `[196,768]`，再乘投影矩阵得到 `[196,d]`。加 batch 轴后是 `[B,196,d]`，这时才与文本 token 的表示形式相近。

Patch embedding 可以通过等价的卷积操作实现，不一定真的在 Python 中逐块切开。展平顺序必须稳定，否则训练权重看到的像素位置会错乱。下方 Notebook 实现 patchify 和 unpatchify，验证还原后的像素与原输入逐项相等。

### Patch 大小改变的是信息与计算的折中

同样 224 分辨率，patch 16 得到 196 个 token，patch 8 得到 784 个，是 4 倍。若后面使用全连接式稠密 attention，分数元素可能增加到 16 倍。更细 patch 可能保留小字和边缘细节，但也增加计算与存储。

分辨率翻倍也会产生类似的二维增长。不要把“从 224 到 448 只是两倍”当作 token 只增加两倍：高宽同时翻倍，面积增加四倍。动态分辨率模型可能使用不同机制，这个例子是固定网格 ViT 的账本。

### 位置信息使小块成为一幅画

将图像块随意打乱后，颜色与局部纹理还在，但物体结构可能消失。位置编码告诉模型 patch 在二维布局中的位置。某些模型还加一个 class token 汇总全图信息；它不是原图中的一个额外像素块。

当输入网格改变时，位置表示如何适配也需要检查。插值位置 embedding 可以让形状兼容，却不自动保证模型在新尺度上表现良好。训练分布和实际目标仍然重要。

### 预处理如何悄悄丢掉答案

商品图边缘的标签、小票最下方的总额、截图角落的按钮，都可能在 center crop 后消失。模型回答错误时，先显示它实际接收的 resize/crop 结果，再讨论架构能力。OCR 失败也可能来自文字缩得太小，而不是模型不会识字。

长宽比很大的图像需要特别关注。强行压成正方形可能扭曲文字，裁剪则可能丢内容。选择填充、分块或多尺度方案，需要同时考虑任务与视觉 token 预算。

### 在 Notebook 里建立最小检查闭环

先在 8×8 灰度图上检查分块次序与还原，再将 patch 从 2 改为 4，观察 token 数从 16 变为 4。输入维度变化是确定的，后续质量变化则需要训练或评估，不能从 shape 单独推断。

完整多模态 Notebook 进一步展示真实 CLIP 和 MAE 预处理与输出。将原图、模型输入、token 数与最终任务输出放在同一次记录中，能快速发现“模型根本没有看到关键区域”的错误。

## 图像不是一长串像素文字

视觉 encoder 常把图片分成固定大小 patch，每块展平后线性投影成 D 维向量。设输入 `[B,3,H,W]`，patch 边长 P，整除时 token 数 $N=(H/P)(W/P)$，每块原始维度为 $3P^2$。这是可学习的线性变换，不是 tokenizer 按词表查找离散 ID。某些架构添加 CLS token，另一些使用 pooling 或压缩，需读具体实现。

224×224 图像配 16×16 patch 得到 196 个 patch，若有 CLS 则是 197。分辨率变成 448×448，patch 数是 784，dense attention 的 pair 数约变为原来的 16 倍，不只是图像边长的 2 倍。

## 一段能运行的 patchify

```python
import torch
image = torch.arange(3*32*32).reshape(1,3,32,32).float()
p = 8
patches = image.unfold(2,p,p).unfold(3,p,p)
patches = patches.permute(0,2,3,1,4,5).reshape(1,16,3*p*p)
projection = torch.nn.Linear(3*p*p, 64)
print(projection(patches).shape)  # [1,16,64]
```

`permute` 的顺序决定像素排列，重建时必须用一致的逆变换。Tensor shape 正确并不能保证颜色通道和空间位置正确。Notebook 会把 patchify / unpatchify 往返结果与原图比较。

## 预处理是模型定义的一部分

Resize、crop、RGB conversion、归一化 mean/std 都会影响结果。把已经除以 255 的浮点图再次交给默认 rescale processor，可能重复缩放。把长图裁成方形则可能直接丢失边缘文字；生成器回答错误前，先检查送入 encoder 的图。

动态分辨率模型可能按长宽比例生成不同视觉 token 数，视频还引入时间维度。不要把固定 ViT 的 196-token 经验搬到所有 VLM。

## 自测

**问题：** 把 patch 边长从 16 改成 8，可以直接沿用原 patch projection 权重吗？

<details><summary>展开答案</summary>通常不行。展平维度从 768 变为 192，权重 shape 已改变；位置表示与训练分布也会受影响。架构适配需要明确方法，而非只改配置整数。</details>

## 原始资料与继续阅读

- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Image processors](https://huggingface.co/docs/transformers/main_classes/image_processor)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
