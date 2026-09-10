---
title: "视觉语言模型：从 CLIP 特征到能回答问题的 LLM"
date: "2026-09-10"
updated: "2026-09-10"
category: "Multimodal Understanding"
language: "zh-CN"
tags: ["multimodal", "Hands-on", "Model Learning"]
pairKey: "multimodal-vlm"
slug: "multimodal-vlm-zh"
excerpt: "理解 vision encoder、projector 与 decoder 的接口，分清对齐训练、指令训练和视觉 grounding。"
series: "multimodal"
seriesOrder: 4
draft: false
---

![视觉语言模型：从 CLIP 特征到能回答问题的 LLM](/learning/multimodal-vlm-zh.svg)

## 从零开始的完整讲解

### 图像编码器与语言模型之间缺一个接口

语言模型习惯接收 `[B,T,d_text]` 的 token 表示，视觉编码器可能输出 `[B,N,d_vision]` 的 patch 特征。N 与 T 的含义不同，两个 hidden dimension 也未必一致。连接器或 projector 将视觉特征变换到语言模型可接收的表示空间，或者通过 cross-attention 建立交互。

投影维度对齐只是接口成功，不等于语义对齐已经完成。就像把文件改成正确扩展名不代表内容符合协议，模型仍需通过适当训练学会利用视觉信息回答问题。

### 用一个具体输入追踪视觉 token

假设视觉侧产生 196 个 patch，每个 768 维，projector 输出 196 个 4096 维向量，再与 30 个文本 token 一起送入某种 early-fusion 布局。仅看这个示例，输入序列就可能超过 226 个位置，因为还可能有特殊分隔符。

并非所有 VLM 都按这个布局工作：有的使用 resampler 压缩 token，有的使用 cross-attention，有的分多个裁剪或动态分辨率。不能把一张图固定说成“等于 196 个 token”。先检查具体模型处理后的数量，再计算上下文和 KV 预算。

### Alignment 训练与 instruction tuning 各解决什么

训练连接器可以让语言侧更容易接收视觉表示，视觉指令数据则教模型在问题条件下使用图像。是否冻结视觉 encoder、LLM 或某些模块，会影响成本与适应能力。少量可训练参数不意味着系统不需要加载被冻结的部分。

如果训练文本经常泄露答案，模型可能学会主要靠语言先验作答。验证视觉依赖可以把图片替换成不相关图片、打乱图文配对，观察答案是否仍然不变。若不变，需进一步判断是题目本来不需要图片，还是模型忽略视觉。

### 看得见、读得懂与答得对是三个关卡

问“收据总额是多少”，首先需要保留小字的分辨率，其次需要识别数字和版面关系，最后要区分小计、税费与总额。提升 LLM 规模可能无法弥补预处理已经删掉的区域。

问“左边的人拿着什么”，则需要对象、空间位置与属性绑定。全图语义接近不够，必须把问题中的“左边”绑定到正确区域。解释错误时把这些关卡分开，能帮助选择提高分辨率、调整视觉编码或改进训练数据。

### 幻觉为什么会出现得很流畅

语言模型擅长生成符合上下文统计的句子。视觉证据弱、模糊或被压缩时，常识先验可能填补空缺，产生不存在的物体、文字或动作。描述越自然不代表证据越可靠。

构建测试时加入不存在对象的问题、模糊图片、与图像冲突的文本诱导，并允许“无法判断”。对 OCR 和数值问题用明确答案核验，对开放描述则检查可见证据与过度推断。单纯要求模型“不要幻觉”并不能保证解决问题。

### 实验如何延伸到内容理解

从真实图像预处理与 CLIP/MAE 输出开始，理解视觉特征接口；再在可支持 VLM 的环境检查 projector 后 shape、视觉 token 数和生成结果。站内 Notebook 用较小实验建立基础，完整 VLM 的 GPU 推理需要相应资源。

将同一图像用于检索、分类、问答和推荐内容特征，可以发现目标之间的区别：适合全局检索的向量未必适合细粒度计数，能生成描述也不意味着向量适合近邻搜索。训练目标应与最终用途相匹配。

## 两种模型，输出目标不同

CLIP 通常给匹配分数；视觉语言生成模型需要根据图像与问题输出文本。一个常见结构是 vision encoder 提取 `[B,N,Dv]`，projector 映射到语言 hidden dimension Dl，视觉表示再按模型协议插入语言序列。Projector 解决接口维度与表示匹配的一部分问题，不是简单“翻译图片成句子”。

具体模型可能采用 cross-attention、resampler 或视觉 token 合并，不能把所有 VLM 都简化为单个线性层。图像 token、special tokens 和位置编码必须与 processor/chat template 配套。

## 训练通常不止一个阶段

对齐阶段可以冻结某些 backbone，只训练桥接模块；视觉指令训练用图像、问题、回答监督生成。哪些模块更新依论文与 recipe 而定。模型会利用语言先验：如果训练问题里“香蕉是什么颜色”的答案几乎都是黄色，它可能不看图也答对。因此需要设计图像改变但问题相同的对照。

```python
import torch
visual = torch.randn(1, 196, 768)
projector = torch.nn.Linear(768, 1024)
visual_tokens = projector(visual)
text_tokens = torch.randn(1, 12, 1024)
combined = torch.cat([visual_tokens, text_tokens], dim=1)
assert combined.shape == (1, 208, 1024)
```

这段只演示 shape 接口，不是完整 VLM：缺 processor、模态标记、position/mask 以及预训练对齐。

## Grounding 要有空间证据

“图里有猫”与“猫在红色沙发左侧，框坐标是多少”是不同能力。坐标还可能用像素、归一化 0–1 或离散 bins，resize/crop 后需要映射回原图。OCR 要保留足够分辨率；细粒度属性需要局部证据。语言流畅不能代替这些检查。

读 Qwen2.5-VL 等报告时，重点看动态分辨率、视觉 token 预算和视频时间表示如何连接任务。不要把报告中的最大上下文理解为任何设备都能高效运行。

## 自测

**问题：** 遮住图像后准确率不变，说明视觉 encoder 很鲁棒吗？

<details><summary>展开答案</summary>更应怀疑模型利用问题或数据集先验。比较图片打乱、空图、问题-only 与反事实图像，验证答案是否依赖视觉证据。</details>

## 原始资料与继续阅读

- [LLaVA / Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- [Qwen2.5-VL report](https://arxiv.org/abs/2502.13923)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
