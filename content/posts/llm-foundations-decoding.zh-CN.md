---
title: "从 logits 到一句话：采样、KV cache 与停止条件"
date: "2026-09-10"
updated: "2026-09-10"
category: "LLM Foundations"
language: "zh-CN"
tags: ["llm-foundations", "Hands-on", "Model Learning"]
pairKey: "llm-foundations-decoding"
slug: "llm-foundations-decoding-zh"
excerpt: "亲手计算 softmax、温度与 top-p，分清生成策略和模型分布，也理解缓存如何避免重复计算。"
series: "llm-foundations"
seriesOrder: 5
draft: false
---

![从 logits 到一句话：采样、KV cache 与停止条件](/learning/llm-foundations-decoding-zh.svg)

## 从零开始的完整讲解

### 一次 forward 为什么还不是一句回答

语言模型的一次前向得到每个位置对词表的分数。生成时，我们通常取最后一个有效位置的 logits，决定下一个 token，再把这个 token 加入上下文，继续执行下一步。回答 100 个 token 通常涉及连续的生成决策；不能把一个 `[B,T,V]` 张量误认为模型已经同时生成了 T 个未来 token。

Logit 可以为负，也不需要加起来等于 1。Softmax 才把它转为分布。模型、解码器和停止规则共同决定可见文本：相同权重配不同采样设置，输出可以不同。因此复现实验时只写 checkpoint 名称不够。

### 温度到底改变了什么

设两个候选 logits 为 `[2,1]`。温度为 1 时，概率约为 `[0.731,0.269]`；温度为 0.5 时相当于计算 `[4,2]` 的 softmax，概率约为 `[0.881,0.119]`；温度为 2 时概率约为 `[0.622,0.378]`。降低温度使分布更尖锐，但没有添加新的事实知识。

Greedy 直接选最大项。Top-k 只保留得分最高的 k 个候选；top-p 根据累计概率保留一组候选，然后重新归一化。温度和过滤的先后次序会影响结果，实际库中的顺序要核对。把温度直接设为零会造成除零，API 通常用独立的 greedy 设置表达确定性选择。

### KV cache 是缓存历史计算，不是缓存答案

在因果模型中，过去 token 的表示不应受后来生成的 token 影响。已经算好的历史 K、V 因此可以重复使用。第一次 prefill 处理整个 prompt，随后 decode 只为新增 token 计算新的 Q、K、V，并让新 query 读取累计的历史 KV。

缓存通常按层保存，因此不能只计算一层的开销。以 32 层、8 个 KV heads、每头维度 128、每元素 2 bytes 为例，每个 token 的 KV 为 `2*32*8*128*2=131072` bytes，即 128 KiB。单条 4,096-token 序列占约 512 MiB；8 条同长度序列约 4 GiB，不含权重、临时激活和分配器开销。

GQA/MQA 减少的是 KV head 数，不是简单缩短生成文本。缓存省掉历史投影与重复前向，却仍需要读取历史 KV；上下文增长后 decode 的读取负担仍然增加。

### 如何验证 cache 没有改变语义

最直接的对照是同一前缀走两条路径：一条完整重算，另一条用缓存增量执行。比较下一 token 的 logits，允许合理浮点误差。若差异很大，先看位置编号、attention mask、cache 长度，以及是否误把最后一个 token 再输入了一次。

训练模式下 dropout 会扰乱比较，先切换 eval；不同 attention 内核也可能造成小数值差别。不要只比较最终生成文本：轻微 logits 差异也可能在接近并列时导致不同采样路径，而相同文本也不证明中间实现完全正确。

### 停止条件属于产品语义

EOS 是模型词表中的结束符，最大新增 token 数是计算预算。它们不是同一件事。输出因长度上限中断时，JSON、代码或引用可能尚未闭合。按字符串停止还需要考虑 token 边界以及 stop string 是否应展示给用户。

下方 Notebook 在实际训练的 bigram 模型上采样。固定随机种子，分别设温度 0.3、1、2，记录重复率与连贯性。这个模型没有长程理解，出现重复不等于大模型一定有同样行为；它让你观察采样机制本身。完整 PyTorch 实验则检查真正自回归模型的 KV 与 logits。

## 模型给分数，解码器做选择

最后一个位置的 logits 是词表上的未归一化分数。Greedy 取最大值；随机采样先得到概率再抽样。温度使用 $p_i\propto\exp(z_i/T)$，通常 T 越低分布越集中。Greedy 是独立模式，代码不应直接用 T=0 做除法。Top-k 保留 k 个候选；top-p 按概率降序取最小前缀，使累计概率达到阈值，再重新归一化。

`[0.6,0.25,0.15]` 在 top-p=0.8 下保留前两个，总质量 0.85，新的概率约为 `[0.706,0.294]`。保留集合依赖当前步分布，不是固定词表子集。温度与截断的顺序也会影响结果。

## Prefill 与 decode 做了不同的工作

Prefill 一次处理已有 prompt，生成各层的 K/V；decode 每步只产生一个新位置的 Q/K/V，并读取缓存。单序列每层 KV 通常是 `[Hkv,T,Dh]`，batch 与 layout 因实现而异。缓存避免重算旧位置投影，但新 query 仍需要访问上下文，标准 dense attention 的每步读取量仍随 T 增长。

缓存不是答案缓存，也不是把所有历史 hidden states 都保存下来。修改旧 prompt、位置编号或模型权重后，旧 KV 不一定还能复用。分页管理和前缀共享属于系统层实现。

## 一个可验证的缓存实验

在 Notebook 中对同一 prompt 先跑完整前向，再取前缀的 `past_key_values`，仅把最后一个 token 输入第二次前向。比较最后位置的 logits，允许浮点误差。对带 padding 的真实 batch 还需要正确传 attention mask、position IDs 或 cache position；不要把单序列示例机械推广。

停止也属于解码器职责：EOS、最大新 token 数、stop string 与工具调用协议不是同一件事。记录是否因为长度截断停止，否则“答案不完整”可能是配置问题。

## 自测

**问题：** 改变温度后答案更正确，说明模型学到了新知识吗？

<details><summary>展开答案</summary>没有。权重不变，只是从已有分布选择输出的策略改变。需要多次采样、固定评估预算和正确性指标，才能判断这种变化是否稳定有益。</details>

## 原始资料与继续阅读

- [Transformers: GPT-2 model and outputs](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [The generation API](https://huggingface.co/docs/transformers/main_classes/text_generation)

资料核对：2026-09-10。教学示例不代表生产基准；框架 API 和模型支持请以所链接版本为准。
