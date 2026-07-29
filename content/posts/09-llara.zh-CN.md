---
title: "世界知识与协同行为怎样合体？｜生成式推荐 09：LLaRA"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - LLaRA
  - Multimodal Alignment
pairKey: "generative-recommendation-09-llara"
slug: "generative-recommendation-09-llara-zh"
excerpt: "LLaRA 把传统序列推荐器的 Item ID embedding 投影成行为 token，与商品标题 token 拼接，并通过 curriculum learning 让 LLM 同时利用世界知识与协同规律。"
series: "generative-recommendation"
seriesOrder: 9
draft: false
---

“泳镜”这两个字告诉 LLM：它属于游泳、会防雾、可能与泳帽一起使用。

`item_06` 在 SASRec 里的 embedding 则告诉模型：点击过某些商品的人，下一步常点击它；它与哪些匿名 ID 有协同和顺序关系。

这两类知识互补，却住在完全不同的坐标系：

```text
语言空间：标题、描述、常识
推荐空间：ID、共现、行为转移
```

[上一篇 ReLLa](/blog/generative-recommendation-08-rella-zh) 用语义向量从长历史中检索相关行为，但送入 LLM 的仍主要是商品文字。[LLaRA: Large Language-Recommendation Assistant](https://arxiv.org/abs/2312.02445) 提出更直接的融合：

> 把传统推荐器学到的 Item ID embedding 看作一种“行为模态”，投影到 LLM embedding 维度，作为额外 token 与商品标题一起写入 prompt。

这不是让 LLM 取代 SASRec，而是让 SASRec 成为 LLM 的一个感知器官。

## 30 秒看懂本文

1. **ReLLa 留下的问题**：文本语义能筛选相关行为，却无法完整表达传统推荐器学到的匿名共现与顺序模式。
2. **LLaRA 的答案**：对每个商品同时构造 textual tokens 与 projected behavioral token；先用纯文本 prompt 学任务，再按 $p(\tau)=\tau/T$ 逐渐增加 hybrid prompt，让 LLM 适应新模态。
3. **最重要的证据**：LLaRA 在 MovieLens、Steam、LastFM 的 HitRatio@1 分别达到 0.4737、0.4949、0.4508，高于 TALLRec 的 0.3895、0.4637、0.4180；原始 Llama2 的合法输出率最低只有 16.53%，LLaRA 全部超过 95%。

![文本知识与协同行为的融合](/blog/generative-recommendation/09-llara/paradigm-shift.svg)

*图 1：hybrid item representation 不是在文本和 ID 中二选一，而是让两个 token 序列共同进入 LLM。*

## 上一篇留下了什么问题

到这里，LLM 推荐主要有两种商品表示。

### 只写 Item ID

```text
The user interacted with item_17, item_42, item_9...
```

优点是 ID 与日志一一对应。缺点是预训练 LLM 不知道这些编号是什么；数字被 tokenizer 拆开后还可能产生虚假的邻近关系。

### 只写商品文本

```text
The user watched Titanic, Roman Holiday...
```

优点是可以调用 LLM 的世界知识，也能描述新商品。缺点是文字没有完整保存“谁与谁共同被点击”的协同结构。

例如两部电影题材相似，受众却可能不同；两件商品文本不相似，却可能在同一任务链里连续出现。传统推荐器正是在大量匿名行为中学习这种分布。

LLaRA 的核心假设是：

> 序列行为不是文本的一个普通字段，而是一种独立模态。融合前要先建立模态接口。

## 用八件商品理解 hybrid token

以 `I06 泳镜` 为例。

### 文本 token

LLM tokenizer 处理标题：

$$
\langle\mathrm{emb}_{I06}^{t}\rangle
=
\operatorname{LLM\text{-}TKZ}(\text{“I06 泳镜”}).
$$

它可能对应多个 token embedding，携带“游泳”“护目”等语言先验。

### 行为 token

先训练一个 GRU4Rec、Caser 或 SASRec，读取其中的 Item embedding：

$$
\mathbf e_{I06}^{s}
=
\operatorname{SR\text{-}EMB}(I06;\Theta_e)
\in\mathbb R^d.
$$

这个向量不直接是 LLM token。LLaRA 使用两层感知机 SR2LLM：

$$
\langle\mathrm{emb}_{I06}^{s}\rangle
=
\operatorname{Proj}(\mathbf e_{I06}^{s};\Theta_p).
$$

输出维度与 LLM hidden size 一致，可以插进 token 序列。

### 混合表示

$$
\langle\mathrm{emb}_{I06}^{c}\rangle
=
\operatorname{Concat}
\left(
\langle\mathrm{emb}_{I06}^{t}\rangle,
\langle\mathrm{emb}_{I06}^{s}\rangle
\right).
$$

Prompt 于是类似：

```text
历史：
网球拍 <behavior_I01>,
攀岩鞋 <behavior_I03>,
羽毛球 <behavior_I05>

候选：
泳镜 <behavior_I06>,
瑜伽垫 <behavior_I07>, ...

只输出一个候选标题。
```

![一个商品如何变成文本 token 与行为 token](/blog/generative-recommendation/09-llara/toy-hybrid.svg)

*图 2：文本侧调用世界知识，行为侧携带传统推荐器的分布；projector 只建立可输入接口，不保证知识已完全对齐。*

## 模型从输入到输出发生了什么

如果从第一步就把陌生行为 token 全量塞进 prompt，LLM 要同时学习：

1. 什么是下一项推荐；
2. 输出必须来自候选；
3. 投影向量怎样解释；
4. 文本与行为冲突时听谁的。

LLaRA 因此使用 curriculum prompt tuning。

### Easy task：text-only prompt

每个商品标题后先放特殊占位 token `[PH]`：

```text
Titanic [PH], Roman Holiday [PH], ...
```

输入仍是 LLM 熟悉的文本。模型先学会任务与输出格式。

### Hard task：hybrid prompt

把 `[PH]` 替换成对应的 projected behavioral token：

```text
Titanic <behavior_14>, Roman Holiday <behavior_20>, ...
```

历史与候选全部使用混合表示。

### Curriculum scheduler

训练时间为 $\tau$、总步数为 $T$，采样 hard task 的概率线性增长：

$$
p(\tau)=\frac{\tau}{T}.
$$

最初几乎全是 easy；末期几乎全是 hard。LLM 用 LoRA 更新，projector 同时训练，传统推荐器 embedding 也可纳入目标。

![LLaRA 从纯文本到混合 token 的课程学习](/blog/generative-recommendation/09-llara/mechanism.svg)

*图 3：curriculum 不是“先两轮 text、再三轮 hybrid”的硬切换；每一步按逐渐变化的概率采样。*

若 batch size 为 $B$，历史长度 $n$，候选数 $m$：

| 表示 | 典型 Shape | 含义 |
|---|---:|---|
| recommender item emb | $[B,n+m,d_s]$ | ID 行为知识 |
| projected behavior | $[B,n+m,d_{\text{LLM}}]$ | 可插入 LLM 的单 token |
| text tokens | 可变长 | 标题对应的语言 token |
| final prompt | $[B,L,d_{\text{LLM}}]$ | 文本与行为交错后的输入 |

## 核心公式逐项拆解

课程训练在 easy loss 与 hard loss 之间随机选择：

$$
\mathcal L_\tau
=
\left(1-I_\tau\right)\mathcal L_{\text{easy}}
+
I_\tau\mathcal L_{\text{hard}},
$$

$$
I_\tau
\sim
\operatorname{Bernoulli}\left(\frac{\tau}{T}\right).
$$

- $\mathcal L_{\text{easy}}$：只看商品标题和 `[PH]` 的 token NLL；
- $\mathcal L_{\text{hard}}$：标题后加入行为 token 的 token NLL；
- $I_\tau=1$：本步训练 hybrid prompt；
- $\tau/T$：随训练推进增加 hard 样本概率。

期望损失是：

$$
\mathbb E[\mathcal L_\tau]
=
\left(1-\frac{\tau}{T}\right)\mathcal L_{\text{easy}}
+
\frac{\tau}{T}\mathcal L_{\text{hard}}.
$$

这条式子直观展示“从语言先验向行为模态迁移”的连续路径。它不会保证每个 batch 难度单调增加，但总体比例单调变化。

## 训练和推理分别怎么做

```text
# step 0: train a conventional sequential recommender
SR = train(GRU4Rec or Caser or SASRec, item_sequences)
item_behavior = SR.item_embeddings

# LLaRA tuning
for step tau in 1...T:
    p_hard = tau / T
    if random() < p_hard:
        prompt = titles + Project(item_behavior)
        loss = hard_token_nll(prompt, true_next_title)
    else:
        prompt = titles + [PH]
        loss = easy_token_nll(prompt, true_next_title)

    update LoRA and projector
```

推理：

```text
history = hybrid_tokens(user_history)
candidates = hybrid_tokens(candidate_set)
prompt = format(history, candidates)
answer = constrained_or_parsed_generation(prompt)
return matching candidate
```

原论文每个测试样本从 20 个未交互商品中采样负候选，再加入一个真实下一项，共 21 个候选。LLM 生成一个标题，以 HitRatio@1 评估。

## 实验究竟证明了什么

论文在 MovieLens、Steam、LastFM 上比较传统序列模型、原始 Llama2、GPT-4、MoRec、TALLRec 与 LLaRA。

| 模型 | MovieLens HR@1 | Steam HR@1 | LastFM HR@1 |
|---|---:|---:|---:|
| TALLRec | 0.3895 | 0.4637 | 0.4180 |
| LLaRA 最佳变体 | **0.4737** | **0.4949** | **0.4508** |

![LLaRA 对 TALLRec 的 HitRatio@1](/blog/generative-recommendation/09-llara/evidence.svg)

*图 4：数值来自原论文 Table 2；LLaRA 在每个数据集选取 GRU4Rec、Caser、SASRec 三种行为底座中的最佳值。*

另一组容易被忽略的证据是 ValidRatio：

- 原始 Llama2：MovieLens 0.4421、Steam 0.1653、LastFM 0.3443；
- LLaRA 各变体：均高于 0.95，最高达到 1.0。

这说明 instruction tuning 不只提升排序，也让模型更常输出候选集中可解析的答案。

Ablation 支持两部分设计：

- 仅数字索引、仅行为 token、仅商品文本，都弱于 hybrid representation；
- curriculum 在 MovieLens / Steam 上优于 direct 与 two-stage；
- LastFM 上 curriculum 0.4508 与 direct 0.4508 持平，不能说它在所有数据集严格提升；
- 三种传统推荐器都能提供行为 embedding，说明框架不绑定 SASRec。

论文 case study 还展示两类互补：

- 当正确答案依赖电影题材常识时，TALLRec 与 LLaRA 能胜过只看行为的 SASRec；
- 当正确答案来自数据里的匿名顺序模式时，SASRec 与 LLaRA 能胜过只看文本的 TALLRec。

case study 是解释性证据，不是因果证明，但与 hybrid 假设一致。

## 它失败在哪里

### 1. 仍是小候选集重排

21 个候选中的 Hit@1 远小于百万目录生成 / 检索难度。候选由外部采样提供，LLaRA 没有统一召回。

### 2. 先要训练一个传统推荐器

所谓“一个 LLM 推荐器”实际依赖 GRU4Rec、Caser 或 SASRec 先产出 embedding。训练、更新与版本管理是双系统。

### 3. 新商品没有成熟行为 token

文本 token 能描述冷商品，但推荐器 embedding 缺少交互。hybrid 表示在冷启动时会退化为不平衡的两侧。

### 4. Projector 只解决接口，不保证语义对齐

两层 MLP 把维度映射到 LLM 空间，却没有显式约束“相似行为 token 应对应怎样的语言概念”。模型可能学会使用向量，而不形成可解释对应。

### 5. 自由生成仍有无效输出

ValidRatio 大幅提高但不天然为 100%。生产系统仍需 trie / candidate-constrained decoding 或字符串映射。

### 6. Curriculum 收益不大且不稳定

Table 3 的提升通常只有几个百分点，LastFM 与 direct 持平。更复杂的 scheduler 是否值得，要按数据与训练成本验证。

### 7. 文本长度与候选数共同放大成本

每个历史和候选都带标题 token + 行为 token。候选集扩大时，prompt 很快增长，LLM 前向成本成为瓶颈。

## 下一篇为什么会出现

第二幕到这里形成了五步：

1. **P5**：把任务统一为 text-to-text；
2. **M6-Rec**：把统一底座放进工业推荐链路；
3. **TALLRec**：用少量监督跨越 language–recommendation gap；
4. **ReLLa**：为当前目标检索长期历史；
5. **LLaRA**：融合语言知识与协同行为。

但它们大多仍在一个候选集里评分或生成标题。召回索引仍在模型外，Item ID 也仍是任意编号。

[下一篇 DSI](/blog/generative-recommendation-10-dsi-zh) 将暂时离开推荐论文，回到生成式检索的源头：如果模型能把文档内容记进参数，并直接生成 doc ID，检索还需要“查询向量 → 外部索引 → 最近邻”这条路径吗？

这个大胆设想会催生 TIGER 的 Semantic ID，也会成为后半部所有生成式推荐模型的共同祖先。

---

**论文**：Liao et al., “LLaRA: Large Language-Recommendation Assistant,” SIGIR 2024.

**原文**：[arXiv:2312.02445](https://arxiv.org/abs/2312.02445)

**系列导航**：[二十篇论文看懂生成式推荐的前世今生](/series/generative-recommendation)
