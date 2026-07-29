---
title: "如果检索不再查索引，而是直接生成 ID｜生成式推荐 10：DSI"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - DSI
  - Generative Retrieval
pairKey: "generative-recommendation-10-dsi"
slug: "generative-recommendation-10-dsi-zh"
excerpt: "DSI 把文档写进模型参数，让同一个 Transformer 同时学习 document→ID 与 query→ID；检索从向量近邻搜索变成了受条件约束的 ID 生成。"
series: "generative-recommendation"
seriesOrder: 10
draft: false
---

搜索引擎通常像图书馆：先把每本书放进索引，查询来了，再去索引里找最近的书。

[上一篇 LLaRA](/blog/generative-recommendation-09-llara-zh) 已经把推荐器的行为 embedding 变成 LLM 能接收的 token，但候选商品仍然在模型外部：LLM 负责“理解”，检索系统负责“找”。[Differentiable Search Index（DSI）](https://arxiv.org/abs/2202.06991) 提出了一个激进问题：

> 索引能不能不再是一张外部表，而直接住进 Transformer 的参数里？查询能不能不再返回向量，而是生成文档 ID？

DSI 研究的是文档检索，不是推荐；但它给后来的生成式推荐提供了最关键的动作：**把可检索对象的标识符当成语言序列来生成。**

## 30 秒看懂本文

1. **上一代的边界**：LLM 与推荐器可以融合表示，但最后仍要打分或查询外部候选库，模型并没有学会“说出目标对象”。
2. **DSI 的答案**：用同一个 encoder–decoder 做两种任务——索引时学习 `document → docid`，检索时学习 `query → docid`；文档内容由模型参数记忆。
3. **最重要的证据与警告**：在 NQ320K 的 228K 个唯一文档上，DSI-XXL Semantic String 的 Hits@1 为 40.4，高于 T5-XXL 双塔的 24.3；但该结果来自“最小遗忘”检查点，同一训练过程的最大遗忘点只有 12.2。

![从查索引到生成文档 ID](/blog/generative-recommendation/10-dsi/paradigm-shift.svg)

*图 1：DSI 把外部索引的一部分职责压进模型参数；它改变的是检索接口，而不只是换了一个编码器。*

## 上一篇留下了什么问题

假设用户行为是：

```text
网球拍 → 攀岩鞋 → 羽毛球拍 → 泳帽
```

传统推荐器先得到一个用户向量，再对全部商品做点积，或者在 ANN 索引中找近邻。即使前面换成 LLM，末端仍然通常是：

```text
理解用户 → 产生 query embedding → 查候选索引 → 重排
```

这个架构成熟而高效，却把知识分成两份：模型参数知道“如何理解”，外部索引知道“有哪些对象”。DSI 想把它们统一为一个条件生成问题：

\[
P(\text{目标 ID}\mid \text{输入})
\]

统一并不等于免费。外部索引容易新增、删除和重建；写进神经网络参数的知识更新更难，也会遗忘。DSI 的价值，正是在这种大胆统一与工程代价之间打开了一条新路线。

## 用八件商品理解“可微索引”

沿用全系列的八件商品：

| ID | 商品 | 简化内容 |
| --- | --- | --- |
| I01 | 网球拍 | 碳素、控制型 |
| I02 | 网球 | 耐打训练球 |
| I03 | 攀岩鞋 | 室内抱石 |
| I04 | 镁粉袋 | 攀岩防滑 |
| I05 | 羽毛球拍 | 轻量进攻型 |
| I06 | 泳镜 | 防雾训练款 |
| I07 | 泳帽 | 硅胶长发款 |
| I08 | 运动毛巾 | 速干 |

先做“索引任务”：

```text
输入：防雾训练泳镜
输出：6
```

再做“检索任务”：

```text
输入：游泳时眼睛容易进水，想要防雾装备
输出：6
```

两条样本的输入不同，目标相同。第一条迫使模型建立“内容与编号”的绑定；第二条学习“查询与编号”的映射。真实 DSI 用文档而非商品，但机制完全相同。

![八件商品如何被写入模型](/blog/generative-recommendation/10-dsi/toy-index.svg)

*图 2：索引不是一次前向计算后保存向量，而是一批持续训练模型的监督样本。*

## ID 不是无意义的输出格式

DSI 比较了三种 docid。

### Atomic ID

每篇文档拥有一个独立 token，例如 `<doc_228031>`。生成只需一步，但输出 softmax 随文档数线性增长；新文档还需要新增 token。

### Naïve String ID

把任意整数拆成普通字符或数字 token，例如 `2 → 2 → 8 → 0 → 3 → 1`。词表小了，编号之间却没有结构；前缀相同只是一场数字巧合。

### Semantic String ID

先用 8 层 BERT 表示文档，再做分层 k-means。论文使用分支因子 10，节点中文档少于 100 时停止；从根到叶的簇编号组成 ID。内容相近的文档共享较长前缀：

```text
泳镜：〈运动, 水上, 游泳, 眼部装备〉
泳帽：〈运动, 水上, 游泳, 头部装备〉
```

这样，解码第一个 token 是粗分类，后面的 token 逐步细化。Semantic ID 既压缩词表，又把搜索树嵌入输出空间，成为 TIGER 等后续方法的直接思想来源。

## 模型从输入到输出发生了什么

DSI 使用 T5 encoder–decoder。设输入 token 长度为 \(n\)，ID 长度为 \(m\)：

```text
input_ids       [B, n]
encoder states  [B, n, d]
decoder prefix  [B, t]
next-token logit[B, |V_id|]
```

索引与检索共享全部参数，只通过输入内容区分任务。生成 Semantic String 时，decoder 自回归地输出 \(j_1,j_2,\ldots,j_m\)。

![DSI 的训练与检索机制](/blog/generative-recommendation/10-dsi/mechanism.svg)

*图 3：两类监督共同把模型变成“索引 + 检索器”；缺少索引任务时，模型并不知道 ID 指向哪篇文档。*

## 核心公式逐项拆解

把文档和查询统一写成输入 \(x\)，目标 docid 写成 token 序列 \(j=(j_1,\ldots,j_m)\)，训练目标是：

\[
\mathcal L_{\text{DSI}}
=-\sum_{(x,j)}\sum_{t=1}^{m}
\log P_\theta(j_t\mid j_{<t},x)
\]

- \(x\)：可以是“防雾训练泳镜”这篇文档，也可以是“游泳防雾装备”这个查询；
- \(j_t\)：Semantic ID 的第 \(t\) 层，例如先生成“运动”，再生成“水上”；
- \(j_{<t}\)：已经生成的地址前缀；
- \(\theta\)：同时承担记忆与检索的 Transformer 参数。

这仍是标准交叉熵。革命性不在损失函数，而在**把什么定义为输入和输出**：分类标签变成可组合的地址，索引构建变成参数学习。

## 训练和推理分别怎么做

论文比较了先索引后检索的 sequential training，以及把两类样本混合的 multitask co-training。后者更好，因为完全切换到查询任务后，模型会忘记文档—ID 绑定。

```text
for batch in mixed(index_examples, retrieval_examples):
    if batch.type == INDEX:
        x = document_tokens
    else:
        x = query_tokens
    target = docid_tokens
    loss = cross_entropy(model(x), target)
    update(loss)

retrieve(query):
    return beam_search(model, query)
```

原论文没有在主实验中使用合法 ID 约束。也就是说，模型理论上可能生成不存在的字符串；后来的生成式推荐普遍引入 trie constrained decoding，把非法分支的 logit 直接屏蔽。

## 实验究竟证明了什么

NQ320K 含约 320K 条 query–document 对、228K 个唯一文档。表 2 的核心结果如下。

![DSI 在 NQ320K 上的检索结果](/blog/generative-recommendation/10-dsi/evidence.svg)

*图 4：Hits@1 的跃升说明模型确实能把大量文档映射写进参数；Hits@10 与强双塔更接近，说明优势并非所有位置都同样大。*

| 方法 | Hits@1 | Hits@10 |
| --- | ---: | ---: |
| BM25 | 11.6 | 34.4 |
| T5-XXL Dual Encoder | 24.3 | 67.3 |
| DSI-XXL Semantic String | **40.4** | **70.3** |

但必须读附录。验证文档被加入训练流并经有限 buffer 打乱，模型在不同时间点的遗忘程度差异很大。Semantic String DSI-XXL 在：

| 检查点 | Hits@1 | Hits@10 |
| --- | ---: | ---: |
| 最小遗忘 | **40.4** | **70.3** |
| 平均遗忘 | 26.3 | 50.2 |
| 最大遗忘 | 12.2 | 30.1 |

因此更准确的结论是：**DSI 证明 Transformer 有能力承担可微索引，但没有解决稳定写入和持续更新。**

另一个关键消融是：不做索引任务，Hits@1 为 0；直接用完整文档预测 ID 有效，而 Targets2Inputs 或 span corruption 等替代任务几乎得不到有意义的检索。模型不是靠“见过相似语言”自然知道编号，显式绑定不可缺。

## 它失败在哪里

### 1. 规模仍小

最大的实验索引只有 228K 个唯一文档，离大型电商或视频平台的千万、十亿级目录很远。参数容量、解码成本和 ID 冲突如何随规模增长，论文没有证明。

### 2. 更新不像索引那样可控

传统索引插入一条记录即可；DSI 加一篇文档需要训练。删除也不是删一行，而是要让模型“忘掉”。这涉及灾难性遗忘、版本一致性和合规删除。

### 3. 结果依赖检查点

最小与最大遗忘点的巨大差距说明平均指标背后存在时间稳定性问题。如果只引用 40.4，会把“最好瞬间”误写成“稳定能力”。

### 4. 生成并不天然合法

没有约束解码时，token 序列可能不是任何文档的 ID；约束解码虽然能修复合法性，却会引入目录 trie、beam search 和在线版本同步。

### 5. 记住文档不等于理解用户

DSI 的 query–document 任务没有多步用户历史、曝光偏差、重复消费、业务规则和列表目标。它是生成式检索的起点，不是完整推荐系统。

## 下一篇为什么会出现

DSI 已经回答了“能否生成 ID”，但它的 Semantic String 来自通用文档语义，尚未解决三个推荐特有的问题：

1. 商品地址如何从内容自动学习，而不是靠层次聚类规则？
2. 相似商品如何共享前缀，同时保证每个真实商品都有唯一地址？
3. 新商品没有交互时，能否仅凭内容得到可检索 ID？

[下一篇 TIGER](/blog/generative-recommendation-11-tiger-zh) 把这些问题带入序列推荐：用 RQ-VAE 把商品内容压缩成多级 Semantic ID，再让一个生成模型写出下一件商品的“语义地址”。

---

**本篇批判性结论**：DSI 最深的贡献不是某个 Hits 数字，而是把检索重新表述为“学习一门对象 ID 的语言”。它同时暴露了这门语言最难的工程问题：地址怎样设计、知识怎样更新、模型怎样不忘。接下来五篇论文，本质上都在修这三个问题。
