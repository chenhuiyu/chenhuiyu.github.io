---
title: "生成式推荐到底在生成什么？"
date: "2026-07-24"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - LLM
pairKey: "generative-recommendation-preface"
slug: "generative-recommendation-preface-zh"
excerpt: "同样叫生成式推荐，模型可能在生成文本、商品 ID、整张列表，甚至一段思考。序章先把这些概念彻底分开。"
series: "generative-recommendation"
seriesOrder: 0
draft: false
---

假设小林最近点击了网球拍，买了吸汗带，又连续看了几双攀岩鞋。周五晚上，她打开 App，系统向她推荐了一个粉袋。

这次推荐可能来自四种完全不同的计算过程：

1. 系统先召回几千件商品，再给每件商品打分，粉袋恰好排在最前面；
2. 模型把她最近的行为看成一个序列，预测“下一个最可能发生的行为”是点击粉袋；
3. 大语言模型生成一句话：“你最近在看室内运动，可以试试这个粉袋”；
4. 模型不再遍历候选商品，而是直接生成粉袋对应的一串 **Semantic ID**，甚至继续生成一整张周末运动清单。

它们都可能出现在“生成式推荐”的论文里，但生成的对象、训练目标、解码方式和工程代价并不相同。

这个系列的 20 篇论文，只追问一个问题：

> 推荐系统从“给已有商品打分”，是怎样一步步变成“直接说出答案”的？

## 30 秒结论

传统推荐的核心是**评估候选**：给定用户 $u$ 和候选商品 $i$，计算匹配分数 $s(u,i)$，再取 Top-K。

生成式推荐的核心是**构造输出序列**：给定用户历史 $\mathcal H_u$，模型逐 token 生成 $y_1,y_2,\ldots,y_T$。这些 token 可以组成一句解释、一个商品标识，也可以组成一整张推荐列表。

两者最本质的差异，不是有没有使用 Transformer，也不是模型参数够不够大，而是：

- 输出空间由什么组成；
- 后一个结果能否依赖前面已经生成的结果；
- 商品如何被表示成 token；
- 输出怎样被约束在真实商品库和业务规则之内。

![同一个推荐请求对应四种不同的输出空间](/blog/generative-recommendation/preface/generation-outputs.svg)

*图 1：同样是推荐攀岩装备，系统可能输出候选分数、下一个 Item ID、自然语言，或一串可解码的 Semantic ID。*

## 一、旧世界：推荐系统是一台多阶段打分机

先不要急着埋葬传统推荐。它之所以长期采用 **召回（retrieve）—排序（rank）—重排（rerank）**，不是因为工程师没有想到端到端模型，而是因为真实商品库可能包含数百万乃至更多对象。

一个常见流程是：

1. **召回**从完整商品库 $\mathcal I$ 中快速取回几百到几千个候选；
2. **排序**为每个候选计算用户—商品匹配分数；
3. **重排**加入多样性、库存、合规、广告和业务目标；
4. 最终展示 Top-K。

可以把它写成：

$$
\mathcal C_u = \operatorname{Retrieve}(\mathcal H_u), \qquad
\hat S_u = \operatorname{TopK}_{i\in\mathcal C_u} s_\theta(u,i)
$$

$\mathcal C_u$ 是召回得到的候选集合，$s_\theta(u,i)$ 是模型对用户 $u$ 与商品 $i$ 的匹配判断。

这种范式非常务实：

- 每一层都能独立替换、扩容和排查；
- 热门商品、用户向量和中间结果可以缓存；
- 排序模型可以使用大量交叉特征；
- 强规则可以在最后一步确定性执行；
- 新商品进入索引后，不必等待一个生成模型重新“记住”整个目录。

但它也有一个结构性前提：**模型只能评价已经被放到面前的候选**。如果召回阶段没有把粉袋放进 $\mathcal C_u$，后面的排序模型再聪明，也无法把它推荐出来。

而且，当系统分别计算

$$
s(u,I04),\quad s(u,I05),\quad s(u,I06)
$$

时，三个分数通常不能自然表达“已经推荐了粉袋之后，下一件是否还应该推荐另一只粉袋”。列表内部的依赖关系往往交给额外的重排规则处理。

生成式推荐试图把一部分复杂性从多阶段管线移进模型和解码过程。

![传统多阶段推荐与生成式推荐的结构对比](/blog/generative-recommendation/preface/pipeline-shift.svg)

*图 2：传统系统把复杂性拆在召回、排序和重排中；生成式系统把更多依赖压进表示、解码和约束。后者不是自动更好，只是优化边界不同。*

## 二、“生成式推荐”其实指五件不同的事

“生成”这个词太宽泛。如果不先区分输出对象，讨论很容易变成鸡同鸭讲。

| 路线 | 模型主要生成什么 | 代表工作 | 与本系列的关系 |
|---|---|---|---|
| 概率生成模型 | 潜变量、用户表示或交互分布 | Mult-VAE、GAN、Diffusion-based Rec | 同名旁支，在序章说明但不占 20 篇主线 |
| 语言生成 | 评分答案、解释、偏好描述或对话 | [P5](https://arxiv.org/abs/2203.13366)、[M6-Rec](https://arxiv.org/abs/2205.08084) | LLM 支线的起点 |
| 商品 ID 生成 | 一个可映射回目录的 Item ID | [DSI](https://arxiv.org/abs/2202.06991)、[TIGER](https://arxiv.org/abs/2305.05065) | 生成式检索主线 |
| 列表或 session 生成 | 相互依赖的一串商品或未来行为 | [GPTRec](https://arxiv.org/abs/2306.11114)、[OneRec](https://arxiv.org/abs/2502.18965) | 工业统一路线 |
| 理由与推理生成 | 推荐理由、兴趣归纳或显式思考轨迹 | [OneRec-Think](https://arxiv.org/abs/2510.11639)、[OneReason](https://arxiv.org/abs/2606.06260) | 系列最后的开放问题 |

本系列把“生成式推荐”收窄为一个可检验的定义：

> 模型把推荐目标表示成 token 序列，并通过自回归或等价的序列生成过程，直接产生商品标识、推荐列表或与其联合的语言输出。

对应的目标可以写成：

$$
P_\theta(y_{1:T}\mid \mathcal H_u)
=
\prod_{t=1}^{T}
P_\theta(y_t\mid y_{<t},\mathcal H_u)
$$

这条公式最重要的不是“使用了概率”，而是条件项 $y_{<t}$：第 $t$ 个输出可以看到已经生成的前 $t-1$ 个输出。

于是，模型不仅能判断“泳镜适不适合小林”，还可以学习：

- 已经生成攀岩鞋后，再生成粉袋是否更连贯；
- 已经推荐三件攀岩用品后，是否应该加入游泳或羽毛球来提高多样性；
- 哪些 token 前缀对应一个真实商品，哪些路径必须在解码时禁止。

## 三、不是所有序列推荐都等于生成式推荐

这里有一个很容易混淆的边界。

[GRU4Rec](https://arxiv.org/abs/1511.06939)、[SASRec](https://arxiv.org/abs/1808.09781) 和 [BERT4Rec](https://arxiv.org/abs/1904.06690) 都在预测商品，也都借用了语言建模的思想。但在常见实现中，它们仍然会为商品词表或候选集计算分数，再取 Top-K。

它们为什么仍然是这套历史的“前世”？

因为它们完成了三次必要的观念迁移：

1. **用户由静态向量变成行为序列**；
2. **商品 ID 由数据库主键变成可学习 token**；
3. **推荐目标由一次匹配变成条件序列预测**。

换句话说，序列推荐和现代生成式推荐之间不是一条清晰的国境线，而是一条连续谱。判断一项工作位于哪里，最好问四个具体问题：

- 推理时是否必须显式枚举候选？
- 一个商品用一个独立 token，还是用多个可共享 token？
- 模型生成一个 item，还是联合生成整个 slate/session？
- 解码过程是否显式建模列表内部依赖和目录约束？

比起仅凭论文标题里有没有 “Generative”，这四个问题更可靠。

## 四、真正困难的地方：商品不是天然的词

语言模型之所以可以逐 token 生成句子，是因为大量单词共享一个相对稳定的子词词表。“攀岩鞋”和“登山鞋”虽然不是同一个商品，却会共享“鞋”、运动和户外等语义。

商品目录完全不同：

- 每件商品必须能精确映射回库存中的唯一对象；
- 商品数量可能远大于普通语言词表；
- 商品不断上新、下架、合并和换版本；
- 两件文本相似的商品，可能面向完全不同的人群；
- 两件文本完全不同的商品，也可能因为用户共购而高度相关。

因此，“怎样给商品发 token”不是输入预处理的小问题，而是生成式推荐的核心建模选择。

### Atomic ID：擅长记忆

最直接的方法是一件商品一个 token，例如攀岩鞋就是 `<I03>`。

它唯一、精确、容易映射回目录，但词表会随商品数线性增长。`<I03>` 与 `<I04>` 的编号本身没有告诉模型它们分别是攀岩鞋和粉袋，更没有告诉模型两者经常被一起购买。

### Textual ID：擅长语义

另一种方法是用商品标题或属性，例如 “indoor climbing shoes”。

它可以利用语言模型已有的世界知识，对新商品也更友好；但商品名称不保证唯一、稳定和可执行。模型生成了一段听起来合理的文本，不代表目录里存在那件精确商品。

### Semantic ID：在记忆和泛化之间折中

Semantic ID 把商品连续向量量化成一串短 code，例如：

```text
<sport> <climb> <shoe>
```

现实中的 code 通常不会这么可读，但直觉相同：不同商品可以共享前缀或部分 code，模型逐层缩小搜索空间。TIGER 使用 RQ-VAE 构造这类标识；LC-Rec、LETTER 和 ETEGRec 又继续追问，code 中应该放多少文本语义、多少协同行为，以及 tokenizer 是否应该和推荐目标联合训练。

这种方法也不是免费午餐。量化会丢失信息，多个商品可能发生碰撞，少数 code 可能被过度使用，动态目录还会带来编码维护问题。

![Atomic ID、Textual ID 与 Semantic ID 的对比](/blog/generative-recommendation/preface/item-identifiers.svg)

*图 3：Atomic ID 更擅长精确记忆，Textual ID 更擅长开放语义，Semantic ID 试图在可解码、共享结构与目录规模之间取得平衡。*

## 五、为什么 Transformer 和 LLM 让这条路线突然加速

Transformer 并不是因为“会聊天”才进入推荐，而是因为它提供了一套统一的序列机器：

- 用户行为可以写成输入序列；
- 商品可以写成 token；
- 多任务可以写成不同 prompt；
- 推荐结果可以写成输出序列；
- 训练可以统一成 masked prediction 或 next-token prediction；
- 模型规模、上下文长度和数据量可以在同一框架下扩展。

但推荐系统并不是换个词表就能直接复制语言模型。

语言建模与推荐之间至少有四个冲突：

1. **语言语义不等于协同语义**：描述相似的商品不一定被同一群用户喜欢；
2. **长上下文不等于有效历史**：能塞进窗口，不代表模型能从几十年行为中找出当前相关部分；
3. **开放词表不等于真实目录**：语言可以自由生成，商品必须真实存在、可售且合规；
4. **next-token likelihood 不等于平台价值**：点击、时长、多样性、满意度与长期留存可能互相冲突。

因此第二幕的 LLM4Rec 论文一直在做“对齐”：把通用语言模型的世界知识，与推荐系统从共现、点击和购买中学到的协同知识接起来。

第三幕的 Semantic ID 论文则在做另一种对齐：让商品 token 同时适合压缩、检索、冷启动和下游推荐目标。

## 六、两条支线最终在工业系统里汇合

这 20 篇不会机械地按年份排成一列，因为 2022 年之后，领域明显分成两条并行路线。

**LLM 支线**从 P5 和 M6-Rec 出发，依次面对“通用 LLM 不懂推荐”“长行为难以理解”“语言知识与协同知识分离”等问题。

**生成式检索支线**从 DSI 与 TIGER 出发，把焦点放在 Item ID、Semantic ID、codebook、列表生成与端到端 tokenizer 上。

到了 [HSTU](https://arxiv.org/abs/2402.17152)、OneRec 和 [MTGR](https://arxiv.org/abs/2505.18654)，问题不再只是某个离线 benchmark 上的 Recall@K，而是：

- 一个序列模型能否承接真实工业流量；
- 是否真的能够统一召回、排序与重排；
- 传统交叉特征是否应该被放弃；
- 训练与推理成本能否随规模扩展；
- 用户反馈能否通过 DPO 或 RL 进入生成目标。

再往后，OneRec-Think 与 OneReason 开始生成显式 reasoning。但“能够生成一段理由”和“理由忠实地反映了模型为什么推荐”是两件事。这个系列不会预设 reasoning 一定有用，而会把它作为最后需要审判的假设。

![20 篇生成式推荐论文的五幕路线图](/blog/generative-recommendation/preface/series-roadmap.svg)

*图 4：LLM 与 Semantic ID 是两条并行支线；它们在工业统一模型、偏好对齐和推荐 reasoning 中重新汇合。*

## 七、我们会怎样读这 20 篇论文

每篇文章都使用同一个玩具用户“小林”和八件运动商品。变化的不是例子，而是模型看待例子的方式。

每篇分成三层：

1. **零基础层**：先说清上一代方法为什么不够；
2. **工程层**：明确输入输出、tensor shape、训练数据、负采样或解码过程、复杂度；
3. **论文层**：只保留一个核心公式、一张关键实验、一个重要 ablation，以及论文没有证明什么。

全系列统一使用以下符号：

| 符号 | 含义 |
|---|---|
| $u$ | 用户 |
| $i$ | 商品 |
| $\mathcal I$ | 完整商品库 |
| $\mathcal H_u$ | 用户历史行为序列 |
| $s(u,i)$ | 用户—商品匹配分数 |
| $z_i=(z_i^1,\ldots,z_i^m)$ | 商品的多 token Semantic ID |
| $S_u$ | 一次请求生成的推荐列表或 session |

我们的目标不是记住 20 个模型缩写，而是每读完一篇，都能回答：

> 它改变了推荐系统的哪一个基本假设？它又把什么难题留给了下一篇？

## 八、生成式推荐还远没有大结局

截至今天，这条路线仍有许多没有解决的问题：

- 商品目录持续更新时，生成索引怎样增量学习而不遗忘？
- Semantic ID 应该优先保留内容语义，还是协同信号？
- Beam search、约束解码与低延迟线上服务怎样兼容？
- 自回归生成整个列表，是否真的优于强召回模型加重排器？
- 历史点击存在曝光偏差时，模型究竟在模仿偏好还是模仿旧系统？
- 多目标 reward 会改善长期体验，还是只制造新的代理指标？
- 推荐 CoT 如何验证忠实性，而不仅是可读性？

这也是为什么本系列不会写成一部“新范式必然淘汰旧范式”的胜利史。

生成式推荐真正有趣的地方，恰恰是它迫使推荐系统重新回答三个古老问题：

1. 用户偏好应该怎样表示？
2. 数百万件商品应该怎样组织？
3. 我们究竟在优化一次点击，还是一次更长期、更完整的体验？

在讨论模型怎样生成商品之前，我们必须先回到一切开始的地方：用户从来没有明确告诉系统“我不喜欢什么”。系统只有点击、购买和沉默。

下一篇，我们从 [BPR](https://arxiv.org/abs/1205.2618) 开始，看推荐系统怎样先学会一件最朴素的事：

> 对同一个用户，谁应该排在谁前面？

[查看完整的 20 篇路线图 →](/series/generative-recommendation)

## 本文涉及的原始论文

- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)
- [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)
- [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- [BERT4Rec](https://arxiv.org/abs/1904.06690)
- [P5: Recommendation as Language Processing](https://arxiv.org/abs/2203.13366)
- [M6-Rec](https://arxiv.org/abs/2205.08084)
- [Transformer Memory as a Differentiable Search Index](https://arxiv.org/abs/2202.06991)
- [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- [Generative Sequential Recommendation with GPTRec](https://arxiv.org/abs/2306.11114)
- [Actions Speak Louder than Words: HSTU](https://arxiv.org/abs/2402.17152)
- [OneRec](https://arxiv.org/abs/2502.18965)
- [MTGR](https://arxiv.org/abs/2505.18654)
- [OneRec-Think](https://arxiv.org/abs/2510.11639)
- [OneReason](https://arxiv.org/abs/2606.06260)
