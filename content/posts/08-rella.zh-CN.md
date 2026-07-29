---
title: "上下文放得下，不等于模型看得懂｜生成式推荐 08：ReLLa"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - ReLLa
  - Retrieval-Augmented Generation
pairKey: "generative-recommendation-08-rella"
slug: "generative-recommendation-08-rella-zh"
excerpt: "ReLLa 发现 LLM 在远未用满 context window 时就无法利用更长行为序列，并用目标相关行为检索 SUBR 与混合样本微调 ReiT 提高信噪比。"
series: "generative-recommendation"
seriesOrder: 8
draft: false
---

大模型支持 2K、8K、甚至更长上下文。那推荐系统是不是只要把用户过去所有点击塞进 prompt，就能获得越来越准确的画像？

直觉说“历史越多，信息越多”。ReLLa 的实验却画出另一条曲线：

```text
历史变长 → AUC 上升 → 很早到顶 → 继续变长反而下降
```

下降发生在大约 500–700 个 token，离 Vicuna-13B 的 2048 token 上限还很远。不是窗口装不下，而是模型无法从混杂历史中持续抽取与当前候选有关的证据。

[上一篇 TALLRec](/blog/generative-recommendation-07-tallrec-zh) 用少量 instruction tuning 教会 LLM 判断偏好。[ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation](https://arxiv.org/abs/2308.11131) 继续解决更接近真实用户的问题：

> 不要默认最近 $K$ 次行为就是最有用的 $K$ 次。先以目标商品为查询，从一生历史中检索最相关的行为，再交给 LLM。

## 30 秒看懂本文

1. **TALLRec 留下的问题**：它只使用约十件历史商品，没有验证 LLM 能否理解长期、多兴趣、含噪行为。
2. **ReLLa 的答案**：零样本时用 Semantic User Behavior Retrieval（SUBR）将 recent-$K$ 换成 relevant-$K$；少样本时把原始样本与检索增强样本混合，用 Retrieval-enhanced Instruction Tuning（ReiT）学习两种模式。
3. **最重要的证据**：用不到 10% 训练样本，ReLLa 在 BookCrossing、MovieLens-1M、MovieLens-25M 的 AUC 分别为 0.7575、0.8033、0.8477，超过使用全量数据的 SIM：0.7541、0.7992、0.8344。

![从最近 K 条历史到最相关 K 条历史](/blog/generative-recommendation/08-rella/paradigm-shift.svg)

*图 1：context capacity 只回答“放得下多少”；recommendation comprehension 还取决于信号密度与目标相关性。*

## 上一篇留下了什么问题

传统序列推荐常取最近 $K$ 个行为：

$$
H_{\text{recent}}
=
[i_{T-K+1},\ldots,i_T].
$$

这隐含“越近越有用”。对于短期 session 很合理，但长期用户同时存在：

- 稳定兴趣：长期喜欢户外；
- 临时意图：今天准备游泳；
- 代购行为：昨天替朋友看母婴用品；
- 噪声点击：误触或价格比较；
- 周期兴趣：每年冬季购买滑雪装备。

把它们按时间全部写进自然语言，token 数量还没超限，主题异质性已经很高。LLM 的 attention 能访问每个 token，不代表它知道哪个行为应当支持当前候选。

ReLLa 将这个现象称为 **lifelong sequential behavior incomprehension**。论文在多个 LLM 上都观察到早期峰值：

- MovieLens-1M 上，Vicuna-13B 在 $K=15$ 附近达到 AUC 峰值，继续加到 30 并未改善；
- Falcon-7B 甚至从 $K=5$ 的 0.5906 持续下降到 $K=30$ 的 0.5452；
- LLaMA2-70B 同样在短序列附近见顶。

更大的模型、足够长的窗口，都没有自动解决选择问题。

## 用八件商品理解 SUBR

假设小林的完整历史按时间是：

```text
I01 网球拍 → I02 吸汗带 → I03 攀岩鞋 → I04 粉袋
→ I05 羽毛球 → I07 瑜伽垫 → I08 蛋白棒 → 一次游泳毛巾点击
```

现在要判断他是否会点击目标 `I06 泳镜`，prompt 预算只允许四条历史。

**recent-4** 可能是：

```text
羽毛球、瑜伽垫、蛋白棒、游泳毛巾
```

**relevant-4** 则先比较每件历史与“防雾泳镜”的语义相似度，可能选出：

```text
游泳毛巾、吸汗带、羽毛球、网球拍
```

这里的玩具选择只是机制示意，不是论文个案。真实 SUBR 对每件商品构造描述文本，使用 LLM 最后一层 hidden state 的平均池化，再经 PCA 得到 512 维向量。

![以目标商品为 query 检索行为历史](/blog/generative-recommendation/08-rella/toy-retrieval.svg)

*图 2：SUBR 保持输入条数近似不变，改变的是选择函数。目标候选不同，同一用户被取出的历史也会不同。*

这意味着“用户表示”不再是一个固定摘要。判断泳镜时调用水上 / 运动相关历史；判断蛋白棒时可能调用健身与营养相关历史。它是 target-conditioned memory。

## 模型从输入到输出发生了什么

### 1. 为商品构造语义向量

把标题、类别、类型等字段写成描述：

```text
这是一件商品。名称是泳镜，类别是游泳装备，特征是防雾……
```

经过 LLM 后，对最后一层所有 token hidden states 求平均：

$$
\mathbf u_i\in\mathbb R^D.
$$

再用 PCA 降维与去噪：

$$
\mathbf v_i=\operatorname{PCA}(\mathbf u_i)\in\mathbb R^{512}.
$$

### 2. 对每个目标检索历史

计算目标 $t$ 与历史商品 $i$ 的 cosine：

$$
\operatorname{sim}(t,i)
=
\frac{\mathbf v_t^\top\mathbf v_i}
{\|\mathbf v_t\|_2\|\mathbf v_i\|_2},
$$

选 Top-$K$，再按指定方式写回 prompt。

### 3. 零样本：只做 SUBR

模型参数不变。原始 recent-$K$ prompt 被 relevant-$K$ prompt 替代，直接比较 `Yes` / `No` token 的 logits。

### 4. 少样本：ReiT

对每条训练样本构造两个版本：

```text
原始 recent-K 样本
+ SUBR relevant-K 样本
= mixed dataset，共 2N 条
```

再进行 causal LM instruction tuning。论文认为混合模式既增加样本数，也通过 pattern enrichment 充当正则化，降低仅适应检索样式的过拟合。

![SUBR 与 ReiT 的完整数据流](/blog/generative-recommendation/08-rella/mechanism.svg)

*图 3：SUBR 是数据层改写，ReiT 是参数层适配。测试时只使用检索增强样本。*

## 核心公式逐项拆解

LLM 输出整个词表的 logits $\mathbf s_i\in\mathbb R^{|V|}$。ReLLa 不额外训练 CTR head，而是截取答案词 `Yes` 与 `No` 的两个 logit：

$$
\hat y_i
=
\frac{\exp(s_{i,\mathrm{Yes}})}
{\exp(s_{i,\mathrm{Yes}})
+
\exp(s_{i,\mathrm{No}})}
\in(0,1).
$$

- $s_{i,\mathrm{Yes}}$：在当前 prompt 后生成 `Yes` 的未归一化分数；
- $s_{i,\mathrm{No}}$：生成 `No` 的分数；
- $\hat y_i$：目标商品的点击 / 喜欢分数。

只在两个答案词上重新 softmax，而不是使用整个词表概率。这样 $\hat y_i+\hat y_i^{\text{No}}=1$，可以直接算 AUC 与 Log Loss。

ReiT 的训练目标仍是：

$$
\max_\Theta
\sum_{(x,y)\in\mathcal M}
\sum_j\log P_\Theta(y_j\mid x,y_{<j}),
$$

其中 $\mathcal M$ 包含原始与 SUBR 两个版本。

## 训练和推理分别怎么做

```text
# offline item encoding
for each item:
    text = describe(item)
    hidden = LLM(text).last_layer
    u[item] = mean_pool(hidden)
v = PCA(u, dim=512)

# construct one target-conditioned sample
scores = cosine(v[target], v[user_history])
relevant = top_k(user_history, scores, K)
prompt = format(profile, relevant, target)
```

零样本：

```text
logits = frozen_LLM(prompt)
ctr = softmax([logit_yes, logit_no])[yes]
```

少样本：

```text
mixed = []
for sample in N training records:
    mixed += original_recent_k(sample)
    mixed += subr_relevant_k(sample)
instruction_tune(LLM, mixed)  # 2N instances
```

这个计算有一个容易被图示隐藏的代价：对每个候选商品都要检索一套历史。如果排序阶段有上千候选，需要高效向量索引、预筛选或批量矩阵计算。

## 实验究竟证明了什么

### 1. 长历史确实会让原始 LLM 退化

Vicuna-13B 零样本在三个数据集分别在 $K=30/15/15$ 左右达到峰值，然后下降；对应文本约 500/700/700 token，远未达到 2048 上限。ReLLa 的曲线则随 $K$ 基本持续改善，没有相同转折点。

### 2. 少样本 ReLLa 可以超过全量传统模型

| 数据集 | SIM full-shot AUC | ReLLa <10% AUC |
|---|---:|---:|
| BookCrossing | 0.7541 | **0.7575** |
| MovieLens-1M | 0.7992 | **0.8033** |
| MovieLens-25M | 0.8344 | **0.8477** |

![ReLLa 与全量 SIM 的 AUC 对比](/blog/generative-recommendation/08-rella/evidence.svg)

*图 4：数值来自原论文 Table 2。ReLLa 使用的 <10% 在不同数据集对应不同绝对样本数。*

### 3. 检索与混合训练各有贡献

在 <1% 设置下，完整 ReLLa 对比去掉 retrieval：

| 数据集 | w/o Retrieval | ReLLa |
|---|---:|---:|
| BookCrossing | 0.7167 | **0.7482** |
| MovieLens-1M | 0.7718 | **0.7927** |
| MovieLens-25M | 0.8174 | **0.8352** |

去掉 mixture、只训练检索增强样本也会退化。作者进一步控制样本量后认为，双倍数据和模式丰富都有效，而 pattern enrichment 的正则化更关键。

但零样本不是全面胜利：MovieLens-25M 上，Vicuna-13B 的 AUC 0.7503，高于 SUBR 版 ReLLa 的 0.7324；后者在 Log Loss 与 Accuracy 更好。指标之间的冲突提醒我们不能只复述摘要中的“全面提升”。

## 它失败在哪里

### 1. 语义相关不等于行为相关

SUBR 以商品文本相似度检索。用户可能把“咖啡机”和“除垢剂”连续购买，二者语义距离未必最近，却有强互补关系。

### 2. 每个候选都需要一套历史

pointwise CTR 场景可以接受 target-conditioned prompt；全目录召回无法为百万候选逐一检索与运行 LLM。

### 3. 过滤可能丢掉兴趣转移

只选择与目标相似的历史会强化已有主题，却可能压制探索、跨品类组合和突然兴趣。相关性检索不是用户全部意图。

### 4. 与 full-shot 基线的比较维度不完全相同

ReLLa 使用预训练 LLM 的外部世界知识；传统 SIM 从推荐数据训练。少量域内样本超过全量 SIM 证明数据效率，但不代表总训练算力或预训练数据更少。

### 5. 商品向量可能随模型更新而失效

ReiT 改变 LLM 参数。如果商品编码器与微调后的模型并非同一固定版本，需要处理向量重算、版本一致性与索引刷新。

### 6. “理解”仍由预测指标间接定义

曲线改善与 attention case study 支持模型更会利用历史，但没有证明它形成了可验证、因果正确的用户理解。

## 下一篇为什么会出现

ReLLa 解决了输入侧：从长历史中挑出更相关的内容。但它仍主要把商品表示成文字。

传统 SASRec 知道 `I03 → I04` 的协同与顺序关系，LLM 知道“攀岩鞋”和“粉袋”的世界含义。只保留文字会丢协同，只有 ID embedding 又无法调用语言知识。

[下一篇 LLaRA](/blog/generative-recommendation-09-llara-zh) 会把这两个空间真正放进同一个 prompt：将传统推荐器的 item embedding 投影成一种“行为 token”，与商品标题 token 拼接，再用 curriculum learning 从纯文本逐步过渡到混合模态。

---

**论文**：Lin et al., “ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation,” WWW 2024.

**原文**：[arXiv:2308.11131](https://arxiv.org/abs/2308.11131)

**系列导航**：[二十篇论文看懂生成式推荐的前世今生](/series/generative-recommendation)
