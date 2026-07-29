---
title: "会写思维链，不等于会推理｜生成式推荐 20：OneReason"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "zh-CN"
tags:
  - Generative Recommendation
  - Recommender Systems
  - OneReason
  - Recommendation Reasoning
pairKey: "generative-recommendation-20-onereason"
slug: "generative-recommendation-20-onereason-zh"
excerpt: "OneReason 将推荐推理拆成 item perception 与 preference cognition，用四层任务、专才再统一的强化学习和快慢系统，回答“为什么 thinking mode 真的有效”。"
series: "generative-recommendation"
seriesOrder: 20
draft: false
---

上一篇留下了一个看似反常、其实非常重要的结果：

> 给推荐模型更多“思考 token”，thinking mode 不一定比直接回答更准。

[OneRec-Think](/blog/generative-recommendation-19-onerec-think-zh) 已经完成 item code 对齐、文字思维链监督和 GRPO，为什么仍会这样？因为推荐不是数学证明。证明题有一条可核验的演绎链，而用户下一步可能买护腕、看游泳教学、点恢复训练，也可能什么都不做。模型面对的是不完整历史与多个合理未来，需要从结果倒推潜在意图——这更接近**溯因推理**。

2026 年 6 月发布、仍标注为 work in progress 的 [OneReason Technical Report](https://arxiv.org/abs/2606.06260) 给出的诊断是：有效推荐推理需要两个地基。

1. **Perception（感知）**：模型真的知道 itemic tokens 对应什么内容；
2. **Cognition（认知）**：模型能把冗长、冲突的行为重组为稳定偏好、兴趣演化和当前意图。

这篇终章不把“推理”当一段漂亮文字，而是追问：中间过程是否让最终推荐更好？

## 30 秒看懂本文

1. **前一代的问题**：有 CoT 不等于会推理。OneReason 的 SFT 实验里，thinking mode 在视频、商品、广告和直播四个域的 Recall@64 全部低于 non-thinking mode。
2. **OneReason 的答案**：用 token、item、relation、user 四粒度预训练打牢感知；把认知分为 R0 感知、R1 关系推导、R2 兴趣演化、R3 推荐；再用“分域专才 → 统一模型”的 RL，让思维链真正导向可命中的商品。
3. **最重要的证据**：经过 RFT，thinking mode 在四个域全部反超 non-thinking；10 天线上实验中，OneReason 直接召回与增强 OneRec 的组合方案报告曝光 +10.332%、收入 +8.234%。但论文仍是私有工业数据上的技术报告，推理忠实性尚未被证明。

![从更多思考 token 到感知与认知](/blog/generative-recommendation/20-onereason/paradigm-shift.svg)

*图 1：OneReason 的关键转向不是把 CoT 写得更长，而是先让模型认得商品，再让它压缩历史、判断兴趣转移。*

## 用八件商品理解“推荐推理”

沿用全系列的八件商品：

| 编号 | 商品 | 三级 itemic pattern |
| --- | --- | --- |
| A | 网球拍 | `<sport><racket><tennis>` |
| B | 网球 | `<sport><ball><tennis>` |
| C | 攀岩鞋 | `<sport><shoe><climb>` |
| D | 防滑粉 | `<sport><gear><climb>` |
| E | 羽毛球拍 | `<sport><racket><badminton>` |
| F | 羽毛球 | `<sport><ball><badminton>` |
| G | 泳帽 | `<sport><wear><swim>` |
| H | 速干毛巾 | `<sport><gear><swim>` |

用户依次发生：

```text
买网球拍 → 看攀岩鞋 → 搜羽毛球拍 → 购买泳帽
```

下一个商品是 H。一个脆弱的思维链会复述：

```text
用户喜欢运动，最近购买泳帽，所以推荐速干毛巾。
```

答案合理，却没有处理三个难点：用户是在频繁换运动，还是持续尝试“轻装备”？泳帽是长期偏好，还是旅行前的一次性需求？不同设备上的行为是否来自同一个人？

OneReason 希望模型分层完成：

```text
R0 感知：<sport><wear><swim> = 泳帽，而不是任意 ID。
R1 推导：泳帽与速干毛巾共享游泳/洗浴场景，是互补关系。
R2 演化：网球 → 攀岩 → 羽毛球 → 游泳，显示“尝试新运动”；
          最近的游泳信号强于早期球拍信号。
R3 推荐：在“新运动探索”与“当前游泳场景”交点上生成 H。
```

![八件商品中的两轴压缩与三级认知](/blog/generative-recommendation/20-onereason/toy-reasoning.svg)

*图 2：推荐推理不是把历史逐项翻译成文字，而是压缩稳定画像、提取兴趣演化，再建立局部商品关系。*

## 第一步：让商品 code 真正可感知

### 多模态 item tokenizer

OneReason 先用视觉编码器、语言模型和音频信息理解商品内容，再把稠密 embedding 量化为三级 Residual Quantization K-Means code。每级有 8192 个 code：

\[
s_i=[d_i,c_i^{(1)},c_i^{(2)},c_i^{(3)}],
\]

其中 \(d_i\) 是域 token，例如 `video`、`prod`、`ad`、`living`；后三项逐级表示从粗到细的内容残差。与早期 Semantic ID 不同，域 token 明确区分业务空间，格式也不再需要结束 token。

但量化只保证“相似内容地址靠近”，不保证语言模型能解释地址。OneReason 因此混合四个粒度的预训练数据：

- **Token 粒度**：解释单级 code 以及不同级 code 的组合语义；
- **Item 粒度**：完整 itemic pattern 与标题、描述、问答、多模态内容双向转换；
- **Relation 粒度**：说明两个商品为何相关，补上内容相似之外的常识与协同迁移；
- **User 粒度**：把一段行为映射到画像与兴趣，也从画像生成合理行为。

同时保留通用文本和多模态数据，避免模型只会说“商品 code 方言”。这比单纯增加新 token embedding 更重，却直接对应 Perception：如果 `<prod><17><903><44>` 没有可恢复的内容语义，后面的自然语言 CoT 只是对未知符号编故事。

## 第二步：把认知拆成可训练的层级

OneReason 将监督微调任务分为 R0–R3：

| 层级 | 问题 | 八商品例子 |
| --- | --- | --- |
| R0 Perception | 这个 code 是什么？ | 识别 G 是泳帽 |
| R1 Derivation | 两个 item 为什么相关？ | G 与 H 是游泳场景互补品 |
| R2 Evolution | 兴趣怎样随时间变化？ | 从球拍类转向游泳 |
| R3 Recommendation | 综合后下一项是什么？ | 生成 H 的 itemic pattern |

其中 R3 的 CoT 数据有一个关键约束：**目标商品不能出现在 reasoning trace 中**。模型必须引用历史证据并在最后才输出答案，减少“先泄露答案、再倒写解释”。

### 两轴压缩

面对长历史，逐条总结既浪费 token，也把噪声等同于信号。论文先沿两条轴压缩：

1. **Persona compression**：长期类目、生活阶段、消费节奏、价格敏感度、互动深度，以及共享设备造成的冲突；
2. **Interest-evolution compression**：触发、扩展、收窄、延续、饱和替代、跨域呼应、从浏览到购买的闭环。

然后用 R1 建立局部桥梁，用 R2 判断时间演化，最后由 R3 生成目标。这一设计把“总结”和“决策”分开：摘要回答用户大体是谁，演化回答此刻哪种偏好应该占上风。

![OneReason 从预训练到专才再统一](/blog/generative-recommendation/20-onereason/mechanism.svg)

*图 3：四粒度预训练解决感知，R0–R3 SFT 建立认知脚手架，分域 RL 再把会讲道理变成会命中答案。*

## 第三步：为什么要“专才再统一”

把视频、商品、广告、直播样本直接混在一起做 RL，会产生梯度干扰：

- 视频偏好更新快、有效目标多；
- 电商购买稀疏、决策周期长；
- 广告还受竞价与曝光机制影响；
- 直播强依赖时间与主播状态。

OneReason 先从同一个 SFT checkpoint 出发，分别对四个域做 GRPO，得到四个 domain teachers；再用两条路线统一：

### RFT：只学习成功轨迹

让专才教师 rollout，保留命中正样本且质量较高的 reasoning，以 rejection sampling 重新微调统一学生。优点是稳定、简单；缺点是忽略失败轨迹，长尾域可能留下很少样本。

### MOPD：在学生自己的轨迹上吸收多教师

Multi-Teacher On-Policy Distillation 让统一学生在线采样，再用对应域教师与学生的 token 概率差构造优势：

\[
\hat A_t^{\text{MOPD}}
=\operatorname{sg}\!\left[
\log \pi_{d}(y_t\mid x,y_{<t})
-\log \pi_{\theta}(y_t\mid x,y_{<t})
\right].
\]

教师明显比学生更相信的 token 获得正向信号。论文还用 information-gain filter 降低大量“师生已经一致”的样本对稀有知识的稀释。MOPD 能利用失败与长尾轨迹，但上限受教师能力约束，优化也更复杂。

## 核心公式：奖励的不只是命中，还要避免重复

推荐答案不是唯一的。OneReason 对每条 CoT 不只生成一个商品，而是生成 \(K\) 条 itemic sequences。若 \(N\) 条 reasoning 各自生成 \(K\) 个答案，就有 \(N\times K\) 个 rollout；长 CoT 只算 \(N\) 次，降低训练成本。

第 \(i\) 条 reasoning 下第 \(j\) 个答案的奖励是：

\[
R_{u,i,j}
=R_{\text{rule}}(c_{i,j})\,
R_{\text{div}}(\mathrm{CoT}_i),
\]

\[
R_{\text{rule}}(c)=\mathbf 1[c\in C_u^+],
\qquad
R_{\text{div}}
=\frac{\max(0,m_i^{(1)}-1)}{K-1}.
\]

逐项映射到玩具例子：

- \(C_u^+\)：日志里该用户可视为正反馈的商品集合，不必只有 H；
- \(c_{i,j}\)：第 \(i\) 条 reasoning 生成的第 \(j\) 个商品 code；
- \(m_i^{(1)}\)：这 \(K\) 个答案中，第一级语义 code 有多少种；
- \(R_{\text{rule}}\)：命中正样本才有分；
- \(R_{\text{div}}\)：同一 CoT 若只复制 K 个近似地址，奖励会变小。

例如一条 reasoning 的四个答案全是游泳毛巾，命中可能高但多样性低；另一条覆盖毛巾、泳镜、恢复装备，若其中命中正样本，会得到更强信号。这个 reward 仍以结果为准，不评价文字推理是否真实，却至少承认“多个未来都合理”与“候选不能塌缩”。

## 训练与推理到底怎样运行

完整训练顺序可以压缩为：

```text
多模态商品 → 三级 RQ-KMeans codes
          ↓
四粒度 itemic pre-training + 通用数据
          ↓
R0–R3 SFT（thinking / non-thinking 都训练）
          ↓
四个单域 GRPO teachers
          ↓
RFT 或 MOPD 统一
```

推理则有两种形态：

- **Thinking**：生成两轴压缩、R1/R2 桥梁，再生成 itemic answer；计算多、可提供可读中间态；
- **Non-thinking**：直接生成商品 code；延迟低，而且论文发现 CoT 监督的能力会部分迁移到直接解码。

线上进一步采用 Fast–Slow Thinking：

- **Slow**：OneReason 在近线直接做生成式召回；
- **Fast**：把 OneReason 的输出变成 embedding 或 Thinking Token，增强实时 OneRec。

因此，工业“推理系统”不是每次请求都输出长篇文字，而是让慢模型定期重整兴趣，让快模型在毫秒级消费压缩后的结论。这与上一篇 Think-Ahead 一脉相承，但输出不再只是一组 code 前缀。

## 实验究竟证明了什么

最有价值的不是最大模型击败基线，而是 thinking/non-thinking 的方向变化。

![SFT 与 RFT 前后 thinking mode 的反转](/blog/generative-recommendation/20-onereason/evidence.svg)

*图 4：数值为 Recall@64（%）。仅做 SFT 时，多写一段 CoT 在四域全部更差；RFT 后 thinking mode 才稳定反超。*

| 训练 | 模式 | 视频 | 商品 | 广告 | 直播 |
| --- | --- | ---: | ---: | ---: | ---: |
| SFT | non-thinking | **0.11** | **2.96** | **6.49** | **15.52** |
| SFT | thinking | 0.06 | 1.65 | 3.41 | 14.32 |
| RFT | non-thinking | 0.19 | 3.96 | 7.26 | 18.17 |
| RFT | thinking | **0.24** | **4.19** | **7.50** | **18.35** |

证据支持三件事：

1. **CoT 本身不是免费增益**：SFT thinking 在四域全部退步；
2. **推理需要结果反馈校准**：RFT 后，thinking 才在四域稳定超过 non-thinking；
3. **差距不大但方向一致**：广告和直播的绝对优势只有 0.24 与 0.18 个百分点，不能写成“推理取得巨大飞跃”。

在跨域基线表中，OneReason-RFT-thinking 的 Recall@64 分别为 0.24%、4.19%、7.50%、18.35%，高于 LC-Rec-PT-SFT-8B 的 0.13%、3.00%、6.55%、16.70%。但不同目录的正样本数量差异很大，四列不能横向解释为“直播比视频容易多少”。

10 天、每组 5% 流量的线上 A/B 报告：

| 部署方式 | 曝光 | 收入 |
| --- | ---: | ---: |
| OneReason 直接召回 | +0.940% | +4.528% |
| OneReason 增强 OneRec | +6.831% | +4.636% |
| 两者组合 | **+10.332%** | **+8.234%** |

这是很强的系统证据：慢召回与快模型增强有互补性。它仍不能单独归因于可见 CoT；收益可能同时来自更好的 tokenizer、更多预训练数据、分域 RL、更大算力和新的召回通路。

## 它仍然失败在哪里

### 1. 这是未完成的技术报告

论文 v1 发布于 2026 年 6 月，明确标注 work in progress，模型也以“将开源”表述。庞大的私有训练集、四域 benchmark 和在线链路难以独立复现，数字应视为团队报告，而不是已经被外部验证的定论。

### 2. 奖励验证答案，不验证推理

命中正商品的 CoT 可以包含虚假事实、遗漏关键行为，甚至只是幸运地导向正确 code。论文证明“带 reasoning 的策略更准”，没有证明每一句 reasoning 是决策的真实因果路径。

### 3. 日志不是完整偏好

\(C_u^+\) 只包含曝光后观察到的正反馈。用户没有见过的好商品、同样合理但未发生的未来都被当作无奖励。溯因问题仍被压回不完整的离线标签。

### 4. 业务指标不等于用户福祉

曝光与收入提升重要，但不直接代表满意度、多样性、长期信任或内容健康。一个更会推理的商业优化器，也可能更有效地放大短期诱因。

### 5. 慢思考仍被搬到近线

Fast–Slow 设计诚实地承认长 CoT 的延迟。生产收益来自系统分层，而不是 8B 模型在每次在线请求中实时深思。缓存陈旧、兴趣突变、版本一致性仍是工程风险。

## 二十篇论文之后，我们到底看懂了什么

这条故事线从来不只是“把 Transformer 做大”：

1. [BPR](/blog/generative-recommendation-01-bpr-zh) 把隐式反馈写成“正样本应排在负样本前”；
2. GRU4Rec、SASRec、BERT4Rec 依次让模型理解顺序、选择相关历史、从双向上下文恢复缺失行为；
3. P5 到 LLaRA 把用户、商品与任务写进语言模型，却暴露了语言空间和协同空间的错位；
4. DSI 到 ETEGRec 把商品变成可逐 token 生成的 Semantic ID，也带来量化碰撞、目录更新与约束解码；
5. HSTU、OneRec、MTGR 把生成范式推到万亿参数、session 目标和工业多任务；
6. OneRec-Think 与 OneReason 最后追问：模型生成正确地址之前，能否形成有用、可干预、可验证的中间认知。

生成式推荐的“前世”是排序：从候选中打分；“今生”是生成：把检索、排序甚至解释折叠到统一 token 空间。但它的下一阶段还取决于五个没有解决的问题：

- **身份如何更新**：商品变化、上新与下架时，Semantic ID 怎样版本化而不破坏旧行为？
- **反事实如何学习**：只有一次曝光日志，怎样奖励多个有效未来？
- **推理如何验真**：删改一段 CoT，答案是否按可预测方向改变？
- **规模如何支付**：训练、近线慢思考、缓存与在线快模型的收益如何覆盖成本？
- **目标由谁定义**：点击、收入、满意度、多样性与长期生态冲突时，系统应该优化什么？

系列走到这里，真正开放的问题才刚开始。

回到[《二十篇论文看懂生成式推荐的前世今生》专题页](/series/generative-recommendation)，你可以按四幕主线重读，也可以从任意一个尚未解决的问题反向追踪它的技术来源。
