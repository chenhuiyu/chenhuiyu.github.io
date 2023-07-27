---
title: "Prompt Engineering"
date: 2023-07-27 15:44:10
categories:
- NLP Insights
tags:
- Prompt
- LLM
---
# Prompt Engineering

Prompt Engineering, 也被称为上下文提示，是指在不更新模型权重的情况下，与LLM（语言模型）进行交互以引导其产生期望输出的方法。它是一门实证科学，提示工程方法的效果在不同模型之间可能会有很大的差异，因此需要进行大量的实验和试探。

本文仅关注自回归语言模型的提示工程，不涉及填空测试、图像生成或多模态模型。在本质上，提示工程的目标是实现模型的对齐和可操控性。您可以查阅我之前关于可控文本生成的帖子。

## 基本提示方法

zero-shot学习和few-shot学习是两种最基本的提示模型方法，这些方法由许多LLM论文首创，并且通常用于评估LLM性能。

### zero-shot学习

zero-shot学习是将任务文本直接输入模型并要求获得结果。

（所有情感分析示例来自于SST-2数据集）

```
Text: i'll bet the video game is a lot more fun than the film.
Sentiment:
```
### few-shot学习

few-shot学习通过提供一组高质量的示例演示，每个示例都包含目标任务的输入和期望输出。当模型首先看到好的示例时，它可以更好地理解人类的意图和期望的答案类型。因此，few-shot学习通常比zero-shot学习表现更好。然而，这样做的代价是更多的记号消耗，并且在输入和输出文本较长时可能会达到上下文长度限制。

```
Text: (lawrence bounces) all over the stage, dancing, running, sweating, mopping his face and generally displaying the wacky talent that brought him fame in the first place.
Sentiment: positive

Text: despite all evidence to the contrary, this clunker has somehow managed to pose as an actual feature movie, the kind that charges full admission and gets hyped on tv and purports to amuse small children and ostensible adults.
Sentiment: negative

Text: for the first time in years, de niro digs deep emotionally, perhaps because he's been stirred by the powerful work of his co-stars.
Sentiment: positive

Text: i'll bet the video game is a lot more fun than the film.
Sentiment:
```
许多研究都探讨了如何构建上下文示例以最大化性能，并观察到提示格式、训练示例和示例的顺序选择可能会导致截然不同的性能，从几乎随机猜测到接近SOTA（State-of-the-Art）。

赵等人（2021年）研究了few-shot分类的情况，并提出了一些与LLM（他们在实验中使用了GPT-3）相关的偏差，这些偏差导致了高方差的情况：
- （1）多数类别偏差存在于示例的标签分布不平衡的情况下；
- （2）最近偏差是指模型可能在结尾重复标签；
- （3）常见记号偏差表明LLM倾向于更频繁地生成常见的记号而不是罕见的记号。为了克服这些偏差，他们提出了一种方法，通过对模型输出的标签概率进行校准，使其在输入字符串为N/A时保持均匀。

## 提示工程技巧

### 示例选择的建议

- 使用嵌入空间中的NN聚类（Liu等人，2021年）来选择与测试示例在语义上相似的示例。

- Su等人（2022年）提出了一种基于图的方法来选择多样且代表性的示例：
    1. 首先，根据样本之间的嵌入（例如SBERT或其他嵌入模型）余弦相似性构建一个有向图，其中每个节点指向其最近的邻居；
    2. 开始时有一组已选择的示例和一组剩余示例。每个示例都通过得分函数进行评分，其中得分函数的目标是保持低值，以鼓励选择多样化的示例。具体得分函数的计算公式未提供。

- Rubin等人（2022年）提出了针对上下文学习示例选择的对比学习方法。对于每个训练对（格式化的输入-输出对），可以通过LM分配的条件概率来衡量一个示例的质量。然后，可以根据得分对训练对进行排名，选择得分较高和得分较低的示例作为对比学习的正样本和负样本集。

- 有些研究人员尝试使用Q-Learning进行示例选择（Zhang等人，2022年）。

- 受不确定性主导的主动学习的启发，Diao等人（2023年）建议确定具有多次采样试验中高度不一致或熵值较高的示例，并注释这些示例以在few-shot提示中使用。

### 示例排序的建议

- 一般建议保持示例选择的多样性，与测试示例相关，并以随机顺序进行排列，以避免多数类别偏差和最近偏差。

- 增加模型大小或包含更多训练示例并不能减少上下文示例不同排列之间的方差。同一顺序对一个模型可能有效，但对另一个模型可能无效。当验证集有限时，可以考虑选择顺序，以使模型不会产生极端不平衡的预测或对其预测过于自信（Lu等人，2022年）。

## 指令提示

- 在提示中展示few-shot示例的目的是向模型解释我们的意图；换句话说，用示例来描述任务指令，以便模型能够理解用户意图并遵循指令。然而，few-shot的使用可能会消耗较多的记号，并限制输入长度，因为上下文长度有限。所以，为什么不直接给出指令呢？

- Instructed LM（例如InstructGPT，自然语言指令）使用高质量的（任务指令，输入，真实输出）元组对预训练模型进行微调，以使LM更好地理解用户意图并遵循指令。RLHF（人类反馈的强化学习）是一种常见的方法。采用指令遵循风格的微调使得模型更加符合人类意图，并极大地降低了通信成本。

- 在与指令模型进行交互时，我们应该详细描述任务要求，尽量具体和准确，并避免使用"不做某事"的表述，而是要明确指定要做什么。

# Chain-of-Thought (CoT) Prompting

Chain-of-Thought (CoT) Prompting（Wei等人，2022年）通过生成一系列简短的句子，逐步描述推理逻辑，即所谓的推理链或理由链，最终引导出最终答案。CoT在复杂的推理任务中效果更显著，特别是在使用大型模型（例如超过50亿参数的模型）时。对于简单的任务，CoT提示的受益较小。

CoT提示的两种主要类型：

## few-shot CoT

few-shot CoT是使用少量演示来引导模型，每个演示包含人工编写（或模型生成）的高质量推理链。

（以下所有数学推理示例来自GSM8k数据集）

问题：Tom和Elizabeth比赛爬山。Elizabeth花了30分钟爬上山。Tom花费的时间是Elizabeth的四倍。Tom爬上山需要多少小时？

答案：Tom需要30 * 4 = 120分钟爬上山。
Tom需要120/60 = 2小时爬上山。
所以答案是2。

===

问题：Jack是个足球运动员。他需要买两双袜子和一双足球鞋。每双袜子的价格是9.50美元，鞋子的价格是92美元。Jack有40美元。Jack还需要多少钱？

答案：两双袜子的总费用是9.50美元 x 2 = 19美元。
袜子和鞋子的总费用是19美元 + 92美元 = 111美元。
Jack还需要111美元 - 40美元 = 71美元。
所以答案是71。

===

问题：Marty有100厘米的缎带，他必须将其分成4等份。每个切割部分必须再分成5等份。每个最终切割部分将有多长？

答案：（待填写）

## zero-shot CoT

zero-shot CoT是使用自然语言陈述，例如“让我们逐步思考”，明确地鼓励模型首先生成推理链，然后再通过“因此，答案是”等提示来产生答案（Kojima等人，2022年）。或者使用类似的语句“让我们一步一步来计算，确保我们得到正确的答案”（Zhou等人，2022年）。

问题：Marty有100厘米的缎带，他必须将其分成4等份。每个切割部分必须再分成5等份。每个最终切割部分将有多长？

答案：让我们逐步思考。