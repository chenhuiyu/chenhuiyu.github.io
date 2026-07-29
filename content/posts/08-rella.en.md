---
title: "Fitting in Context Is Not the Same as Being Understood | Generative Recommendation 08: ReLLa"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - ReLLa
  - Retrieval-Augmented Generation
pairKey: "generative-recommendation-08-rella"
slug: "generative-recommendation-08-rella-en"
excerpt: "ReLLa finds that LLMs fail to benefit from longer behavior sequences far below their context limit, then raises signal-to-noise with target-conditioned behavior retrieval (SUBR) and mixed-sample tuning (ReiT)."
series: "generative-recommendation"
seriesOrder: 8
draft: false
---

Large language models support 2K, 8K, or much longer contexts. Should a recommender simply place every past click in the prompt and obtain an increasingly accurate user profile?

ReLLa measures a different curve:

```text
longer history → higher AUC → early peak → lower AUC
```

The decline begins around 500–700 tokens, far below Vicuna-13B’s 2,048-token limit. The sequence fits. The model fails to keep extracting target-relevant evidence from a heterogeneous history.

[TALLRec, the previous paper](/blog/generative-recommendation-07-tallrec-en), teaches an LLM preference prediction with a small number of instructions. [ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation](https://arxiv.org/abs/2308.11131) addresses the more realistic user:

> Do not assume the latest $K$ actions are the most useful $K$. Use the target item as a query, retrieve the most relevant behavior from a lifelong history, and only then ask the LLM.

## Understand the paper in 30 seconds

1. **TALLRec’s limitation:** it uses roughly ten historical items and does not test whether an LLM can comprehend long, multi-interest, noisy histories.
2. **ReLLa’s answer:** for zero-shot recommendation, replace recent-$K$ with relevant-$K$ using Semantic User Behavior Retrieval (SUBR). For few-shot learning, mix original and retrieved versions and perform Retrieval-enhanced Instruction Tuning (ReiT).
3. **The decisive evidence:** with under 10% of domain training data, ReLLa reaches AUC 0.7575, 0.8033, and 0.8477 on BookCrossing, MovieLens-1M, and MovieLens-25M, beating full-data SIM at 0.7541, 0.7992, and 0.8344.

![From the latest K actions to the most relevant K actions](/blog/generative-recommendation/08-rella/paradigm-shift.svg)

*Figure 1. Context capacity says how much can fit. Recommendation comprehension also depends on signal density and relevance to the current target.*

## What problem did the previous paper leave behind?

Sequential recommenders commonly retain the most recent $K$ actions:

$$
H_{\text{recent}}
=
[i_{T-K+1},\ldots,i_T].
$$

Recency is sensible for short sessions. Lifelong behavior mixes stable interests, temporary intent, purchases for others, accidental clicks, and seasonal patterns.

Writing all of those events as natural language creates high topical heterogeneity before it exhausts the token budget. Attention can technically access every token without knowing which action supports the current candidate.

ReLLa calls this **lifelong sequential behavior incomprehension**. Early peaks appear across backbones:

- Vicuna-13B on MovieLens-1M peaks around $K=15$ and does not improve at 30;
- Falcon-7B falls from 0.5906 AUC at $K=5$ to 0.5452 at $K=30$;
- LLaMA2-70B also peaks at a short sequence.

A larger backbone and sufficient context capacity do not automatically solve selection.

## Understand SUBR with eight items

Suppose Lin’s history is:

```text
I01 racket → I02 sweatband → I03 climbing shoes → I04 chalk bag
→ I05 shuttlecock → I07 yoga mat → I08 protein bar → swim towel
```

The target is `I06 goggles`, and the prompt budget retains four actions.

**recent-4** might contain:

```text
shuttlecock, yoga mat, protein bar, swim towel
```

**relevant-4** compares each historical item with “anti-fog goggles” and may retain:

```text
swim towel, sweatband, shuttlecock, racket
```

This selection is illustrative rather than a paper datapoint. Actual SUBR constructs item description text, mean-pools final-layer LLM hidden states, and applies PCA to obtain 512-dimensional semantic vectors.

![Using the target item to retrieve behavior history](/blog/generative-recommendation/08-rella/toy-retrieval.svg)

*Figure 2. SUBR leaves the input count roughly unchanged and modifies the selection function. A different target retrieves a different view of the same user.*

The user representation is no longer one fixed summary. Goggles query water and sport evidence; a protein bar may query fitness and nutrition evidence. This is target-conditioned memory.

## What happens from model input to output?

### 1. Encode every item semantically

Title, category, genre, and other fields form descriptive text. The LLM’s final-layer token states are mean-pooled:

$$
\mathbf u_i\in\mathbb R^D.
$$

PCA reduces dimension and noise:

$$
\mathbf v_i=\operatorname{PCA}(\mathbf u_i)\in\mathbb R^{512}.
$$

### 2. Retrieve history for each target

Compute cosine similarity:

$$
\operatorname{sim}(t,i)
=
\frac{\mathbf v_t^\top\mathbf v_i}
{\|\mathbf v_t\|_2\|\mathbf v_i\|_2},
$$

select Top-$K$, and write those actions into the prompt.

### 3. Zero-shot: SUBR only

Model parameters remain frozen. The relevant-$K$ prompt replaces the recent-$K$ prompt, and the system compares `Yes` and `No` logits.

### 4. Few-shot: ReiT

Each training example creates two forms:

```text
original recent-K instance
+ SUBR relevant-K instance
= mixed dataset of 2N instances
```

Causal-LM instruction tuning then learns from both. The paper argues that the mixture both doubles samples and enriches patterns, acting as regularization against overfitting one presentation.

![Complete SUBR and ReiT data flow](/blog/generative-recommendation/08-rella/mechanism.svg)

*Figure 3. SUBR changes data; ReiT adapts parameters. Testing uses retrieved instances only.*

## Unpack the central equation

The LLM produces vocabulary logits $\mathbf s_i\in\mathbb R^{|V|}$. ReLLa selects the two answer logits:

$$
\hat y_i
=
\frac{\exp(s_{i,\mathrm{Yes}})}
{\exp(s_{i,\mathrm{Yes}})
+
\exp(s_{i,\mathrm{No}})}
\in(0,1).
$$

- $s_{i,\mathrm{Yes}}$ scores `Yes` after the current prompt;
- $s_{i,\mathrm{No}}$ scores `No`;
- $\hat y_i$ is the click or preference score.

The softmax is recomputed over the two answer words, not the full vocabulary. This gives a normalized binary probability for AUC and Log Loss.

ReiT itself retains causal language modeling:

$$
\max_\Theta
\sum_{(x,y)\in\mathcal M}
\sum_j\log P_\Theta(y_j\mid x,y_{<j}),
$$

where $\mathcal M$ contains both original and SUBR forms.

## Training versus inference

```text
# offline item encoding
for each item:
    text = describe(item)
    hidden = LLM(text).last_layer
    u[item] = mean_pool(hidden)
v = PCA(u, dim=512)

# one target-conditioned sample
scores = cosine(v[target], v[user_history])
relevant = top_k(user_history, scores, K)
prompt = format(profile, relevant, target)
```

Zero-shot:

```text
logits = frozen_LLM(prompt)
ctr = softmax([logit_yes, logit_no])[yes]
```

Few-shot:

```text
mixed = []
for sample in N training records:
    mixed += original_recent_k(sample)
    mixed += subr_relevant_k(sample)
instruction_tune(LLM, mixed)  # 2N instances
```

One cost is easy to hide in a diagram: every candidate requires its own history retrieval. Ranking thousands of candidates needs an efficient vector index, batched matrix operations, or earlier filtering.

## What do the experiments actually prove?

### 1. Long history can degrade an unadapted LLM

Zero-shot Vicuna-13B peaks around $K=30/15/15$ on the three datasets and then declines. The corresponding prompts contain roughly 500/700/700 tokens, well below the 2,048 limit. ReLLa’s curves largely continue upward without the same turning point.

### 2. Few-shot ReLLa can beat a full-shot traditional model

| Dataset | SIM full-shot AUC | ReLLa <10% AUC |
|---|---:|---:|
| BookCrossing | 0.7541 | **0.7575** |
| MovieLens-1M | 0.7992 | **0.8033** |
| MovieLens-25M | 0.8344 | **0.8477** |

![ReLLa versus full-data SIM](/blog/generative-recommendation/08-rella/evidence.svg)

*Figure 4. Values from Table 2. “<10%” corresponds to a different absolute sample count on each dataset.*

### 3. Retrieval and mixture both contribute

Under the <1% setting:

| Dataset | without retrieval | ReLLa |
|---|---:|---:|
| BookCrossing | 0.7167 | **0.7482** |
| MovieLens-1M | 0.7718 | **0.7927** |
| MovieLens-25M | 0.8174 | **0.8352** |

Removing the mixture and training only on retrieved instances also hurts. A sample-count control suggests that both extra volume and pattern diversity matter, with pattern enrichment acting as the more important regularizer.

Zero-shot results are not a universal win: on MovieLens-25M, Vicuna-13B has AUC 0.7503 and SUBR ReLLa has 0.7324, although ReLLa improves Log Loss and accuracy. Conflicting metrics are a useful correction to a blanket “all settings improve” summary.

## Where does it fail?

### 1. Semantic relevance is not behavioral relevance

SUBR retrieves by item text. Coffee machines and descaling tablets may have strong co-purchase value without being nearest semantic neighbors.

### 2. Every candidate needs a history view

Target-conditioned prompts suit pointwise CTR ranking. They cannot score a million-item catalog one candidate at a time.

### 3. Filtering can suppress interest shifts

Selecting only target-similar history reinforces established themes and may hide exploration, cross-category complements, or sudden intent.

### 4. Few-shot and full-shot compare different priors

ReLLa uses world knowledge from large pretraining, while SIM learns from recommendation data. The result proves domain-data efficiency, not lower total compute or fewer total training tokens.

### 5. Item embeddings need version discipline

ReiT changes model parameters. If item vectors and the tuned model are not tied to a fixed encoder version, the system must handle recomputation and index refresh.

### 6. “Comprehension” is inferred from prediction

Improved curves and attention case studies support better history use, but they do not establish a causally correct, inspectable user model.

## Why does the next paper need to exist?

ReLLa improves the input side by selecting relevant text. It still represents items mainly through language.

SASRec knows the collaborative transition from `I03` to `I04`; an LLM knows what “climbing shoes” and “chalk bag” mean. Text alone loses collaboration, while an ID embedding alone cannot invoke world knowledge.

[LLaRA, the next paper](/blog/generative-recommendation-09-llara-en), places both in one prompt. It projects a conventional recommender’s item embedding into a behavioral token, concatenates that token with the title, and uses curriculum learning to move gradually from text-only to hybrid input.

---

**Paper:** Lin et al., “ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation,” WWW 2024.

**Primary source:** [arXiv:2308.11131](https://arxiv.org/abs/2308.11131)

**Series:** [Twenty Papers to Understand the Past and Present of Generative Recommendation](/series/generative-recommendation/en)
