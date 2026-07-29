---
title: "Top-K Is Not the Only Answer: A List Can Be Written Item by Item | Generative Recommendation 12: GPTRec"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - GPTRec
  - List Generation
pairKey: "generative-recommendation-12-gptrec"
slug: "generative-recommendation-12-gptrec-en"
excerpt: "GPTRec contrasts one-pass Top-K scoring with autoregressive Next-K list generation and quantizes SVD item coordinates into compact multi-token IDs."
series: "generative-recommendation"
seriesOrder: 12
draft: false
---

A recommender ultimately delivers a list, not one isolated item.

Top-K systems score candidates independently and keep the K largest values. That is efficient, but item two never sees item one. Two near-identical shoes may both score highly; the model has no opportunity to say, “running is already covered, so I will represent the user's climbing interest next.”

[GPTRec: Generative Sequential Recommendation with GPTRec](https://arxiv.org/abs/2306.11114) separates two questions:

1. Must one product equal one token?
2. Must a recommendation list be produced in one Top-K operation?

Its answers are SVD-based multi-token IDs and autoregressive **Next-K** generation.

## The paper in 30 seconds

1. **TIGER's open problem:** it generates one item's Semantic ID but does not directly model dependencies among list members.
2. **GPTRec's answer:** use a GPT-2-style decoder; Top-K emits independent catalog scores once, whereas Next-K learns \(\prod_kP(r_k\mid H,r_{<k})\), letting later recommendations observe earlier choices.
3. **The result:** on MovieLens-1M, GPTRec-TopK reaches 0.146 NDCG@10, close to BERT4Rec's 0.152; NextK reaches 0.105. Four SVD tokens with 512 values each use only 2,048 embeddings and score 0.108.

One fact is easy to miss:

> GPTRec uses the GPT-2 decoder architecture, but it does **not** initialize from a pretrained GPT-2 checkpoint. “GPT” here denotes autoregressive structure, not imported world knowledge.

![Top-K versus Next-K](/blog/generative-recommendation/12-gptrec/paradigm-shift.svg)

*Figure 1. Top-K scores candidates in parallel; every Next-K decision conditions on the list prefix already generated.*

## List conditioning with eight items

Our user history remains:

```text
I01 tennis racket → I03 climbing shoes
→ I05 badminton racket → I07 swim cap
```

Top-K might independently assign:

```text
I06 goggles       0.92
I08 sports towel  0.89
I02 tennis balls  0.87
```

Next-K instead generates:

```text
step 1: P(I06 | history)
step 2: P(I08 | history, I06)
step 3: P(I04 | history, I06, I08)
```

Step two can use the fact that goggles are already present; step three can cover climbing. The architecture creates a channel for list interaction. It does not automatically create diversity: if training still rewards only ordinary next-item likelihood, the model may continue copying popular or similar items.

![Building a list one item at a time](/blog/generative-recommendation/12-gptrec/toy-next-k.svg)

*Figure 2. Next-K can express a list policy, but actual diversity requires data, objectives, or reinforcement signals that reward it.*

## A second route to item tokens: SVD quantization

TIGER learns Semantic IDs from content embeddings. GPTRec proposes a direct collaborative compression based on the user–item matrix.

For \(M\in\mathbb R^{|U|\times|I|}\), compute a truncated decomposition:

\[
M\approx U\Sigma E^\top.
\]

\(E_i\in\mathbb R^t\) is item \(i\)'s collaborative coordinate. For each dimension:

1. normalize item values to \([0,1]\);
2. add Gaussian noise with standard deviation \(10^{-5}\) to break boundary ties;
3. quantize into \(v\) bins;
4. offset dimension \(d\)'s token range by \(v(d-1)\).

With \(t=4,v=512\), every product has four tokens but the model stores only \(4\times512=2048\) embeddings, independent of item count:

```text
I06 goggles → 〈dim1-117, dim2-403, dim3-026, dim4-288〉
```

These are not content semantics. They discretize collaborative coordinates, so items consumed by similar users tend to have related patterns.

## The central equation

For history \(H\) and list \(R=(r_1,\ldots,r_K)\), Next-K factorizes:

\[
P(R\mid H)
=\prod_{k=1}^{K}
P(r_k\mid H,r_{<k}).
\]

- \(r_1\) sees history only;
- \(r_2\) also sees the first recommendation;
- \(r_{<k}\) is the generated list prefix;
- order is therefore part of the distribution.

With atomic IDs, each \(r_k\) is one token. With SVD IDs, it expands into \(t\) subtokens. The latter saves item embeddings but requires more autoregressive steps.

![GPTRec's token and list choices](/blog/generative-recommendation/12-gptrec/mechanism.svg)

*Figure 3. Item compression and list factorization are orthogonal axes: one-token Next-K and multi-token Top-K are both possible.*

## Training and inference

### Top-K

Read the final hidden state, score it against all item embeddings once, and return the largest K logits. This resembles SASRec.

### Next-K

Append each output to the context and score the next item. A ten-item list takes roughly ten scoring passes; multi-token items require still more decoder steps.

```text
context = user_history
for k in 1..K:
    item = decode_one_item(context)
    result.append(item)
    context.append(item)
```

The paper's training remains based on observed next items rather than a mature list-level RL or diversity objective. **Next-K is an expressive parameterization, not evidence that list utility has already been optimized.**

## What the experiments establish

The evidence comes from MovieLens-1M alone. Table 3 reports:

![MovieLens-1M results](/blog/generative-recommendation/12-gptrec/evidence.svg)

*Figure 4. Top-K approaches the strongest baseline; Next-K pays a substantial accuracy cost for conditional list modeling.*

| Method | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| BERT4Rec | **0.152** | **0.282** |
| GPTRec-TopK | 0.146 | 0.254 |
| SASRec | 0.108 | 0.199 |
| GPTRec-NextK | 0.105 | 0.157 |

Next-K retains roughly 72% of Top-K NDCG, summarized by the authors as about 75%, while matching SASRec's NDCG. Sequentializing the list is not free: early errors contaminate later conditions.

Multi-token experiments expose a capacity–quality trade-off:

| Representation | Embeddings | NDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| One token per item | 3,416 | 0.146 | 0.254 |
| 2 tokens × 2,048 values | 4,096 | 0.124 | — |
| 4 tokens × 512 values | **2,048** | 0.108 | 0.182 |

The four-token version uses about 40% fewer embeddings than the 3,416 atomic item embeddings and matches SASRec NDCG. Two tokens with larger bins perform better. A prose sentence in the paper appears to call the one-token NDCG 0.253; this chapter uses Table 3's 0.146.

## Where GPTRec breaks

### Expressivity is not an objective

Next-K permits later items to depend on earlier ones, but likelihood training mainly rewards relevance. Coverage, diversity, and long-term satisfaction must still be measured and optimized.

### Autoregressive error accumulation

An incorrect first recommendation becomes part of step two's context. Teacher forcing shows gold prefixes in training while inference sees model prefixes.

### Latency grows with K

Top-K scores once; Next-K scores K times. Multi-token IDs multiply the number of decoding steps again.

### SVD IDs have no content cold start

They come from interactions, so a new item has no reliable column coordinate. Recomputing SVD can also shift coordinates and quantization boundaries, creating address-version migrations.

### Evidence is narrow

MovieLens-1M has only 3,416 items. It does not demonstrate industrial scaling for either multi-token output or Next-K serving.

## Why LC-Rec comes next

We now have two item languages:

- TIGER: content-based Semantic IDs, useful for semantics and cold start;
- GPTRec-SVD: collaborative tokens from the interaction matrix.

Recommendation needs both. Textually similar products need not attract the same users, while collaboratively similar products may have unrelated names. [LC-Rec, in the next chapter](/blog/generative-recommendation-13-lc-rec-en), combines a pretrained LLM, Semantic IDs, and collaborative alignment tasks—and introduces uniform semantic mapping to remove ID conflicts.

---

**Critical takeaway:** GPTRec separates two questions often conflated in generative recommendation: how to tokenize an item and how to factorize a list. It demonstrates that compact tokens and Next-K are feasible, while its accuracy and latency costs show that a generative interface merely enlarges the optimization space; the list objective still has to be specified.
