---
title: "From Item Vectors to Valid Semantic IDs | A Generative Recommendation Hands-on Lab"
date: "2026-07-30"
updated: "2026-07-30"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Semantic ID
  - Residual Quantization
  - Beam Search
  - Python
pairKey: "generative-recommendation-hands-on"
slug: "generative-recommendation-hands-on-en"
excerpt: "An interactive generative-recommendation lab: inspect residual quantization, tune decoding, compare Trie constraints, and run editable Python in the browser."
series: "generative-recommendation"
seriesOrder: 21
draft: false
---

After reading TIGER, LETTER, or OneRec, it is easy to leave with two misleading intuitions: a Semantic ID is merely a renamed item number, and beam search simply “keeps a few more candidates.” Running the pipeline exposes what those summaries omit: why an embedding can be discretized, how the address enters a generation objective, why individually legal tokens can compose an item that does not exist, and whether the final evaluation should measure recommendation or generation quality.

This tutorial compresses the pipeline into a teaching system that can be run from end to end. It follows the same arc as “Generative Recommendation Through 20 Papers,” but does not require reading the complete series first.

## Choose a runtime

| Entry point | Best for | What you can change |
|---|---|---|
| Interactive lab on this page | Build intuition and traverse the pipeline in minutes | Item, temperature, beam width, decode depth, Trie constraint, and Python |
| [Download the full notebook](/notebooks/generative-recommendation-hands-on.ipynb) | Work cell by cell in local Jupyter or VS Code | BPR training, codebooks, Semantic IDs, decoder, and evaluation |
| [Open in Colab](https://colab.research.google.com/github/chenhuiyu/chenhuiyu.github.io/blob/source/public/notebooks/generative-recommendation-hands-on.ipynb) | Run without installing an environment and save a copy to Drive | Every notebook cell and parameter |

The page uses eight items and emphasizes immediate interaction. The notebook uses 12 items and eight user histories, trains a small BPR model, fits residual codebooks, and executes constrained and unconstrained generation plus checks. They teach the same mechanism at different magnifications.

## Map the lab to the paper series

| Lab stage | Series chapters | Question preserved here |
|---|---|---|
| Continuous scoring | [01 · BPR](/blog/generative-recommendation-01-bpr-en) | How did recommenders learn user and item vectors before generation? |
| Sequences become tokens | [02 · GRU4Rec](/blog/generative-recommendation-02-gru4rec-en), [03 · SASRec](/blog/generative-recommendation-03-sasrec-en), [04 · BERT4Rec](/blog/generative-recommendation-04-bert4rec-en) | Why can a history no longer be collapsed into one static point? |
| Retrieval becomes generation | [10 · DSI](/blog/generative-recommendation-10-dsi-en), [11 · TIGER](/blog/generative-recommendation-11-tiger-en) | How can the model write the target’s discrete address directly? |
| Semantic ID design | [13 · LC-Rec](/blog/generative-recommendation-13-lc-rec-en), [14 · LETTER](/blog/generative-recommendation-14-letter-en), [15 · ETEGRec](/blog/generative-recommendation-15-etegrec-en) | Should a good ID preserve content, collaborative behavior, or downstream loss? |
| Unified generation | [17 · OneRec](/blog/generative-recommendation-17-onerec-en) | Can retrieval and ranking share one generation objective? |
| Failure and generalization | [2026 · CARE](/blog/generative-recommendation-2026-02-care-en), [2026 · Generalization audit](/blog/generative-recommendation-2026-07-generalization-en) | How should depth error, invalid IDs, and unseen items be measured? |

## 0. What are we optimizing?

Given a user history \(x_{1:t}\), a conventional scorer computes \(s(u,i)\) for candidates and selects Top-K from the catalog. A generative model writes the target as \(M\) discrete tokens:

$$
i \longrightarrow (c_1,c_2,\ldots,c_M), \qquad
p(i\mid x_{1:t})=\prod_{m=1}^{M}p(c_m\mid x_{1:t},c_{<m})
$$

That rewrite introduces three explicit interfaces:

1. a **tokenizer** that turns continuous item representations into stable and discriminative addresses;
2. a **decoder** that maps user context to a probability at every token level;
3. a **catalog constraint** that guarantees the complete address names a real item.

The notebook implements these interfaces separately so that a single “large model” does not hide the engineering boundaries.

## 1. Learn item vectors with BPR

The notebook holds out the last item in each history for a next-item check, then trains a NumPy BPR model on the remaining implicit feedback. For user \(u\), positive item \(i\), and sampled negative \(j\):

$$
\mathcal{L}_{\text{BPR}}
=-\log \sigma\big(s(u,i)-s(u,j)\big)+\lambda\lVert\Theta\rVert_2^2
$$

You will see the pairwise loss, each user’s Top-3 items, and a two-dimensional projection of learned item embeddings. The toy Recall@3 is not the point. The important contract is that the tokenizer receives a continuous representation learned from recommendation behavior.

[BPR](/blog/generative-recommendation-01-bpr-en) explains the scoring objective; [LC-Rec](/blog/generative-recommendation-13-lc-rec-en) asks why content semantics alone may not be enough.

## 2. Produce Semantic IDs with residual quantization

At level \(m\), the codebook quantizes only the residual left by previous levels:

$$
q_m=\arg\min_k\lVert r_{m-1}-e_{m,k}\rVert_2^2,\qquad
r_m=r_{m-1}-e_{m,q_m}
$$

Four codebooks give each item a four-token address. Early tokens identify coarse regions; later tokens repair reconstruction details. The notebook reports every item ID, residual error by level, collision rate, and token usage at each codebook.

Low reconstruction error does not automatically mean good recommendation. Nearby vectors can belong to different collaborative populations, while a codebook can reconstruct well and still overload a few tokens. That distinction separates [TIGER](/blog/generative-recommendation-11-tiger-en), [LETTER](/blog/generative-recommendation-14-letter-en), and [DIGER](/blog/generative-recommendation-2026-01-diger-en).

## 3. Rewrite recommendation as next-token generation

The page uses fixed logits for immediate feedback. The notebook estimates a smoothed toy token model from generated Semantic IDs. It is not a Transformer, but preserves the central decoding computation:

$$
\log p(c_{1:M})=\sum_{m=1}^{M}\log p(c_m\mid c_{<m},x)
$$

- **Temperature** controls distribution sharpness.
- **Beam width** controls how many prefixes survive each level.
- **Decode depth** reveals how errors accumulate through the hierarchy.
- A **complete prefix**, rather than an isolated token, is the catalog address.

This connects [DSI](/blog/generative-recommendation-10-dsi-en), [GPTRec](/blog/generative-recommendation-12-gptrec-en), and [OneRec](/blog/generative-recommendation-17-onerec-en).

## 4. Why the Trie is not decoration

Even when every token belongs to the legal vocabulary, their complete sequence may name no product. An unconstrained model can splice high-probability local choices from different items. A Trie keeps only tokens that can still lead to at least one catalog item:

$$
\mathcal{A}(c_{<m})=
\{c\mid \exists i,\ \operatorname{SID}(i)_{1:m}=c_{<m}\oplus c\}
$$

The page lets you disable the constraint directly. The notebook colors valid and invalid beam candidates, then samples across several temperatures. The default experiment exposes many out-of-catalog samples without constraints and exactly zero invalid IDs with the Trie.

The Trie guarantees that an item exists; it does not guarantee that the item suits the user. It is a decoding-validity layer, not a relevance model.

## 5. Do not keep only Recall@K

| Layer | Minimum metrics | Misreading prevented |
|---|---|---|
| Recommendation | Recall@K, NDCG@K, cohort and head/tail slices | A model can generate fluently and recommend poorly |
| Tokenizer | Reconstruction error, collision rate, codebook usage, prefix load | Beautiful reconstruction does not imply usable IDs |
| Decoding | Invalid-ID rate, catalog coverage, latency, beam cost | Top-1 can look normal while the long beam is broken |
| Generalization | Unseen items, cold start, temporal shift, catalog mutation | Memorizing the training catalog is not generalization |

The notebook ends with assertions that can fail: BPR loss must decline, residuals must shrink, IDs must have fixed length, and constrained decoding must contain no invalid item. Small checks make the artifact closer to reproducible research than a collection of attractive plots.

## 6. A useful edit order

Run everything once with defaults, then change one variable at a time:

1. compare temperatures `0.4` and `1.4`;
2. compare beam widths `4` and `32`;
3. change codebook size to `3` or `6`;
4. change the number of codebooks to `2` or `5`;
5. replace the BPR item embedding with a content encoder or sequence model output.

Steps 3–4 connect to LETTER and CARE. Step 5 opens the path toward LLaRA, HSTU, ContRec, and DIGER.

## 7. What this tutorial does not pretend to reproduce

Neither the page nor the notebook is a benchmark reproduction of TIGER, OneRec, or DIGER. They omit a production-scale sequence Transformer, end-to-end tokenizer learning, load-balancing losses, distributed beam search, online latency, and A/B testing.

What they provide is an inspectable skeleton: every intermediate representation can be printed, every decoding decision can be edited, and every validity assumption can be turned off. Use the page below to build intuition, then open the notebook for the complete code path.
