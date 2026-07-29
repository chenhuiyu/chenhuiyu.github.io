---
title: "Give Every Item a Semantic Address | Generative Recommendation 11: TIGER"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - TIGER
  - Semantic ID
pairKey: "generative-recommendation-11-tiger"
slug: "generative-recommendation-11-tiger-en"
excerpt: "TIGER compresses item content into coarse-to-fine Semantic IDs with an RQ-VAE, then trains T5 to generate the next item's address token by token."
series: "generative-recommendation"
seriesOrder: 11
draft: false
---

[DSI in the previous chapter](/blog/generative-recommendation-10-dsi-en) generated document IDs, but its semantic strings came from hierarchical k-means and its task was document retrieval.

[TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) brings the idea into sequential recommendation. Instead of one enormous, unrelated class token per item, it compresses product content into a coarse-to-fine discrete address:

```text
goggles I06 → 〈water sports, swimming, eye gear, unique suffix〉
```

The model no longer chooses I06 from tens of thousands of flat classes in one step. It writes the item's address one component at a time.

## The paper in 30 seconds

1. **DSI's open problem:** arbitrary IDs have no semantics, atomic output layers grow with the catalog, and hierarchical clustering is separate from the generator.
2. **TIGER's answer:** encode product content with Sentence-T5, learn a three-level residual-quantization ID, append a collision token for uniqueness, and train a T5-like model to generate the next ID from history.
3. **The strongest evidence:** on Beauty, TIGER reaches 0.0321 NDCG@5 versus SASRec's 0.0249, a 29.04% relative gain. Replacing RQ IDs with random IDs reduces it to 0.0205.

![From item classification to semantic addresses](/blog/generative-recommendation/11-tiger/paradigm-shift.svg)

*Figure 1. A multi-token address turns one huge flat classification into a hierarchy of decisions with shared prefixes.*

## Why DSI's address is not enough for products

Catalogs have three special properties.

First, items arrive continually. One atomic token per item makes the embedding table and softmax grow linearly.

Second, a new product has no clicks but does have a title, brand, price, and category. If its identifier can be computed from content, it can enter the address space before the whole recommender is retrained.

Third, items are not isolated labels. Goggles and swim caps should first share a “water–swimming” prefix and diverge only at a fine level. Prefix sharing lets sparse items reuse statistical strength.

TIGER therefore splits the system into:

```text
content → Semantic-ID tokenizer
behavioral history of Semantic IDs → generative recommender
```

This “invent the vocabulary, then learn the sentences” architecture becomes the standard Semantic-ID blueprint.

## Residual quantization with eight items

Assume Sentence-T5 maps our eight products to 768-dimensional content vectors. A single VQ codebook must represent an entire vector at once. Residual quantization instead explains it in stages.

For `I06 goggles`:

1. Level one picks code 17 to explain a broad sports component.
2. Subtract code 17; level two picks code 83 for the remaining water/swimming information.
3. Subtract again; level three picks code 41 for eye gear and anti-fog detail.
4. If another pair of goggles receives the same first three codes, a collision token separates the instances.

```text
I06 goggles  = 〈17, 83, 41, 0〉
I07 swim cap = 〈17, 83, 92, 0〉
```

The first two codes express similarity; all four codes preserve uniqueness.

![Semantic IDs for the toy catalog](/blog/generative-recommendation/11-tiger/toy-semantic-id.svg)

*Figure 2. Shared prefixes emerge from quantizing content vectors. The readable labels in the figure are interpretations, not words seen by the model.*

## The central equation

Let the first residual be the content latent, \(r_1=z\). At level \(l\), choose the nearest entry in codebook \(\{e_k^l\}_{k=1}^{K}\), then pass the unexplained residual forward:

\[
c_l=\arg\min_k\lVert r_l-e_k^l\rVert_2^2,
\qquad
r_{l+1}=r_l-e_{c_l}^l.
\]

After three levels:

\[
\hat z=e_{c_1}^1+e_{c_2}^2+e_{c_3}^3.
\]

Early codes capture broad structure; later codes refine what remains. RQ-VAE also optimizes reconstruction and commitment/codebook terms. The number of possible combinations grows like \(K^L\), while codebook parameters grow only like \(K L\).

TIGER uses three quantization levels, 256 codes per level, and a 32-dimensional latent. A fourth collision position makes the code-token vocabulary \(256\times4=1024\), independent of catalog size.

## From input to output

The history is rewritten as ID sequences:

```text
I01 → 〈12, 07, 44, 0〉
I03 → 〈31, 18, 09, 0〉
I05 → 〈12, 56, 22, 0〉
I07 → 〈17, 83, 92, 0〉
```

These tokens plus a hashed user token enter a T5-like encoder. The decoder autoregressively emits the next item's four codes. The paper's model is about 13M parameters, hashes user IDs into 2,000 user tokens, and truncates history to 20 items.

![TIGER's two-stage mechanism](/blog/generative-recommendation/11-tiger/mechanism.svg)

*Figure 3. The tokenizer learns how to name items; the recommender learns which name follows a history. Original TIGER trains them separately.*

## Training and inference

### Stage 1: learn the tokenizer

Titles, prices, brands, and categories are embedded by Sentence-T5 into 768 dimensions. RQ-VAE compresses them to a 32-dimensional latent and learns three codebooks.

### Stage 2: learn recommendation

Every user sequence supplies next-item examples:

```text
〈SID(I01), SID(I03), SID(I05)〉 → SID(I07)
```

The objective is token-level cross-entropy:

\[
\mathcal L_{\text{rec}}
=-\sum_t\log P_\theta(c_t^{*}\mid c_{<t}^{*},H_u).
\]

### Inference

Beam search produces candidate four-token addresses, which map back to items. The paper reports roughly 0.1%–1.6% invalid IDs in top-10 outputs and filters them with a larger beam. Modern implementations commonly constrain decoding with a catalog trie.

Autoregressive beam search is more expensive than ANN: every code needs a decoder pass and each beam retains multiple branches. The authors explicitly state that efficiency is not the paper's objective.

## What the experiments establish

The three Amazon subsets contain roughly 10K–20K items. The main comparison reports:

| Dataset | Strong baseline NDCG@5 | TIGER | Relative gain |
| --- | ---: | ---: | ---: |
| Sports | S3-Rec 0.0161 | **0.0181** | +12.42% |
| Beauty | SASRec 0.0249 | **0.0321** | +29.04% |
| Toys | SASRec 0.0306 | **0.0371** | +21.24% |

The tokenizer ablation is even more diagnostic:

![The effect of item-ID design](/blog/generative-recommendation/11-tiger/evidence.svg)

*Figure 4. With the generator held conceptually fixed, RQ Semantic IDs beat random and LSH IDs on all three datasets. The address is part of model capacity.*

| Tokenizer | Sports | Beauty | Toys |
| --- | ---: | ---: | ---: |
| Random IDs | 0.0050 | 0.0205 | 0.0270 |
| LSH | 0.0146 | 0.0259 | 0.0299 |
| RQ Semantic IDs | **0.0181** | **0.0321** | **0.0371** |

For cold start, the authors remove interactions for 5% of test items. Content still yields an SID; unseen items sharing the first three codes with a generated candidate can be introduced under an \(\epsilon\) cap. This demonstrates a content-based route into retrieval, although the final rule is engineered rather than a complete cold-start policy learned end to end.

The paper also corrects a subtle preprocessing leak in earlier P5 data: remapping items to consecutive IDs before splitting can reveal test-catalog membership. Generative models can exploit the mere presence of an item in the vocabulary.

## Where TIGER breaks

### The tokenizer sees content, not collaboration

Textually similar products need not attract the same users. The RQ address does not explicitly model co-consumption.

### Two objectives are disconnected

RQ-VAE optimizes reconstruction; the generator optimizes next-item likelihood. A code ideal for reconstructing content may be difficult to predict or poor for ranking, and recommendation loss cannot repair the frozen tokenizer.

### Uniqueness is patched

The collision token guarantees a one-to-one mapping but its value is arbitrary rather than semantic.

### Scale and latency remain unproven

Catalogs are only tens of thousands and history is capped at 20. There is no industrial throughput study for autoregressive beam search.

### IDs are versioned data

Retraining the tokenizer can change every product address. Logs, caches, the online trie, and the generator must migrate together or the same code sequence may denote different items.

## Why GPTRec comes next

TIGER writes one next item as a sequence, but recommendation normally returns a list. Independently scored items cannot condition on earlier list members. A sequential list generator can let item two observe item one and express diversity, coverage, or order.

[GPTRec, in the next chapter](/blog/generative-recommendation-12-gptrec-en), explores both compact multi-token IDs based on SVD and **Next-K generation**, where a recommender writes the list one item at a time.

---

**Critical takeaway:** TIGER's main contribution is not “using T5.” It establishes the generative-recommendation triad: a compositional item vocabulary, a history-to-address objective, and an address-to-catalog decoder. It shifts the central question from how large the recommender is to how an item should be tokenized.
