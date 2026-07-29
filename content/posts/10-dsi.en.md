---
title: "What If Retrieval Generated IDs Instead of Looking Them Up? | Generative Recommendation 10: DSI"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - DSI
  - Generative Retrieval
pairKey: "generative-recommendation-10-dsi"
slug: "generative-recommendation-10-dsi-en"
excerpt: "DSI writes documents into model parameters and trains one Transformer on document-to-ID and query-to-ID mappings, turning retrieval into conditional ID generation."
series: "generative-recommendation"
seriesOrder: 10
draft: false
---

A conventional search engine resembles a library: put every book in an index, then look up the nearest books when a query arrives.

[LLaRA in the previous chapter](/blog/generative-recommendation-09-llara-en) made behavioral embeddings readable to an LLM, but the candidate catalog still lived outside the model. The LLM could understand; another system still had to retrieve. [Differentiable Search Index (DSI)](https://arxiv.org/abs/2202.06991) asked a much more radical question:

> Can the index live in Transformer parameters, and can a query retrieve a document by generating its ID?

DSI is a document-retrieval paper, not a recommender-system paper. Yet it supplied the decisive operation behind later generative recommenders: **represent a retrievable object by an identifier sequence and generate that sequence directly.**

## The paper in 30 seconds

1. **The old boundary:** a model could encode intent, but a separate index still owned the candidate set and returned objects.
2. **DSI's answer:** train one encoder–decoder on two mappings—`document → docid` for indexing and `query → docid` for retrieval—so document knowledge is stored in its parameters.
3. **The result and its caveat:** on NQ320K's 228K unique documents, DSI-XXL with semantic-string IDs reached 40.4 Hits@1 versus 24.3 for a T5-XXL dual encoder. That number came from the minimum-forgetting checkpoint; the maximum-forgetting checkpoint scored only 12.2.

![From index lookup to ID generation](/blog/generative-recommendation/10-dsi/paradigm-shift.svg)

*Figure 1. DSI changes the retrieval interface; it is more than replacing one encoder with another.*

## The problem inherited from the LLM branch

Take our running history:

```text
tennis racket → climbing shoes → badminton racket → swim cap
```

A standard recommender creates a user vector and scores all items, or searches an approximate-nearest-neighbor index. Even an LLM-backed stack normally ends with:

```text
understand intent → produce query embedding → search index → rerank
```

Understanding and inventory therefore live in different places. DSI collapses part of that split into:

\[
P(\text{target ID}\mid \text{input})
\]

The unification is not free. An external index is easy to rebuild, insert into, and delete from. Knowledge written into neural parameters is difficult to update and can be forgotten. DSI opened a research path precisely by exposing this trade-off.

## A differentiable index with eight items

We keep the same toy catalog throughout the series:

| ID | Item | Short content |
| --- | --- | --- |
| I01 | Tennis racket | Carbon, control |
| I02 | Tennis balls | Durable training balls |
| I03 | Climbing shoes | Indoor bouldering |
| I04 | Chalk bag | Grip aid |
| I05 | Badminton racket | Light, attacking |
| I06 | Goggles | Anti-fog training |
| I07 | Swim cap | Silicone, long hair |
| I08 | Sports towel | Quick-dry |

First create an **indexing** example:

```text
input: anti-fog training goggles
target: 6
```

Then create a **retrieval** example:

```text
input: my eyes get wet when swimming; I need anti-fog gear
target: 6
```

The inputs differ but share the same label. The first binds content to an address; the second learns to route a query to that address. DSI uses documents rather than products, but the mechanism is identical.

![Writing eight objects into the model](/blog/generative-recommendation/10-dsi/toy-index.svg)

*Figure 2. Indexing is not a one-off forward pass whose vectors are saved; it is supervised optimization that changes model parameters.*

## The ID is part of the model

DSI evaluates three document-ID designs.

### Atomic IDs

Every document owns one token such as `<doc_228031>`. Retrieval takes one decoding step, but the output softmax grows linearly with the collection and a new document requires a new token.

### Naive string IDs

An arbitrary integer is broken into character or number tokens. The vocabulary is small, but shared prefixes have no meaning.

### Semantic string IDs

DSI embeds documents with an eight-layer BERT and recursively applies k-means. The paper uses branching factor 10 and stops when a node contains fewer than 100 documents. Cluster choices along the path become the ID:

```text
goggles: 〈sports, water, swimming, eyes〉
swim cap:〈sports, water, swimming, head〉
```

Early tokens make coarse decisions and later tokens refine them. Similar documents share prefixes, so a compact vocabulary also encodes a search hierarchy. This idea becomes the foundation of TIGER's Semantic IDs.

## From input to output

DSI is a T5 encoder–decoder. For input length \(n\), ID length \(m\), and hidden width \(d\):

```text
input IDs       [B, n]
encoder states  [B, n, d]
decoder prefix  [B, t]
next-token logits [B, |V_id|]
```

Indexing and retrieval share every parameter. For a semantic-string ID, the decoder autoregressively emits \(j_1,j_2,\ldots,j_m\).

![DSI training and retrieval](/blog/generative-recommendation/10-dsi/mechanism.svg)

*Figure 3. The two tasks jointly create an index and a retriever. Without the indexing mapping, an ID has no referent.*

## The central equation

Write either a document or query as \(x\), and a document ID as the token sequence \(j=(j_1,\ldots,j_m)\):

\[
\mathcal L_{\text{DSI}}
=-\sum_{(x,j)}\sum_{t=1}^{m}
\log P_\theta(j_t\mid j_{<t},x).
\]

- \(x\) can be the product text “anti-fog training goggles” or a query asking for that product;
- \(j_t\) is the address component at level \(t\);
- \(j_{<t}\) is the prefix already generated;
- \(\theta\) is the shared Transformer memory and retriever.

The loss is ordinary cross-entropy. The conceptual leap lies in the task definition: class labels become compositional addresses, and index construction becomes parameter learning.

## Training and inference

The paper compares sequential training—index first, then retrieval—with multitask co-training. Mixing both kinds of examples works better because switching entirely to query supervision lets the model forget document–ID bindings.

```text
for batch in mix(index_examples, retrieval_examples):
    x = document_tokens if batch.is_index else query_tokens
    target = docid_tokens
    update(cross_entropy(model(x), target))

retrieve(query):
    return beam_search(model, query)
```

The principal experiments did not constrain decoding to valid IDs. A model can therefore emit a sequence belonging to no document. Later generative recommenders usually build a catalog trie and mask illegal next tokens.

## What the experiments establish

NQ320K contains roughly 320K query–document pairs and 228K unique documents. Table 2 gives the headline comparison:

![Retrieval results on NQ320K](/blog/generative-recommendation/10-dsi/evidence.svg)

*Figure 4. The Hits@1 gain demonstrates substantial parametric indexing capacity; the smaller Hits@10 gap shows that the advantage is not uniform across ranks.*

| Method | Hits@1 | Hits@10 |
| --- | ---: | ---: |
| BM25 | 11.6 | 34.4 |
| T5-XXL dual encoder | 24.3 | 67.3 |
| DSI-XXL semantic string | **40.4** | **70.3** |

The appendix changes how this result should be read. Validation documents were appended to the training stream and shuffled through a finite buffer. Checkpoints therefore differ sharply in forgetting:

| Semantic-string DSI-XXL checkpoint | Hits@1 | Hits@10 |
| --- | ---: | ---: |
| Minimum forgetting | **40.4** | **70.3** |
| Average forgetting | 26.3 | 50.2 |
| Maximum forgetting | 12.2 | 30.1 |

The defensible conclusion is not “a Transformer permanently replaces an index.” It is: **a Transformer can perform differentiable indexing, but stable writes and continual updates remain unsolved.**

Another ablation reinforces the mechanism. Removing the indexing task yields 0 Hits@1. Direct document-to-ID supervision works, while Targets2Inputs and span-corruption alternatives do not produce meaningful retrieval. Language similarity alone does not reveal what an arbitrary identifier denotes.

## Where DSI breaks

### Scale

The largest collection contains 228K unique documents, far below the tens of millions or billions in major catalogs. The paper does not establish how capacity, decoding, or collisions scale.

### Mutable inventory

Adding a row to an index is cheap; adding a document to DSI requires optimization. Deletion becomes machine unlearning rather than removing a posting. Catastrophic forgetting, version consistency, and compliant deletion are fundamental system problems.

### Checkpoint sensitivity

The gap between minimum- and maximum-forgetting checkpoints means a best result can describe a moment rather than a stable capability.

### Validity

Unconstrained generation can produce nonexistent IDs. A trie fixes syntax but reintroduces catalog state, beam-search cost, and strict synchronization between model and inventory.

### Retrieval is not recommendation

DSI does not model long behavioral histories, exposure bias, repeat consumption, list objectives, or business rules. It is the generative-retrieval ancestor, not an end-to-end production recommender.

## Why TIGER comes next

DSI shows that IDs can be generated, but leaves three recommendation-specific questions:

1. Can item addresses be learned from content instead of built by hierarchical clustering?
2. Can similar products share prefixes while every real product remains uniquely addressable?
3. Can a new item receive a retrievable ID before it accumulates interactions?

[TIGER, in the next chapter](/blog/generative-recommendation-11-tiger-en), brings the idea into sequential recommendation. It compresses item content into multi-level Semantic IDs with an RQ-VAE, then trains a generative model to write the next item's semantic address.

---

**Critical takeaway:** DSI's deepest contribution is not a single Hits score. It reframes retrieval as learning a language of object IDs. It also reveals that every such language needs an address design, an update policy, and a defense against forgetting—the three problems that drive the next five papers.
