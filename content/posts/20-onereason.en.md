---
title: "Writing a Chain of Thought Is Not Reasoning | Generative Recommendation 20: OneReason"
date: "2026-07-29"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - OneReason
  - Recommendation Reasoning
pairKey: "generative-recommendation-20-onereason"
slug: "generative-recommendation-20-onereason-en"
excerpt: "OneReason decomposes recommendation reasoning into item perception and preference cognition, then uses hierarchical tasks, specialize-then-unify RL, and fast–slow deployment to make thinking useful."
series: "generative-recommendation"
seriesOrder: 20
draft: false
---

The previous episode left an uncomfortable but important result:

> Giving a recommender more “thinking tokens” does not guarantee a better recommendation.

[OneRec-Think](/blog/generative-recommendation-19-onerec-think-en) already aligned item codes, supervised textual chains of thought, and applied GRPO. Why could thinking still lose to direct decoding?

Recommendation is not a proof. A proof has a verifiable deductive path. A user may next buy a wrist guard, watch a swimming lesson, click a recovery routine, or do nothing. The model observes an incomplete history and several valid futures, then infers a latent intent that could explain them. That is closer to **abductive reasoning**.

The [OneReason Technical Report](https://arxiv.org/abs/2606.06260), released in June 2026 and explicitly marked work in progress, diagnoses two prerequisites:

1. **Perception:** the model actually knows what itemic tokens represent.
2. **Cognition:** it can reorganize a long, contradictory history into persistent traits, interest evolution, and current intent.

This finale therefore treats a fluent CoT as an interface—not as proof of reasoning—and asks whether the intermediate process improves the recommendation.

## The paper in 30 seconds

1. **The previous limitation:** CoT alone is not reasoning. In OneReason's SFT results, thinking mode has lower Recall@64 than non-thinking mode in video, product, advertising, and live streaming.
2. **The answer:** four-granularity pre-training establishes perception; R0 perception, R1 derivation, R2 evolution, and R3 recommendation scaffold cognition; domain specialists followed by unified RL make traces lead to positive items.
3. **The evidence:** after rejection-sampling fine-tuning, thinking beats non-thinking in all four domains. A ten-day online test reports +10.332% impressions and +8.234% revenue for the combined direct-retrieval and OneRec-enhancement deployment. The private industrial evidence is strong, but the report does not establish causal faithfulness of its explanations.

![From more thinking tokens to perception and cognition](/blog/generative-recommendation/20-onereason/paradigm-shift.svg)

*Figure 1. The crucial change is not a longer CoT. The model must first perceive items, then compress history and judge how interests evolve.*

## Recommendation reasoning with eight items

Reuse the series' eight-item world:

| ID | Item | Three-level itemic pattern |
| --- | --- | --- |
| A | tennis racket | `<sport><racket><tennis>` |
| B | tennis ball | `<sport><ball><tennis>` |
| C | climbing shoes | `<sport><shoe><climb>` |
| D | climbing chalk | `<sport><gear><climb>` |
| E | badminton racket | `<sport><racket><badminton>` |
| F | shuttlecock | `<sport><ball><badminton>` |
| G | swim cap | `<sport><wear><swim>` |
| H | quick-dry towel | `<sport><gear><swim>` |

The user's observed path is:

```text
buy tennis racket → view climbing shoes → search badminton racket → buy swim cap
```

Suppose H is the next logged item. A weak trace says:

```text
The user likes sports and recently bought a swim cap,
so recommend a quick-dry towel.
```

It sounds plausible but ignores three questions. Is the user repeatedly switching sports or persistently exploring light equipment? Is the cap a stable preference or a one-off travel need? Could conflicting behavior come from a shared device?

OneReason wants a layered computation:

```text
R0 Perception: <sport><wear><swim> means a swim cap, not an arbitrary ID.
R1 Derivation: a cap and towel are complementary within a swimming context.
R2 Evolution: tennis → climbing → badminton → swimming suggests exploration;
              the recent swimming signal outweighs old racket behavior.
R3 Recommendation: generate H at the intersection of exploration and current context.
```

![Two-axis compression and three cognitive levels](/blog/generative-recommendation/20-onereason/toy-reasoning.svg)

*Figure 2. Reasoning should not translate every event into prose. It should compress persona, extract temporal evolution, and then build local item relations.*

## Step one: make item codes perceptible

### A multimodal item tokenizer

OneReason uses visual, language, and audio signals to build a dense item embedding, then applies three-layer residual-quantization K-Means with 8,192 codes per layer:

\[
s_i=[d_i,c_i^{(1)},c_i^{(2)},c_i^{(3)}].
\]

\(d_i\) is a domain token such as `video`, `prod`, `ad`, `living`, or the generic `sid`. The remaining tokens encode progressively finer residual content. Unlike earlier Semantic ID formats, the domain is explicit and no end token is required.

Quantization can place similar content near one another without making the address intelligible to a language model. OneReason therefore mixes four granularities of pre-training:

- **Token level:** explain subcodes and their compositional semantics;
- **Item level:** translate between a full pattern and descriptions, questions, or multimodal content;
- **Relation level:** explain why two items connect, adding common sense and collaborative transfer;
- **User level:** map behavior to personas and interests, and generate plausible behavior from a persona.

General text and multimodal examples remain in the mixture. If `<prod><17><903><44>` has no recoverable content meaning, a later CoT merely invents a story about unknown symbols.

## Step two: scaffold cognition

Supervised tasks form four levels:

| Level | Question | Eight-item example |
| --- | --- | --- |
| R0 Perception | What is this code? | identify G as a swim cap |
| R1 Derivation | Why are two items related? | G and H complement a swimming session |
| R2 Evolution | How does interest change? | rackets give way to swimming |
| R3 Recommendation | What comes next after integration? | generate H's itemic pattern |

R3 training imposes an important constraint: the target item may not appear in the reasoning trace. The model must cite historical evidence and reveal the answer only at the end, reducing direct answer leakage.

### Two-axis compression

Event-by-event summaries spend tokens while treating noise as evidence. The report first compresses along two axes:

1. **Persona compression:** stable categories, life stage, rhythm, price sensitivity, interaction depth, and conflicts that may signal shared devices;
2. **Interest-evolution compression:** trigger, expansion, narrowing, continuation, saturation and substitution, cross-domain echoes, and browse-to-purchase closure.

R1 then builds local bridges, R2 judges temporal change, and R3 generates the answer. Summary and decision are deliberately separated: persona says who the user tends to be; evolution decides which tendency should dominate now.

![OneReason from pre-training to specialists and unification](/blog/generative-recommendation/20-onereason/mechanism.svg)

*Figure 3. Four-granularity pre-training establishes perception, R0–R3 SFT creates a cognitive scaffold, and domain-specific RL turns plausible prose into successful retrieval.*

## Step three: specialize, then unify

Directly mixing video, product, ad, and live-stream RL creates interference:

- video preferences change quickly and have many valid targets;
- purchases are sparse and unfold over longer horizons;
- advertising is also shaped by auctions and exposure policy;
- live streaming depends strongly on time and creator state.

OneReason starts from one SFT checkpoint and applies GRPO separately to each domain, creating four teachers. It then reunifies them through two routes.

### RFT: learn only from successful trajectories

Domain teachers generate rollouts. High-quality traces that hit positive items survive rejection sampling and fine-tune one student. This is stable and simple, but it discards failures and may retain few long-tail examples.

### MOPD: absorb several teachers on the student's trajectories

Multi-Teacher On-Policy Distillation lets the unified student sample and uses the relevant teacher–student log-probability difference as a token-level advantage:

\[
\hat A_t^{\text{MOPD}}
=\operatorname{sg}\!\left[
\log \pi_{d}(y_t\mid x,y_{<t})
-\log \pi_{\theta}(y_t\mid x,y_{<t})
\right].
\]

Tokens that a domain teacher believes much more strongly receive positive pressure. An information-gain filter removes many already-aligned samples that would otherwise dilute rare expertise. MOPD can learn from failures and tail trajectories, but its ceiling is bounded by the teachers and optimization is more involved.

## The central reward: hit positives without collapsing candidates

There is rarely one valid recommendation. For each CoT, OneReason produces \(K\) itemic sequences. With \(N\) reasoning paths and \(K\) answers per path, training obtains \(N\times K\) answer rollouts while paying for only \(N\) long traces.

The reward for answer \(j\) under reasoning path \(i\) is:

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

In the toy example:

- \(C_u^+\) is the user's logged positive set, potentially containing more than H;
- \(c_{i,j}\) is answer \(j\) from reasoning path \(i\);
- \(m_i^{(1)}\) counts distinct first-level semantic codes among its \(K\) answers;
- \(R_{\text{rule}}\) requires a positive hit;
- \(R_{\text{div}}\) penalizes copying nearly identical addresses.

A path that outputs four near-duplicate swimming towels may hit but has low diversity. A path spanning towels, goggles, and recovery equipment earns a stronger signal if it also hits a positive. The reward still judges outcomes rather than textual truth, but it recognizes both multi-valid futures and candidate collapse.

## Training and serving

The full recipe is:

```text
multimodal items → three-layer RQ-KMeans codes
                ↓
four-granularity itemic pre-training + general data
                ↓
R0–R3 SFT for thinking and non-thinking modes
                ↓
four single-domain GRPO teachers
                ↓
RFT or MOPD unification
```

Inference has two modes:

- **Thinking:** generate compressed persona and evolution plus R1/R2 bridges, then the itemic answer; this costs more but exposes an intermediate interface.
- **Non-thinking:** decode item codes directly; it is faster, and the report finds that some benefits of CoT supervision transfer to direct decoding.

Production adds Fast–Slow Thinking:

- **Slow:** OneReason performs nearline generative retrieval directly.
- **Fast:** OneReason outputs become embeddings or Thinking Tokens consumed by online OneRec.

The industrial reasoning system does not write a long essay for every request. A slow model periodically reorganizes interest; a millisecond-level model consumes the compressed conclusion. This extends OneRec-Think's Think-Ahead idea beyond cached code prefixes.

## What the evidence establishes

The most informative result is not the largest model beating a baseline. It is the reversal between thinking and non-thinking.

![Thinking reverses after RFT](/blog/generative-recommendation/20-onereason/evidence.svg)

*Figure 4. Recall@64 is reported in percent. After SFT alone, extra CoT hurts in every domain. Thinking becomes consistently better only after RFT.*

| Training | Mode | Video | Product | Ad | Live |
| --- | --- | ---: | ---: | ---: | ---: |
| SFT | non-thinking | **0.11** | **2.96** | **6.49** | **15.52** |
| SFT | thinking | 0.06 | 1.65 | 3.41 | 14.32 |
| RFT | non-thinking | 0.19 | 3.96 | 7.26 | 18.17 |
| RFT | thinking | **0.24** | **4.19** | **7.50** | **18.35** |

The table supports three claims:

1. **CoT is not a free gain:** SFT thinking is worse in all four domains.
2. **Outcome feedback calibrates reasoning:** only after RFT does thinking win consistently.
3. **The direction is consistent but margins can be small:** the ad and live gains are 0.24 and 0.18 percentage points, not a dramatic leap.

In the cross-domain comparison, OneReason-RFT-thinking reaches Recall@64 of 0.24%, 4.19%, 7.50%, and 18.35%, versus 0.13%, 3.00%, 6.55%, and 16.70% for LC-Rec-PT-SFT-8B. Target-set sizes differ substantially, so the columns should not be read as direct domain difficulty comparisons.

A ten-day online A/B test with 5% treatment and control traffic per local service reports:

| Deployment | Impressions | Revenue |
| --- | ---: | ---: |
| OneReason direct retrieval | +0.940% | +4.528% |
| OneReason enhancing OneRec | +6.831% | +4.636% |
| Combined | **+10.332%** | **+8.234%** |

This is strong system-level evidence that slow retrieval and fast-model enhancement complement each other. It does not isolate visible CoT: the gain may also come from a stronger tokenizer, more data, domain RL, greater compute, and a new retrieval channel.

## Where OneReason still fails

### It is an unfinished technical report

Version 1 appeared in June 2026 and is explicitly labeled work in progress; model release is described as future work. The private training corpus, four-domain benchmark, and production stack are not independently reproducible. Treat the numbers as a team report, not an externally settled result.

### The reward verifies answers, not reasoning

A trace can contain false claims, omit decisive behavior, or luckily lead to the right code and still receive reward. The paper shows that a policy with reasoning can retrieve better; it does not show that every sentence is the causal path to its decision.

### Logs are incomplete preferences

\(C_u^+\) contains observed positives after an exposure policy. Good unseen items and equally valid unrealized futures receive no reward. An abductive problem is still compressed into partial offline labels.

### Business metrics are not user welfare

Impressions and revenue matter, but they do not directly measure satisfaction, diversity, long-term trust, or ecosystem health. A more capable reasoning optimizer can also optimize short-term inducement more effectively.

### Slow thinking is still moved nearline

Fast–Slow deployment honestly acknowledges CoT latency. Production gains come from system layering, not an 8B model deliberating on every live request. Cache staleness, abrupt interest shifts, and model-version consistency remain engineering risks.

## What twenty papers taught us

This story was never simply “make the Transformer larger”:

1. [BPR](/blog/generative-recommendation-01-bpr-en) formalized implicit preference as a positive item outranking a negative one.
2. GRU4Rec, SASRec, and BERT4Rec progressively modeled order, selected relevant history, and recovered missing behavior with bidirectional context.
3. P5 through LLaRA put users, items, and tasks into a language model, exposing the mismatch between language and collaborative spaces.
4. DSI through ETEGRec turned items into autoregressively generated Semantic IDs, introducing quantization collisions, catalog updates, and constrained decoding.
5. HSTU, OneRec, and MTGR pushed the paradigm into trillion-parameter scaling, session objectives, and industrial multitask systems.
6. OneRec-Think and OneReason finally asked whether a model can form a useful, intervenable, verifiable intermediate cognition before generating the right address.

Generative recommendation's “past life” was ranking: score a supplied candidate set. Its present is generation: fold retrieval, ranking, and sometimes explanation into one token space. Its next phase depends on five unresolved questions:

- **Mutable identity:** how should Semantic IDs be versioned when products change, arrive, or disappear?
- **Counterfactual learning:** how can one exposure log reward several valid futures?
- **Faithful reasoning:** if part of a CoT is edited, does the answer change predictably?
- **The cost of scale:** do training, nearline deliberation, caches, and fast online models repay their compute?
- **Who chooses the objective:** what wins when clicks, revenue, satisfaction, diversity, and long-term ecosystem health conflict?

The series ends here; the genuinely open problems begin here.

Return to the [Twenty Papers to Understand Generative Recommendation series page](/series/generative-recommendation/en) to reread the four-act story, or trace any unresolved question backward to the paper that made it visible.
