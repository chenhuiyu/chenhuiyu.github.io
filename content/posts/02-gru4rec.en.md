---
title: "A User Is Not a Point, but a Story in Motion | Generative Recommendation 02: GRU4Rec"
date: "2026-07-26"
updated: "2026-07-29"
category: "Generative Recommendation"
language: "en"
tags:
  - Generative Recommendation
  - Recommender Systems
  - GRU4Rec
pairKey: "generative-recommendation-02-gru4rec"
slug: "generative-recommendation-02-gru4rec-en"
excerpt: "BPR compresses a user into one static vector. GRU4Rec instead treats a visit as a state that changes after every click and predicts the next item at each step."
series: "generative-recommendation"
seriesOrder: 2
draft: false
---

Lin normally plays tennis, but tonight she has inspected climbing shoes, chalk bags, and finger tape in quick succession.

A system that only remembers her long-term tennis preference may recommend another racket. A system that understands **the behavior unfolding right now** may suggest a bouldering brush or climbing trousers.

[BPR, the previous article](/blog/generative-recommendation-01-bpr-en), learned that a positive item should outrank a negative one, but compressed every user into a fixed vector $\mathbf w_u$. To it, `tennis → climbing` and `climbing → tennis` are almost the same set of interactions.

[Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939), better known as GRU4Rec, changes the definition of the user:

> In session-based recommendation, identity may be unavailable. The user representation at this moment is the hidden state formed by the session so far.

Recommendation begins to move seriously from static matching toward next-step prediction.

## Understand the paper in 30 seconds

1. **The limitation left by BPR:** a static user vector erases event order, and an anonymous or brand-new session may have no learned user vector at all.
2. **GRU4Rec’s answer:** read clicks in order with a GRU, update hidden state $\mathbf h_t$ at every step, and produce scores for the next item.
3. **The decisive evidence:** on the real RSC15 and VIDEO session datasets, a single-layer GRU substantially improves Recall@20 and MRR@20 over a strong item-KNN baseline. The paper also makes the model trainable through session-parallel batches, in-batch negatives, and ranking losses.

![From a static user vector to a dynamic session state](/blog/generative-recommendation/02-gru4rec/paradigm-shift.svg)

*Figure 1. BPR-MF represents a user with one long-term vector. GRU4Rec represents what the user is doing now with a state that changes after each click.*

## What problem did the previous paper leave behind?

BPR-MF scores an item as

$$
s(u,i)=\mathbf w_u^\top\mathbf h_i.
$$

The user embedding $\mathbf w_u$ must be learned from past interactions. That makes sense for a long-lived account in a Netflix-like setting, but many real requests have no stable identity:

- an e-commerce visitor is not logged in;
- a news reader changed device or cleared cookies;
- a user is visiting a small site for the first time;
- two visits by one person have unrelated intentions;
- or the product only needs to predict the next action in this session.

Production systems often handle this with item-to-item co-occurrence: after the user views climbing shoes, recommend products frequently seen with climbing shoes. The method is simple, fast, and often strong, but it usually reduces the session to its most recent event.

GRU4Rec asks a precise question:

> Can a recommender retain item-KNN’s sensitivity to the current action while remembering more than the last click?

## Turn one session into supervised examples

Suppose Lin’s visit contains

```text
I01 racket → I03 climbing shoes → I05 shuttlecock → I06 goggles
```

Shifting a length-four sequence by one position creates three targets:

```text
input I01, target I03
input I03, target I05
input I05, target I06
```

Each target, however, is conditioned on more than the current input. When predicting `I06`, the GRU state $\mathbf h_3$ has processed `I01, I03, I05` in order.

![A session becomes several next-item targets](/blog/generative-recommendation/02-gru4rec/toy-sequence.svg)

*Figure 2. The label is only the next item, but the hidden state summarizes the ordered session from its beginning to the current position.*

Two ingredients needed by later generative recommenders have now appeared:

1. **Item IDs are discrete symbols in a sequence.**
2. **The current output is conditioned on ordered past behavior.**

GRU4Rec is not yet a modern autoregressive slate generator. Common inference still scores many candidate items and takes Top-K. But recommendation is beginning to resemble language modeling: shift the history and predict the next token.

## What happens from model input to output?

### 1. Input: the original paper uses 1-of-N

With $N$ items, the current click $i_t$ is a one-hot vector

$$
\mathbf x_t\in\{0,1\}^N.
$$

A contemporary implementation would usually look up a $d$-dimensional item embedding first. The original paper reports that an additional embedding layer performed slightly worse, so it retained 1-of-N inputs. It is worth separating what a historical paper actually tested from how a modern reimplementation is likely to look.

### 2. GRU: gates control memory

A vanilla RNN combines the current input and old state at every step and can suffer from vanishing gradients over long sequences. A GRU adds two gates:

- the **update gate $\mathbf z_t$** controls how much new information is written;
- the **reset gate $\mathbf r_t$** controls how much old state is used when constructing the candidate state.

The final interpolation is

$$
\mathbf h_t
=(1-\mathbf z_t)\odot\mathbf h_{t-1}
+\mathbf z_t\odot\tilde{\mathbf h}_t.
$$

When $\mathbf z_t$ is near zero, the old state persists. When it is near one, the candidate state dominates.

### 3. Output: preference scores for the next item

The hidden state passes through an output layer to produce scores $\hat r_{t,i}$. A full output can cover all $N$ items, but that is expensive for a large catalog. The paper therefore computes the necessary scores for the true next item and a sampled subset.

![The GRU4Rec session-parallel training flow](/blog/generative-recommendation/02-gru4rec/mechanism.svg)

*Figure 3. Several sessions advance together. The GRU updates their states, and target items from the other sessions become convenient negatives.*

For batch size $B$ and hidden width $d$, the core tensors can be read as:

| Tensor | Shape | Meaning |
|---|---:|---|
| current items | original paper: $[B,N]$ | one-hot inputs for $B$ active sessions |
| hidden state | $[B,d]$ | one dynamic state per session |
| true next items | $[B]$ | shifted targets |
| in-batch scores | $[B,B]$ | one positive per row; other columns can be negatives |

## Unpack the central equation

The complete GRU update contains an update gate, a reset gate, and a candidate state:

$$
\mathbf z_t
=\sigma(W_z\mathbf x_t+U_z\mathbf h_{t-1}),
$$

$$
\mathbf r_t
=\sigma(W_r\mathbf x_t+U_r\mathbf h_{t-1}),
$$

$$
\tilde{\mathbf h}_t
=\tanh\left(
W\mathbf x_t
+U(\mathbf r_t\odot\mathbf h_{t-1})
\right),
$$

$$
\mathbf h_t
=(1-\mathbf z_t)\odot\mathbf h_{t-1}
+\mathbf z_t\odot\tilde{\mathbf h}_t.
$$

Mapped to Lin’s behavior:

- after `I03 climbing shoes`, the reset gate can reduce tennis-related state that is unhelpful for the current intent;
- the update gate determines how strongly “climbing” should rewrite the session;
- if `I05 shuttlecock` is a casual detour, the state may retain part of the climbing signal;
- the prediction of `I06 goggles` depends on accumulated $\mathbf h_3$, not on `I05` alone.

The gates are not human-readable “tennis” and “climbing” switches. They are continuous values across dimensions. Assigning a semantic interest to one neuron would over-interpret the mechanism.

## Three engineering choices that make it trainable

GRU4Rec’s contribution is more than inserting a GRU into recommendation. The paper develops a workable recipe for sparse targets, large item vocabularies, and variable-length sessions.

### 1. Session-parallel mini-batches

Language models can often batch fixed-length fragments. Session lengths vary from two events to hundreds. The paper:

1. opens the first $B$ sessions in $B$ batch slots;
2. advances every active session by one event per step;
3. replaces a finished session with the next available one;
4. resets the corresponding hidden-state row during replacement.

This preserves complete sessions while enabling efficient matrix operations.

### 2. Other targets in the batch become negatives

If session $b$ has true next item $i_b^+$, the targets $i_{b'}^+$ from other sessions are convenient negatives for row $b$. One matrix multiplication yields $B\times B$ scores without a separate sampler.

Popular items appear as batch targets more often, so this approximates popularity-based sampling. It is computationally elegant, but a target from another session may be a false negative for the current one.

### 3. Ranking losses

The paper evaluates BPR and introduces TOP1. Its BPR form is

$$
\mathcal L_{\text{BPR}}
=-\frac1{N_S}
\sum_j
\log\sigma(\hat r_{t,i^+}-\hat r_{t,j}).
$$

TOP1 approximates the relative rank of the positive and adds a term that pulls negative scores toward zero:

$$
\mathcal L_{\text{TOP1}}
=\frac1{N_S}\sum_j
\left[
\sigma(\hat r_{t,j}-\hat r_{t,i^+})
+\sigma(\hat r_{t,j}^2)
\right].
$$

The second term matters because an item can be positive for its own row and negative for other rows. Without score control, all outputs can drift upward together.

## Training versus inference

```text
sort sessions and open B active slots
initialize hidden[B, d] = 0

while sessions remain:
    x = current item from every active session
    y = next item from every active session

    hidden = GRU(x, hidden)
    scores = score(hidden, candidate_items=y)
    loss = TOP1_or_BPR(diagonal positives, off-diagonal negatives)
    update parameters

    advance every session
    replace finished sessions and reset their hidden rows
```

Training predicts next events that are already present in the log.

Serving works incrementally:

1. initialize $\mathbf h_0=0$ when a session begins;
2. update $\mathbf h_t$ after every click;
3. score candidate next items;
4. apply availability, duplication, and policy filters before Top-K;
5. update the state again when the next real event arrives.

This stateful design avoids recomputing the entire history for every request. It also means that lost server state, incorrect session boundaries, or cross-device behavior can break the memory.

## What do the experiments actually prove?

The paper evaluates:

- **RSC15:** about 7.97 million training sessions, 31.64 million clicks, and 37,483 items;
- **VIDEO:** about 3 million sessions, 13 million watch events, and 330,000 videos.

Sessions are replayed event by event and measured with Recall@20 and MRR@20. The figure below redraws the comparison between the strongest item-KNN baseline and `TOP1 + 1000 hidden units` from Table 3.

![Original GRU4Rec results against item-KNN](/blog/generative-recommendation/02-gru4rec/evidence.svg)

*Figure 4. GRU4Rec generally leads item-KNN on both metrics and datasets. Values come from Tables 1 and 3 of the paper.*

On RSC15:

- item-KNN achieves **0.5065** Recall@20;
- GRU4Rec reaches **0.6206**, a relative gain of 22.53%;
- MRR@20 rises from **0.2048** to **0.2693**.

The paper also contains useful irregularities:

- BPR loss can underperform item-KNN on VIDEO MRR;
- cross-entropy becomes numerically unstable at the larger hidden size;
- one GRU layer beats deeper alternatives, which the authors tentatively attribute to short sessions without proving the cause;
- an additional item embedding performs slightly worse in this setup.

The supported conclusion is therefore:

> For anonymous, short-session next-item prediction, an ordered state can be more expressive than a strong last-item co-occurrence baseline, and sampling plus loss design is central to making it work.

## Where does it fail?

### 1. All history is compressed into one vector

$\mathbf h_t$ must retain short-term intent, repetition, and longer-range context at once. The earlier an event is, the more recurrent transitions separate it from the output. Gates mitigate the bottleneck; they do not remove it.

### 2. Time cannot be fully parallelized

Within one session, $\mathbf h_t$ waits for $\mathbf h_{t-1}$. Different sessions run in parallel, but a long sequence remains sequential in time, limiting GPU utilization and the path available for long-range learning.

### 3. In-batch negatives can be false

The correct item for another session may also suit the current user. Small batches offer limited candidate coverage, while popular items become negatives unusually often.

### 4. Candidate scoring remains

The model outputs item scores, not paths composed of shared tokens. A full softmax or exhaustive scoring remains expensive for a million-item catalog.

### 5. Session boundaries are product assumptions

How much inactivity starts a new session? Should a cart survive overnight? Are logged-in and anonymous activity merged? A wrong boundary either erases useful state or joins unrelated intents.

### 6. The evaluation belongs to a 2015 setting

Recall@20 and MRR@20 recover the next logged event, not long-term satisfaction. VIDEO ranks only among 30,000 popular candidates. These results cannot be transferred directly to today’s largest catalogs.

## Why does the next paper need to exist?

GRU4Rec turns a static point into a story.

But it tells the story like a whisper chain: step one passes information to step two, which passes it to step three. If an important climbing-shoe purchase occurred 100 steps ago and Lin now views a chalk bag, that signal must survive dozens of recurrent updates.

Perhaps the model should not average all history into one state. Perhaps it should ask:

> Given what I need to predict now, which earlier actions are directly relevant?

Next: [SASRec](https://arxiv.org/abs/1808.09781)—history is long, but only a few steps may matter.

## Original paper

- Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk. [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939). ICLR 2016.

[Return to the complete roadmap →](/series/generative-recommendation/en)
