---
title: "Gated DeltaNet, illustrated: forgetting and updating a fixed-size memory"
date: "2026-11-02"
updated: "2026-11-02"
category: "NLP Insights"
language: "en"
tags: ["Paper Notes", "Gated DeltaNet", "Linear Attention", "LLM Architecture"]
pairKey: "gated-deltanet-visual-guide"
slug: "gated-deltanet-visual-guide-en"
excerpt: "Six original illustrations explain Gated DeltaNet, from KV storage and state updates to chunkwise training and the paper's evidence."
draft: false
---

<img src="/blog/gated-deltanet/cover-en.svg" width="1200" height="940" alt="Gated DeltaNet visual guide: decay the old state, correct its prediction, and read the result.">

After understanding [how KV caching supports continuation](/blog/llm-infra-kv-cache-en), a natural question follows: can we write the history into a fixed-size state instead of retaining every token's keys and values?

That choice moves the problem from storing history to managing finite capacity. How long should old information survive? How should a new value replace an earlier association? What happens when irrelevant writes interfere with useful ones?

*Gated Delta Networks: Improving Mamba2 with Delta Rule* combines two controls: **decay the previous state with a gate, then correct an association using its prediction error.** This guide follows the path from storage to updates, training, and evidence. [1]

Paper: Songlin Yang, Jan Kautz, and Ali Hatamizadeh, ICLR 2025. This reading uses **arXiv v3, dated March 6, 2025**. Mechanism illustrations and worked arithmetic are original; benchmark charts redraw the authors' tables.

## 1. What does a fixed-size state save?

For one layer and one head of standard multi-head attention, a history of $T$ tokens requires $T(d_k+d_v)$ key/value elements. We leave batch size, head count, other caches, and data types aside for now. Architectures such as GQA require their own accounting.

Linear attention offers another organization. Start with the basic form, omitting kernel feature maps and normalization terms:

$$
S_t=S_{t-1}+v_tk_t^\top,
\qquad o_t=S_tq_t,
\qquad S_t\in\mathbb{R}^{d_v\times d_k}.
$$

Expanding the recurrence gives

$$
o_t=\sum_{i=1}^{t}v_i(k_i^\top q_t).
$$

Each outer product $v_i k_i^\top$ adds an association to the matrix. A historical value contributes in proportion to the match between its key and the current query. For fixed dimensions, the matrix contains $d_vd_k$ elements, independent of $T$. [2]

These dot-product coefficients can be negative; they are not softmax probabilities. The attention rule has changed. Standard softmax attention cannot be rearranged into this recurrence by associativity alone.

<img src="/blog/gated-deltanet/storage-en.svg" width="1200" height="930" alt="Per-token KV storage grows with the history; a recurrent state summarizes it in a fixed-size matrix.">

*Figure 1. Two storage organizations. Colors show structure, not measured activations. [Open at full size](/blog/gated-deltanet/storage-en.svg).*

For a single head with $d_k=d_v=128$ and $T=8192$, the KV history contains 2,097,152 elements, versus 16,384 in the matrix. At two bytes per element for both objects, that is approximately 4 MiB versus 32 KiB. **This is accounting for these simplified objects.** Real implementations may retain recurrent states in FP32 and also need short-convolution caches. The ratio does not establish a 128-fold reduction in total model memory.

The deeper change concerns information: individual historical tokens now share a finite-dimensional associative space. Reading the state reflects the combined effects of multiple writes.

## 2. From accumulation to decay and error correction

We use column vectors for $q_t,k_t,v_t$ and a $d_v\times d_k$ state throughout. Keys in the worked examples have unit length.

### Accumulation keeps adding associations

Basic linear attention adds $v_tk_t^\top$. Starting from zero, if the same unit key is written first with $v_a$ and then with $v_b$, reading that key includes $v_a+v_b$. The update does not infer that the second value should replace the first.

### Decay scales the previous state

The paper summarizes the relevant Mamba2 recurrence as follows, setting its full parameterization aside: [1][3]

$$
S_t=\alpha_tS_{t-1}+v_tk_t^\top,
\qquad 0<\alpha_t<1.
$$

A small $\alpha_t$ rapidly reduces the previous state. Within this head and update, the scalar acts on the entire matrix. It makes clearance easy, while also decaying useful old associations.

### The delta rule first asks what the state predicts

For an incoming key $k_t$, the current prediction is $S_{t-1}k_t$. DeltaNet writes the prediction residual: [4]

$$
\begin{aligned}
e_t&=v_t-S_{t-1}k_t,\\
S_t&=S_{t-1}+\beta_te_tk_t^\top.
\end{aligned}
$$

$\beta_t$ controls write strength. For a unit key, multiplying the update by $k_t$ gives

$$
S_tk_t=(1-\beta_t)S_{t-1}k_t+\beta_tv_t.
$$

The readout interpolates toward the new target. At the boundary value $\beta_t=1$, it becomes $v_t$ exactly for this key. The paper primarily uses gates in an open interval; boundary values help explain limiting behavior.

### The gated delta rule decays before computing the residual

Let $\bar S_t=\alpha_tS_{t-1}$. Combining the controls yields

$$
\begin{aligned}
e_t&=v_t-\bar S_tk_t,\\
S_t&=\bar S_t+\beta_te_tk_t^\top\\
&=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)
+\beta_tv_tk_t^\top.
\end{aligned}
$$

This is Equation (10) of the paper. **The prediction inside the residual must also use the decayed state.** Computing $v_t-S_{t-1}k_t$ and adding decay outside produces a different update.

<img src="/blog/gated-deltanet/rules-en.svg" width="1200" height="1010" alt="Four update rules: accumulation, global decay, residual correction, and their combination in Gated DeltaNet.">

*Figure 2. Four rules in one notation. Matrix colors are schematic; Mamba2 shows only the relevant recurrence. [Open at full size](/blog/gated-deltanet/rules-en.svg).*

## 3. Work through one update

Choose a two-dimensional example:

$$
S_{t-1}=\begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
k_t=q_t=\begin{bmatrix}1\\0\end{bmatrix},\quad
v_t=\begin{bmatrix}0\\1\end{bmatrix},\quad
\alpha_t=0.8,\quad\beta_t=0.5.
$$

Decay gives $\bar S_t=0.8I$. Its prediction for the current key is $[0.8,0]^\top$, so the residual is $[-0.8,1]^\top$. Scale it by the write strength and take the outer product with the key:

$$
\Delta S_t=
0.5\begin{bmatrix}-0.8\\1\end{bmatrix}
\begin{bmatrix}1&0\end{bmatrix}
=\begin{bmatrix}-0.4&0\\0.5&0\end{bmatrix}.
$$

The result is

$$
S_t=\begin{bmatrix}0.4&0\\0.5&0.8\end{bmatrix},
\qquad o_t=S_tq_t=\begin{bmatrix}0.4\\0.5\end{bmatrix}.
$$

<img src="/blog/gated-deltanet/update-en.svg" width="1200" height="1110" alt="Worked arithmetic: scale the identity by 0.8, add a residual outer product affecting the first column, then read out 0.4 and 0.5.">

*Figure 3. Decay, correction, and readout. All numbers follow from the specified recurrence, without training a model. [Open at full size](/blog/gated-deltanet/update-en.svg).*

In this deliberately chosen orthogonal basis, the delta term corrects the first column. The second column experiences global decay only. The example makes the controls' different roles visible.

## 4. A targeted update still affects correlated keys

Real representations rarely form such tidy coordinate axes. For another query direction $u$, the correction changes its readout by

$$
\Delta S_tu=\beta_te_t(k_t^\top u).
$$

The delta term vanishes for a direction orthogonal to $k_t$. A correlated direction is affected. Understanding a key-directed update therefore requires understanding the geometry of the keys. Finite state still permits interference; gating changes how that interference is managed.

This suggests a useful reading habit: **when an intuitive explanation says that one association alone changes, multiply the update by another key and inspect the conditions.**

The correction also has an online-regression interpretation. Define

$$
\ell(S)=\tfrac12\lVert Sk_t-v_t\rVert_2^2,
\qquad \nabla_S\ell=(Sk_t-v_t)k_t^\top.
$$

One negative-gradient step from $\bar S_t$ gives $\bar S_t+\beta_t(v_t-\bar S_tk_t)k_t^\top$. The object being updated rapidly is the **runtime state matrix**. The network parameters producing queries, keys, values, and gates remain fixed after training. This explains the fast-weight interpretation and why the order of decay and residual evaluation matters. [1]

## 5. How can a recurrence train efficiently on a GPU?

Maintaining a state is natural during token-by-token decoding. A literal sequential training loop, however, may spend its time on small operations that use the GPU poorly. Another contribution of the paper is a chunkwise reformulation suited to matrix computation. [1, §3.3]

Divide the sequence into chunks of length $C$. Boundary states pass between chunks. Within a chunk, relationships between keys encode update dependencies, allowing matrix operations to combine the results. Define the within-chunk cumulative decay

$$
\gamma_i=\prod_{r=1}^{i}\alpha_r,
\qquad
\Gamma_{ij}=\begin{cases}\gamma_i/\gamma_j&i\ge j\\0&i<j.\end{cases}
$$

Let the rows of $K\in\mathbb{R}^{C\times d_k}$ and $V\in\mathbb{R}^{C\times d_v}$ contain the chunk's keys and values, and let $B=\operatorname{diag}(\beta)$. The paper's transformed writes satisfy

$$
A\widetilde U=BV,
\qquad
A=I+\operatorname{strictLower}\bigl(B(\Gamma\odot KK^\top)\bigr).
$$

$A$ is a $C\times C$ unit lower-triangular matrix. Its structure encodes dependence on earlier positions. The paper uses a WY representation and related transforms to organize these computations, then incorporates the incoming state's contribution into the outputs and the final state.

<img src="/blog/gated-deltanet/chunks-en.svg" width="1200" height="1020" alt="Boundary states pass between chunks; within each chunk, a unit lower-triangular system and matrix multiplications organize computation.">

*Figure 4. Chunkwise training. The triangular system describes transformed writes; complete outputs also incorporate the incoming state. [Open at full size](/blog/gated-deltanet/chunks-en.svg).*

Causal dependencies remain. Their execution changes: dependencies in many small recurrent steps are encoded in structured within-chunk computation. For fixed $C,d_k,d_v$, total work can scale linearly with sequence length, while still paying for within-chunk computation, state propagation, and intermediate tensors.

**A short update equation, a low asymptotic complexity, and fast GPU execution are separate claims.** This guide explains the paper's algorithmic organization; it adds no throughput measurements.

## 6. The update is the core of a larger block

The complete token mixer also includes: [1, §3.4]

- Linear projections, short convolutions, and SiLU on the $q,k,v$ paths, with additional L2 normalization on $q,k$.
- Input-dependent decay and write gates, followed by output normalization, gating, and projection.
- The surrounding residual architecture and SwiGLU MLP.

Implementing only the recurrence therefore does not reproduce the full model.

The paper also studies two hybrids. H1 combines Gated DeltaNet with sliding-window attention; H2 additionally includes Mamba2. Recurrent layers manage compressed state while local attention retains its own reading mechanism. Those attention layers still have caches, so fixed-state claims should identify which layers they describe.

## 7. Read the evidence at the right level

The chart below selects models from Tables 3 and 4. The main experimental setup uses **1.3B parameters, 100B FineWeb-Edu tokens, and 4K training length**, with a 2K attention window for hybrids. Table 4 truncates retrieval inputs to 2K tokens. [1, §4]

<img src="/blog/gated-deltanet/evidence-en.svg" width="1200" height="1300" alt="Author-reported averages: Gated DeltaNet scores 55.32 on commonsense reasoning and 30.6 on retrieval; H1 scores 56.40 and 39.0; H2 scores 56.18 and 40.1. Metrics are plotted separately.">

*Figure 5. Redrawn author-reported values, without independent reproduction. Table 3 averages eight accuracy-based tasks, excluding perplexity columns; Table 4 averages a different set of six retrieval tasks. [Open at full size](/blog/gated-deltanet/evidence-en.svg).*

Compared with Mamba2, pure Gated DeltaNet moves the reasoning average from 54.89 to 55.32, a gain of **0.43 percentage points**. Retrieval moves from 29.8 to 30.6, a gain of **0.8 percentage points**. These comparisons support combining the two mechanisms under the reported setup.

At the same time, the pure recurrent model's retrieval average of 30.6 remains below Transformer++ at 37.0. Attention-containing H1 and H2 reach 39.0 and 40.1. This makes the hybrid architecture a separate factor worth examining. H2's gains cannot be attributed entirely to the gated delta rule because its layer composition also changes.

The tables do not give confidence intervals for these averages, so they do not establish the statistical significance of small differences. Nor do they establish a universal result across model sizes, budgets, or retrieval tasks.

## 8. Four questions for the next paper

1. **Where is history stored?** Per-token KV, finite-dimensional state, or a hybrid? Account for each layer.
2. **How does state change?** Identify global decay, key-directed correction, and new writes separately.
3. **What conditions make the update selective?** Inspect key normalization, correlations, and gate ranges.
4. **What does the evidence isolate?** Separate the update rule, the complete block, and the hybrid layer composition.

The lasting skill is being able to follow an update equation from retained history to new associations, and then identify the costs that the choice passes to computation and architecture.

## Primary sources

1. [Gated Delta Networks: Improving Mamba2 with Delta Rule, arXiv v3](https://arxiv.org/html/2412.06464v3); [ICLR 2025 paper](https://openreview.net/forum?id=r8H7xhYPwz). Equations: §2–3. Main comparisons: Tables 3 and 4.
2. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). Background on linear attention and its recurrent form.
3. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060). Background on Mamba2 and SSD.
4. [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484). Background on DeltaNet and chunkwise training.
5. [Official NVlabs/GatedDeltaNet implementation](https://github.com/NVlabs/GatedDeltaNet). Software support can change; this reading uses the paper version identified above.

Illustration generator: [`scripts/generate-gated-deltanet-figures.mjs`](https://github.com/chenhuiyu/chenhuiyu.github.io/blob/source/scripts/generate-gated-deltanet-figures.mjs). Chart values and provenance are recorded in [sources.json](/blog/gated-deltanet/sources.json). Rebuild the figures with `node scripts/generate-gated-deltanet-figures.mjs`; no model training is required.
