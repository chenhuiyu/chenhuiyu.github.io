import fs from "node:fs";
import path from "node:path";

const root = path.resolve("public/blog/generative-recommendation");
const W = 1600;
const H = 900;
const c = {
  bg: "#faf7f0",
  ink: "#171612",
  muted: "#6f6a61",
  line: "#c9c0b3",
  blue: "#3b82f6",
  blueDark: "#1d4ed8",
  blueBg: "#eff6ff",
  violet: "#8b5cf6",
  violetDark: "#6d28d9",
  violetBg: "#f5f3ff",
  amber: "#f59e0b",
  amberDark: "#92400e",
  amberBg: "#fff7ed",
  green: "#10b981",
  greenDark: "#166534",
  greenBg: "#ecfdf5",
  rose: "#e11d48",
  roseBg: "#fff1f2",
};

const esc = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const text = (x, y, value, size = 24, fill = c.ink, weight = 400, anchor = "start") =>
  `<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${esc(value)}</text>`;
const rect = (x, y, width, height, fill, stroke = "none", rx = 22, strokeWidth = 2) =>
  `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
const line = (x1, y1, x2, y2, stroke = c.line, width = 3, dash = "") =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${width}"${dash ? ` stroke-dasharray="${dash}"` : ""}/>`;
const arrow = (x1, y1, x2, y2, color = c.violet) =>
  `${line(x1, y1, x2 - 18, y2, color, 5)}
  <path d="M${x2 - 24} ${y2 - 13} L${x2} ${y2} L${x2 - 24} ${y2 + 13}" fill="none" stroke="${color}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>`;
const titleBlock = (title, subtitle) =>
  `${text(80, 82, title, 42, c.ink, 650)}${text(80, 124, subtitle, 21, c.muted)}`;
const svg = (title, desc, body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">
  <title id="title">${esc(title)}</title><desc id="desc">${esc(desc)}</desc>
  <rect width="${W}" height="${H}" fill="${c.bg}"/>
  <g font-family="Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">${body}</g>
</svg>
`;

function shift({ title, subtitle, left, right, bridge, footer }) {
  return svg(title, subtitle, `${titleBlock(title, subtitle)}
    ${rect(90, 225, 575, 420, c.blueBg, c.blue, 30)}
    ${text(135, 285, left.kicker, 18, c.blueDark, 750)}
    ${text(135, 342, left.title, 34, c.ink, 650)}
    ${left.lines.map((value, i) => text(135, 408 + i * 50, value, 22, c.muted)).join("\n")}
    ${arrow(695, 435, 905, 435, c.violet)}
    ${rect(708, 365, 174, 62, c.violetBg, c.violet, 31)}
    ${text(795, 405, bridge, 20, c.violetDark, 750, "middle")}
    ${rect(935, 225, 575, 420, c.greenBg, c.green, 30)}
    ${text(980, 285, right.kicker, 18, c.greenDark, 750)}
    ${text(980, 342, right.title, 34, c.ink, 650)}
    ${right.lines.map((value, i) => text(980, 408 + i * 50, value, 22, c.muted)).join("\n")}
    ${line(90, 710, 1510, 710, c.line, 2, "10 10")}
    ${text(800, 785, footer, 24, c.amberDark, 650, "middle")}`);
}

function cards({ title, subtitle, cards: values, footer }) {
  const styles = [
    [c.blueBg, c.blue, c.blueDark],
    [c.violetBg, c.violet, c.violetDark],
    [c.amberBg, c.amber, c.amberDark],
    [c.greenBg, c.green, c.greenDark],
  ];
  const width = values.length === 3 ? 440 : 335;
  const gap = values.length === 3 ? 50 : 35;
  const start = values.length === 3 ? 90 : 77;
  const body = values.map((card, i) => {
    const x = start + i * (width + gap);
    const [fill, stroke, dark] = styles[i];
    return `${rect(x, 220, width, 420, fill, stroke, 28)}
      ${text(x + 34, 270, `0${i + 1}`, 17, dark, 750)}
      ${text(x + 34, 325, card.title, 27, c.ink, 650)}
      ${card.lines.map((value, j) => text(x + 34, 385 + j * 45, value, 19, c.muted)).join("\n")}`;
  }).join("\n");
  return svg(title, subtitle, `${titleBlock(title, subtitle)}${body}
    ${rect(235, 705, 1130, 100, "#ffffff", c.line, 25)}
    ${text(800, 765, footer, 22, c.ink, 650, "middle")}`);
}

function pipeline({ title, subtitle, steps, formula, note }) {
  const styles = [
    [c.blueBg, c.blue, c.blueDark],
    [c.violetBg, c.violet, c.violetDark],
    [c.amberBg, c.amber, c.amberDark],
    [c.greenBg, c.green, c.greenDark],
  ];
  const width = 275;
  const gap = 55;
  const start = 85;
  const flow = steps.map((step, i) => {
    const x = start + i * (width + gap);
    const [fill, stroke, dark] = styles[i];
    return `${rect(x, 220, width, 270, fill, stroke, 28)}
      ${text(x + 28, 265, `0${i + 1}`, 17, dark, 750)}
      ${text(x + 28, 317, step.title, 27, c.ink, 650)}
      ${step.lines.map((value, j) => text(x + 28, 365 + j * 39, value, 18, c.muted)).join("\n")}
      ${i < steps.length - 1 ? arrow(x + width + 8, 355, x + width + gap - 8, 355, styles[i + 1][1]) : ""}`;
  }).join("\n");
  return svg(title, subtitle, `${titleBlock(title, subtitle)}${flow}
    ${rect(130, 565, 1340, 135, "#ffffff", c.line, 26)}
    ${text(800, 612, "核心公式 / central equation", 18, c.muted, 650, "middle")}
    ${text(800, 665, formula, 29, c.ink, 650, "middle")}
    ${rect(230, 750, 1140, 70, c.amberBg, c.amber, 22)}
    ${text(800, 794, note, 20, c.amberDark, 650, "middle")}`);
}

function bars({ title, subtitle, groups, max, footer = "Source: original paper tables; values redrawn" }) {
  const chartX = 335;
  const chartW = 1060;
  const maxSeries = Math.max(...groups.map((group) => group.values.length));
  const groupH = maxSeries > 2 ? 175 : groups.length > 3 ? 134 : 162;
  const barH = 28;
  const body = groups.map((group, gi) => {
    const top = 184 + gi * groupH;
    return `${text(90, top + 28, group.label, 20, c.ink, 700)}
      ${group.values.map((series, si) => {
        const y = top + 41 + si * 42;
        const width = Math.max(3, (series.value / max) * chartW);
        return `${rect(chartX, y, chartW, barH, "#eee9df", "none", 14)}
          ${rect(chartX, y, width, barH, series.color, "none", 14)}
          ${text(chartX - 18, y + 21, series.label, 16, c.muted, 600, "end")}
          ${text(Math.min(chartX + width + 13, 1505), y + 21, series.display ?? series.value, 16, c.ink, 700)}`;
      }).join("\n")}`;
  }).join("\n");
  return svg(title, subtitle, `${titleBlock(title, subtitle)}${body}
    ${text(1510, 850, footer, 15, c.muted, 500, "end")}`);
}

const files = {
  "10-dsi/paradigm-shift.svg": shift({
    title: "从查索引到生成标识符 / DSI",
    subtitle: "传统检索把文档存进外部索引；DSI 把文档—ID 关联写进 Transformer 参数。",
    left: { kicker: "DUAL ENCODER + ANN", title: "查询向量去索引里找", lines: ["q → embedding", "documents → embeddings", "ANN / MIPS", "索引与模型分离"] },
    right: { kicker: "DIFFERENTIABLE INDEX", title: "查询直接生成 docid", lines: ["document → docid 训练", "query → docid 训练", "beam search 输出", "参数就是索引"] },
    bridge: "generate",
    footer: "检索从“比较所有候选”变成“在 ID 树上逐 token 导航”。",
  }),
  "10-dsi/toy-index.svg": cards({
    title: "八件商品也可以是一座可生成的索引 / DSI toy",
    subtitle: "把商品描述当作 document，把商品 ID 当作 docid；查询只需要生成正确门牌号。",
    cards: [
      { title: "索引样本", lines: ["“防雾游泳护目镜”", "→ I06 / 4-2-1", "教模型记住文档"] },
      { title: "检索样本", lines: ["“游泳时保护眼睛”", "→ I06 / 4-2-1", "教模型关联查询"] },
      { title: "三种 ID", lines: ["atomic: [I06]", "naive: [0,0,6]", "semantic: [4,2,1]"] },
      { title: "推理", lines: ["query → decoder", "beam search", "返回 ID 对应商品"] },
    ],
    footer: "同一个 NLL 同时承担 indexing 与 retrieval；没有 indexing，任意 ID 对模型毫无意义。",
  }),
  "10-dsi/mechanism.svg": pipeline({
    title: "DSI 的两种训练任务与一次生成检索 / Data flow",
    subtitle: "默认采用多任务 co-training；论文发现先索引再微调通常不如联合训练。",
    steps: [
      { title: "文档表示", lines: ["前 L 个 tokens", "direct indexing", "document dⱼ"] },
      { title: "索引任务", lines: ["dⱼ → docid j", "Inputs2Targets", "参数化记忆"] },
      { title: "检索任务", lines: ["query q → j", "共享 decoder", "teacher forcing"] },
      { title: "生成 Top-K", lines: ["semantic ID trie", "beam search", "docid ranking"] },
    ],
    formula: "L = − Σ(x,j) Σₜ log Pθ(jₜ | j<t, x),  x ∈ {document, query}",
    note: "atomic ID 是一次大 softmax；string / semantic ID 用多步解码换取小词表与共享前缀。",
  }),
  "10-dsi/evidence.svg": bars({
    title: "在 228K 文档上，semantic docid 的 Hits@1 提升明显 / NQ320K",
    subtitle: "监督检索；选择各方法论文表中的 XXL / 最强语义配置。",
    max: 75,
    groups: [
      { label: "Hits@1", values: [{ label: "BM25", value: 11.6, color: c.line }, { label: "T5 dual", value: 24.3, color: c.blue }, { label: "DSI semantic", value: 40.4, color: c.violet }] },
      { label: "Hits@10", values: [{ label: "BM25", value: 34.4, color: c.line }, { label: "T5 dual", value: 67.3, color: c.blue }, { label: "DSI semantic", value: 70.3, color: c.violet }] },
    ],
    footer: "Source: DSI Table 2; NQ320K contains 228K unique documents",
  }),

  "11-tiger/paradigm-shift.svg": shift({
    title: "从文档 ID 到商品 Semantic ID / TIGER",
    subtitle: "DSI 证明 ID 可以生成；TIGER 让商品 ID 本身携带内容语义与层次结构。",
    left: { kicker: "ATOMIC ITEM ID", title: "每件商品一个孤立 token", lines: ["I01 与 I02 不共享参数", "词表随目录线性增长", "新商品没有 embedding", "Top-K 依赖全目录打分"] },
    right: { kicker: "SEMANTIC ID", title: "一件商品是一条代码路径", lines: ["4 个 code tokens", "相似商品共享前缀", "粗到细生成", "可由内容编码冷商品"] },
    bridge: "RQ-VAE",
    footer: "Semantic ID 同时是压缩表示、层次地址，也是 autoregressive retrieval 的搜索路径。",
  }),
  "11-tiger/toy-semantic-id.svg": cards({
    title: "相似商品共享门牌号前缀 / TIGER toy IDs",
    subtitle: "代码仅作示意；真实论文使用 3×256 个 RQ codes，再追加唯一冲突 token。",
    cards: [
      { title: "球拍运动", lines: ["I01 网球拍", "〈12, 04, 88, 0〉", "I05 羽毛球", "〈12, 07, 31, 0〉"] },
      { title: "攀岩装备", lines: ["I03 攀岩鞋", "〈34, 19, 02, 0〉", "I04 粉袋", "〈34, 19, 71, 0〉"] },
      { title: "粗到细", lines: ["第 1 code：大类", "后续：残差细节", "共享前缀 = 共享语义"] },
      { title: "唯一性", lines: ["三 code 冲突时", "追加第 4 code", "SID ↔ Item lookup"] },
    ],
    footer: "生成 〈34,19〉 后，beam 已进入攀岩子树；后续 token 只需区分更细商品。",
  }),
  "11-tiger/mechanism.svg": pipeline({
    title: "TIGER 从商品文本到下一件商品 / Two-stage pipeline",
    subtitle: "Tokenizer 与 recommender 分开训练；Semantic ID 生成后冻结。",
    steps: [
      { title: "内容编码", lines: ["title / brand / price", "Sentence-T5", "x ∈ R⁷⁶⁸"] },
      { title: "残差量化", lines: ["RQ-VAE 3 levels", "codebook 256", "SID + collision"] },
      { title: "行为转码", lines: ["history items", "→ SID token stream", "+ hashed user token"] },
      { title: "生成检索", lines: ["T5 encoder-decoder", "4 tokens + EOS", "beam search Top-K"] },
    ],
    formula: "cₗ = argminₖ ‖rₗ−eₖˡ‖²,   rₗ₊₁ = rₗ−eᶫcₗ",
    note: "前层 code 近似粗语义，后层量化剩余误差；所有 code 正确才映射到目标商品。",
  }),
  "11-tiger/evidence.svg": bars({
    title: "Semantic ID 的收益不是“换个名字” / NeurIPS 2023 Table 2",
    subtitle: "NDCG@5；相同生成模型，依次使用 Random ID、LSH SID、RQ-VAE SID。",
    max: 0.042,
    groups: [
      { label: "Sports", values: [{ label: "Random", value: 0.005, color: c.line }, { label: "LSH", value: 0.0146, color: c.blue }, { label: "RQ-VAE", value: 0.0181, color: c.violet }] },
      { label: "Beauty", values: [{ label: "Random", value: 0.0205, color: c.line }, { label: "LSH", value: 0.0259, color: c.blue }, { label: "RQ-VAE", value: 0.0321, color: c.violet }] },
      { label: "Toys", values: [{ label: "Random", value: 0.027, color: c.line }, { label: "LSH", value: 0.0299, color: c.blue }, { label: "RQ-VAE", value: 0.0371, color: c.violet }] },
    ],
  }),

  "12-gptrec/paradigm-shift.svg": shift({
    title: "从一次 Top-K 到逐项写列表 / GPTRec",
    subtitle: "Top-K 独立取最高分；Next-K 让第 k 件推荐依赖前面已经选出的列表。",
    left: { kicker: "TOP-K", title: "一次打分，截取前 K", lines: ["score(i | history)", "商品彼此不看", "速度快", "难直接表达列表整体目标"] },
    right: { kicker: "NEXT-K", title: "推荐列表自回归生成", lines: ["P(rₖ | history,r<k)", "避免重复", "可建模互补 / 多样", "需要 K 次前向"] },
    bridge: "list LM",
    footer: "生成式推荐不只改变商品表示，也改变“列表”这一输出对象的概率分解。",
  }),
  "12-gptrec/toy-next-k.svg": cards({
    title: "同一份历史，两种生成列表的方法 / GPTRec toy",
    subtitle: "小林历史为网球拍 → 攀岩鞋 → 羽毛球；目标输出三件。",
    cards: [
      { title: "Top-K 分数", lines: ["I06 0.82", "I02 0.79", "I04 0.77", "一次排序完成"] },
      { title: "Next-1", lines: ["history → I06", "列表：[泳镜]", "把 I06 追加输入"] },
      { title: "Next-2", lines: ["history + I06 → I04", "列表：[泳镜,粉袋]", "条件已经改变"] },
      { title: "Next-3", lines: ["再生成 I02", "可加入多样性 reward", "论文尚未这样训练"] },
    ],
    footer: "Next-K 的表达力来自条件化；若训练仍只做 next-item，训练目标与列表解码并不对齐。",
  }),
  "12-gptrec/mechanism.svg": pipeline({
    title: "GPTRec 的 SVD 子 ID 与列表解码 / Data flow",
    subtitle: "论文中的 GPTRec 使用 GPT-2 架构但不使用预训练 checkpoint。",
    steps: [
      { title: "交互矩阵", lines: ["M ∈ Rᵁˣᴵ", "truncated SVD", "t latent dims"] },
      { title: "SVD tokenizer", lines: ["normalize + noise", "each dim → v bins", "offset token ranges"] },
      { title: "Decoder LM", lines: ["item token stream", "causal Transformer", "shifted-token CE"] },
      { title: "两种推理", lines: ["Top-K: one pass", "Next-K: K passes", "remove repeats"] },
    ],
    formula: "P(R | H) = ∏ₖ₌₁ᴷ P(rₖ | H, r₁,…,rₖ₋₁)",
    note: "多 token 将 |I| 个 embedding 压到 t×v；代价是一个商品必须连续生成 t 个正确子 token。",
  }),
  "12-gptrec/evidence.svg": bars({
    title: "Next-K 有表达力，但未对齐训练会掉点 / MovieLens-1M",
    subtitle: "NDCG@10；GPTRec 为 one-token-per-item，结果来自原论文 Table 3。",
    max: 0.17,
    groups: [
      { label: "Top-K baselines", values: [{ label: "SASRec", value: 0.108, color: c.line }, { label: "BERT4Rec", value: 0.152, color: c.blue }] },
      { label: "GPTRec", values: [{ label: "Top-K", value: 0.146, color: c.violet }, { label: "Next-K", value: 0.105, color: c.amber }] },
    ],
    footer: "Source: GPTRec Table 3; Next-K model was not trained for list-level generation",
  }),

  "13-lc-rec/paradigm-shift.svg": shift({
    title: "从“有语义的 ID”到“LLM 真懂这个 ID” / LC-Rec",
    subtitle: "RQ-VAE 只让代码来自文本；LC-Rec 再用多种任务显式连接 index、语言与偏好。",
    left: { kicker: "TOKENIZE ONLY", title: "代码树有内容语义", lines: ["title → RQ-VAE codes", "相似商品共享前缀", "新 tokens 对 LLM 仍是 OOV", "推荐只教 index → index"] },
    right: { kicker: "ALIGNMENT TUNING", title: "代码与语言互相翻译", lines: ["title ↔ index", "history index → title", "intention → index", "index history → preference"] },
    bridge: "align",
    footer: "一个好地址不仅要相似，还要能被预训练 LLM 的语言知识读懂和写出。",
  }),
  "13-lc-rec/toy-alignment.svg": cards({
    title: "同一个 I06 index，四种学习信号 / LC-Rec toy",
    subtitle: "示意 index 〈a₄,b₂,c₁,d₇〉；每种任务连接一种语义边。",
    cards: [
      { title: "序列预测", lines: ["I01,I03,I05 codes", "→ I06 codes", "协同行为"] },
      { title: "互相翻译", lines: ["“防雾泳镜” → codes", "codes → 标题描述", "显式语言对齐"] },
      { title: "非对称预测", lines: ["history codes → title", "history titles → codes", "跨表示推断"] },
      { title: "意图 / 偏好", lines: ["“准备游泳” → I06", "index history → 兴趣文本", "推荐语义对齐"] },
    ],
    footer: "LC-Rec 不把 next-item loss 当作唯一老师；它用辅助任务把新 code tokens 接回语言空间。",
  }),
  "13-lc-rec/mechanism.svg": pipeline({
    title: "LC-Rec：均衡建码，再多任务对齐 / Full-ranking generation",
    subtitle: "LLaMA 文本 embedding 经 RQ-VAE；最后一级用 Sinkhorn-Knopp 均匀分配以消除冲突。",
    steps: [
      { title: "文本表示", lines: ["title + description", "LLaMA mean pool", "semantic e"] },
      { title: "均衡量化", lines: ["4×256 codebooks", "RQ coarse-to-fine", "USM at last level"] },
      { title: "对齐微调", lines: ["SEQ + mutual", "+ asymmetric", "+ intent / preference"] },
      { title: "受限生成", lines: ["beam size 20", "illegal logits = 0", "full catalog"] },
    ],
    formula: "min ΣᵣΣₖ q(k|r)‖r−vₖ‖²,  subject to Σᵣ q(k|r)=|B|/K",
    note: "均衡约束让最后一层每个 code 接收近似相同数量残差，避免追加无语义 collision suffix。",
  }),
  "13-lc-rec/evidence.svg": bars({
    title: "对齐后的 Semantic ID 提升全目录生成 / LC-Rec Table III",
    subtitle: "NDCG@10；与同为生成式检索的 TIGER 比较。",
    max: 0.1,
    groups: [
      { label: "Instruments", values: [{ label: "TIGER", value: 0.0803, color: c.line }, { label: "LC-Rec", value: 0.0926, color: c.violet }] },
      { label: "Arts", values: [{ label: "TIGER", value: 0.0703, color: c.line }, { label: "LC-Rec", value: 0.0906, color: c.violet }] },
      { label: "Games", values: [{ label: "TIGER", value: 0.0501, color: c.line }, { label: "LC-Rec", value: 0.0681, color: c.violet }] },
    ],
  }),

  "14-letter/paradigm-shift.svg": shift({
    title: "一个好 ID 要同时满足三件事 / LETTER",
    subtitle: "只保内容会错过共现；只追共现会破坏层次；code 使用失衡又会放大热门偏置。",
    left: { kicker: "CONTENT-ONLY SID", title: "语义清楚但不一定好推荐", lines: ["层次内容语义", "相似标题共享代码", "协同邻居可能被拆开", "热门 code 被过度分配"] },
    right: { kicker: "LEARNABLE TOKENIZER", title: "语义 + 协同 + 多样", lines: ["RQ-VAE reconstruction", "CF contrastive alignment", "code diversity regularizer", "ranking-guided generation"] },
    bridge: "3 losses",
    footer: "LETTER 把 tokenizer 从压缩器提升为推荐系统中独立、可检验的学习组件。",
  }),
  "14-letter/toy-tokenizer.svg": cards({
    title: "八件商品怎样约束同一套 codebook / LETTER toy",
    subtitle: "三个目标会拉扯同一组 code embedding；权重决定最终 Semantic ID 的性质。",
    cards: [
      { title: "层次语义", lines: ["攀岩鞋 ↔ 粉袋", "内容相近共享前缀", "重构原语义 embedding"] },
      { title: "协同信号", lines: ["网球拍 ↔ 吸汗带", "SASRec / LightGCN 邻近", "量化向量做 InfoNCE"] },
      { title: "分配多样", lines: ["避免少数 code 过热", "同 cluster 拉近", "不同 cluster 推远"] },
      { title: "生成排序", lines: ["低温强调 hard negatives", "Trie 保证合法路径", "Top-K 指标更对齐"] },
    ],
    footer: "理想 ID 不是“最像文本”的 ID，而是在三种约束之间为推荐任务找到可用折中。",
  }),
  "14-letter/mechanism.svg": pipeline({
    title: "LETTER tokenizer 的三重正则 / Train then instantiate",
    subtitle: "先独立训练 tokenizer，再替换 TIGER 或 LC-Rec 的原始 Item ID 方案。",
    steps: [
      { title: "Semantic", lines: ["LLM item embedding", "RQ-VAE reconstruction", "coarse-to-fine codes"] },
      { title: "Collaborative", lines: ["pretrained CF hᵢ", "align with ẑᵢ", "in-batch InfoNCE"] },
      { title: "Diversity", lines: ["cluster code embeddings", "pull same group", "push other codes"] },
      { title: "Generation", lines: ["freeze tokenizer", "TIGER / LC-Rec", "temperature loss + Trie"] },
    ],
    formula: "LLETTER = LSem + α LCF + β LDiv",
    note: "α 太大压过内容语义，β 太大干扰两侧对齐；论文的最优长度也停在 4，而不是越长越好。",
  }),
  "14-letter/evidence.svg": bars({
    title: "同一个生成后端，换 tokenizer 就能提升 / CIKM 2024",
    subtitle: "NDCG@10；分别比较 TIGER 与 LC-Rec 在原 tokenizer 和 LETTER 下的结果。",
    max: 0.095,
    groups: [
      { label: "Instruments · TIGER", values: [{ label: "base", value: 0.0797, color: c.line }, { label: "LETTER", value: 0.0831, color: c.violet }] },
      { label: "Instruments · LC", values: [{ label: "base", value: 0.0772, color: c.line }, { label: "LETTER", value: 0.0854, color: c.violet }] },
      { label: "Beauty · TIGER", values: [{ label: "base", value: 0.0331, color: c.line }, { label: "LETTER", value: 0.0364, color: c.violet }] },
      { label: "Beauty · LC", values: [{ label: "base", value: 0.0374, color: c.line }, { label: "LETTER", value: 0.0418, color: c.violet }] },
    ],
    footer: "Source: LETTER Table 1",
  }),

  "15-etegrec/paradigm-shift.svg": shift({
    title: "Tokenizer 不该只在训练前出现一次 / ETEGRec",
    subtitle: "离线 tokenizer 不知道下游生成器学得好不好；ETEGRec 让两者用推荐目标互相校准。",
    left: { kicker: "DECOUPLED", title: "先建码，随后永久冻结", lines: ["tokenizer 自己重构", "生成器接受既定 IDs", "下游错误无法反馈", "两套表示可能错位"] },
    right: { kicker: "END-TO-END ALIGNED", title: "建码与偏好模型交替演化", lines: ["sequence → item distribution", "preference ↔ semantics", "周期性交替优化", "收敛后再冻结"] },
    bridge: "feedback",
    footer: "端到端的核心不是一次反向传播穿过 argmax，而是让两个组件在训练过程中交换监督。",
  }),
  "15-etegrec/toy-coevolution.svg": cards({
    title: "同一个 I06，谁来告诉 tokenizer 应该怎样编码？ / ETEGRec toy",
    subtitle: "目标泳镜的代码不只重构自身 embedding，还要与用户序列和 decoder preference 对齐。",
    cards: [
      { title: "Tokenizer 视角", lines: ["I06 collaborative z", "RQ-VAE → 〈4,2,1〉", "重构 z̃"] },
      { title: "Encoder 视角", lines: ["I01,I03,I05 codes", "得到 sequence zᴱ", "应预测相同 code 分布"] },
      { title: "Decoder 视角", lines: ["BOS hidden hᴰ", "代表当前偏好", "应接近 I06 的 z̃"] },
      { title: "交替更新", lines: ["先冻 recommender 调码", "再冻 tokenizer 学生成", "多 cycle 后稳定"] },
    ],
    footer: "SIA 对齐“历史指向哪个 item code”；PSA 对齐“偏好状态接近哪个 item semantics”。",
  }),
  "15-etegrec/mechanism.svg": pipeline({
    title: "ETEGRec 的双重对齐与交替优化 / Co-evolution",
    subtitle: "实验用 SASRec collaborative embedding 初始化 tokenizer，T5 encoder–decoder 负责生成。",
    steps: [
      { title: "Item tokenizer", lines: ["RQ-VAE 3×256", "semantic quantization", "tokens + z̃"] },
      { title: "SIA", lines: ["sequence state zᴱ", "target item z", "symmetric KL by level"] },
      { title: "PSA", lines: ["decoder state hᴰ", "reconstructed z̃", "bidirectional InfoNCE"] },
      { title: "Alternate", lines: ["1 epoch tokenizer", "C−1 epochs recommender", "then freeze at converge"] },
    ],
    formula: "LIT = LSQ + μLSIA + λLPSA ;   LGR = LREC + μLSIA + λLPSA",
    note: "直接同时更新最不稳定；论文中去掉 alternating training 的降幅大于去掉任一 alignment。",
  }),
  "15-etegrec/evidence.svg": bars({
    title: "联合目标有效，稳定交替更关键 / SIGIR 2025",
    subtitle: "NDCG@10；LETTER、完整 ETEGRec 与去掉 alternating training 的变体。",
    max: 0.056,
    groups: [
      { label: "Instrument", values: [{ label: "LETTER", value: 0.031, color: c.line }, { label: "ETEGRec", value: 0.0331, color: c.violet }, { label: "w/o AT", value: 0.0277, color: c.amber }] },
      { label: "Scientific", values: [{ label: "LETTER", value: 0.023, color: c.line }, { label: "ETEGRec", value: 0.0241, color: c.violet }, { label: "w/o AT", value: 0.0198, color: c.amber }] },
      { label: "Game", values: [{ label: "LETTER", value: 0.0475, color: c.line }, { label: "ETEGRec", value: 0.0507, color: c.violet }, { label: "w/o AT", value: 0.0428, color: c.amber }] },
    ],
    footer: "Source: ETEGRec Tables 3 and 4",
  }),
};

for (const [relative, contents] of Object.entries(files)) {
  const target = path.join(root, relative);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, contents);
}

console.log(`Generated ${Object.keys(files).length} Semantic-ID figures.`);
