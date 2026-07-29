import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../public/blog/generative-recommendation");

const c = {
  paper: "#FCFAF5",
  ink: "#172033",
  muted: "#667085",
  line: "#C8D2E1",
  blue: "#2563EB",
  blueSoft: "#E8F0FF",
  violet: "#7C3AED",
  violetSoft: "#F1EAFE",
  amber: "#D97706",
  amberSoft: "#FFF3DA",
  green: "#16856B",
  greenSoft: "#E2F5EF",
  red: "#C2415B",
  redSoft: "#FCE8EC",
  white: "#FFFFFF",
};

const esc = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const multiline = (x, y, lines, options = {}) => {
  const {
    size = 18,
    fill = c.ink,
    weight = 500,
    gap = 28,
    anchor = "start",
  } = options;
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" font-weight="${weight}" fill="${fill}">${lines
    .map(
      (line, index) =>
        `<tspan x="${x}" dy="${index === 0 ? 0 : gap}">${esc(line)}</tspan>`,
    )
    .join("")}</text>`;
};

const panel = (x, y, width, height, fill = c.white, stroke = c.line, radius = 22) =>
  `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`;

const pill = (x, y, label, fill = c.blueSoft, color = c.blue, width) => {
  const w = width ?? Math.max(82, label.length * 15 + 28);
  return `${panel(x, y, w, 34, fill, fill, 17)}
    <text x="${x + w / 2}" y="${y + 23}" text-anchor="middle" font-size="15" font-weight="700" fill="${color}">${esc(label)}</text>`;
};

const arrow = (x1, y1, x2, y2, color = c.line) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="3" marker-end="url(#arrow)"/>`;

const base = ({ title, subtitle, body, footer = "" }) => `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720" role="img" aria-labelledby="title desc">
  <title id="title">${esc(title)}</title>
  <desc id="desc">${esc(subtitle)}</desc>
  <defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#172033" flood-opacity=".08"/>
    </filter>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
      <path d="M0,0 L12,6 L0,12 Z" fill="${c.line}"/>
    </marker>
  </defs>
  <rect width="1200" height="720" fill="${c.paper}"/>
  <circle cx="1110" cy="70" r="160" fill="${c.blueSoft}" opacity=".55"/>
  <circle cx="85" cy="675" r="145" fill="${c.violetSoft}" opacity=".55"/>
  <text x="64" y="62" font-size="31" font-weight="760" fill="${c.ink}">${esc(title)}</text>
  <text x="64" y="95" font-size="17" font-weight="500" fill="${c.muted}">${esc(subtitle)}</text>
  ${body}
  ${
    footer
      ? `<line x1="64" y1="665" x2="1136" y2="665" stroke="${c.line}"/>
         <text x="64" y="696" font-size="15" fill="${c.muted}">${esc(footer)}</text>`
      : ""
  }
</svg>`;

const compare = ({ title, subtitle, left, right, bridge, footer }) => {
  const card = (x, side, tone) => {
    const color = tone === "old" ? c.amber : c.blue;
    const soft = tone === "old" ? c.amberSoft : c.blueSoft;
    return `${panel(x, 145, 430, 430, c.white, color)}
      ${pill(x + 28, 170, side.kicker, soft, color)}
      ${multiline(x + 28, 235, [side.title], { size: 26, weight: 760 })}
      ${side.rows
        .map(
          (row, i) => `${panel(x + 28, 290 + i * 70, 374, 50, soft, soft, 14)}
            <circle cx="${x + 52}" cy="${315 + i * 70}" r="7" fill="${color}"/>
            <text x="${x + 70}" y="${321 + i * 70}" font-size="17" font-weight="600" fill="${c.ink}">${esc(row)}</text>`,
        )
        .join("")}`;
  };
  return base({
    title,
    subtitle,
    body: `${card(64, left, "old")}
      ${arrow(520, 360, 680, 360, c.violet)}
      ${panel(535, 314, 130, 92, c.violetSoft, c.violet, 46)}
      ${multiline(600, 348, bridge, { size: 16, weight: 760, fill: c.violet, gap: 22, anchor: "middle" })}
      ${card(706, right, "new")}`,
    footer,
  });
};

const flow = ({ title, subtitle, steps, formula, footer }) => {
  const width = steps.length === 5 ? 194 : 238;
  const gap = steps.length === 5 ? 18 : 24;
  const start = 64;
  const tones = [
    [c.blueSoft, c.blue],
    [c.violetSoft, c.violet],
    [c.amberSoft, c.amber],
    [c.greenSoft, c.green],
    [c.redSoft, c.red],
  ];
  const body = steps
    .map((step, i) => {
      const x = start + i * (width + gap);
      const [soft, color] = tones[i % tones.length];
      return `${i > 0 ? arrow(x - gap + 2, 350, x - 8, 350) : ""}
        ${panel(x, 165, width, 350, c.white, color)}
        ${pill(x + 18, 187, `0${i + 1}`, soft, color, 52)}
        ${multiline(x + 18, 250, [step.title], { size: 21, weight: 760 })}
        ${step.lines
          .map(
            (line, j) => `${panel(x + 16, 305 + j * 57, width - 32, 42, soft, soft, 12)}
              <text x="${x + width / 2}" y="${332 + j * 57}" text-anchor="middle" font-size="15" font-weight="620" fill="${c.ink}">${esc(line)}</text>`,
          )
          .join("")}`;
    })
    .join("");
  return base({
    title,
    subtitle,
    body: `${body}
      ${formula ? `${panel(190, 545, 820, 78, c.ink, c.ink, 18)}
        <text x="600" y="593" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="18" font-weight="650" fill="${c.white}">${esc(formula)}</text>` : ""}`,
    footer,
  });
};

const cards = ({ title, subtitle, items, footer }) => {
  const cols = items.length === 5 ? 5 : 4;
  const width = cols === 5 ? 198 : 252;
  const gap = 18;
  const start = 64;
  const tones = [
    [c.blueSoft, c.blue],
    [c.violetSoft, c.violet],
    [c.amberSoft, c.amber],
    [c.greenSoft, c.green],
    [c.redSoft, c.red],
  ];
  return base({
    title,
    subtitle,
    body: items
      .map((item, i) => {
        const x = start + i * (width + gap);
        const [soft, color] = tones[i % tones.length];
        return `${panel(x, 155, width, 445, c.white, color)}
          ${pill(x + 18, 179, item.kicker ?? `0${i + 1}`, soft, color)}
          ${multiline(x + 18, 245, [item.title], { size: 21, weight: 760 })}
          ${item.lines
            .map(
              (line, j) => `${panel(x + 16, 303 + j * 65, width - 32, 48, soft, soft, 12)}
                <text x="${x + width / 2}" y="${333 + j * 65}" text-anchor="middle" font-size="15" font-weight="620" fill="${c.ink}">${esc(line)}</text>`,
            )
            .join("")}`;
      })
      .join(""),
    footer,
  });
};

const bars = ({ title, subtitle, groups, min = 0, max, footer }) => {
  const chartX = 92;
  const chartY = 155;
  const chartW = 1016;
  const groupW = chartW / groups.length;
  const maxBars = Math.max(...groups.map((g) => g.values.length));
  const barW = Math.min(52, (groupW - 52) / maxBars);
  const gap = 10;
  const scale = (v) => ((v - min) / (max - min)) * 300;
  const grid = [0, 0.25, 0.5, 0.75, 1]
    .map((p) => {
      const y = chartY + 330 - 300 * p;
      const value = min + (max - min) * p;
      return `<line x1="${chartX}" y1="${y}" x2="${chartX + chartW}" y2="${y}" stroke="${c.line}" stroke-dasharray="5 7"/>
        <text x="${chartX - 12}" y="${y + 5}" text-anchor="end" font-size="13" fill="${c.muted}">${value >= 1 ? value.toFixed(1) : value.toFixed(3)}</text>`;
    })
    .join("");
  const marks = groups
    .map((group, gi) => {
      const center = chartX + gi * groupW + groupW / 2;
      const total = group.values.length * barW + (group.values.length - 1) * gap;
      return `${group.values
        .map((v, vi) => {
          const h = Math.max(2, scale(v.value));
          const x = center - total / 2 + vi * (barW + gap);
          const y = chartY + 330 - h;
          return `<rect x="${x}" y="${y}" width="${barW}" height="${h}" rx="8" fill="${v.color}"/>
            <text x="${x + barW / 2}" y="${y - 10}" text-anchor="middle" font-size="13" font-weight="700" fill="${v.color}">${esc(v.display ?? v.value)}</text>
            <text x="${x + barW / 2}" y="${chartY + 354}" text-anchor="middle" font-size="12" font-weight="650" fill="${c.muted}">${esc(v.label)}</text>`;
        })
        .join("")}
        <text x="${center}" y="${chartY + 395}" text-anchor="middle" font-size="16" font-weight="760" fill="${c.ink}">${esc(group.label)}</text>`;
    })
    .join("");
  return base({
    title,
    subtitle,
    body: `${panel(64, 125, 1072, 500, c.white, c.line)}${grid}${marks}`,
    footer,
  });
};

const files = {
  "16-hstu/paradigm-shift.svg": compare({
    title: "工业推荐不只是“更大的 SASRec” / Sequential Transduction",
    subtitle: "HSTU 先重写数据与任务，再重写注意力；规模化来自表示、训练与服务的共同改变。",
    left: {
      kicker: "DLRM",
      title: "一条曝光一个样本",
      rows: ["千级手工特征", "候选逐个交叉", "计算随候选数增长", "扩模型后质量易饱和"],
    },
    right: {
      kicker: "Generative Recommender",
      title: "一位用户一条时间线",
      rows: ["行为与类别特征序列化", "一次编码监督多个位置", "ranking / retrieval 同一骨干", "质量随 compute 继续增长"],
    },
    bridge: ["重排数据", "重写任务"],
    footer: "关键变化：用户 action 被视为一种生成建模模态，而不是把商品标题改写成自然语言。",
  }),
  "16-hstu/toy-transduction.svg": cards({
    title: "八件商品中的两种转导任务 / Ranking & Retrieval",
    subtitle: "商品 Φ 与动作 a 交错，既能预测用户怎样响应候选，也能预测下一次正向互动。",
    items: [
      { kicker: "输入", title: "历史时间线", lines: ["I01 网球拍 + 点击", "I03 攀岩鞋 + 完播", "I05 羽毛球拍 + 跳过", "I07 泳帽 + 收藏"] },
      { kicker: "Ranking", title: "预测动作", lines: ["给定候选 I06", "读取完整历史", "输出点击 / 完播 / 时长", "target-aware 一次完成"] },
      { kicker: "Retrieval", title: "预测内容", lines: ["只在正反馈处监督", "历史状态 uᵢ", "输出下一正向商品", "I06 泳镜"] },
      { kicker: "时间", title: "相对偏置", lines: ["位置距离", "真实时间间隔", "近期与周期兴趣", "进入 rabᵖ,ᵗ"] },
    ],
    footer: "同一 token 流可在不同位置发出不同监督；“生成式”首先指训练范式，而不必输出自然语言。",
  }),
  "16-hstu/mechanism.svg": flow({
    title: "HSTU Layer：投影、聚合、门控 / High-performance block",
    subtitle: "用 pointwise aggregated attention 保留兴趣强度，用时间偏置与逐元素门控替代传统 FFN 堆叠。",
    steps: [
      { title: "Pointwise projection", lines: ["X → U,V,Q,K", "两次线性层", "SiLU + fused kernel"] },
      { title: "Spatial aggregation", lines: ["SiLU(QKᵀ + rabᵖ,ᵗ)", "不用 sequence softmax", "保留相关行为数量"] },
      { title: "Feature interaction", lines: ["LayerNorm(AV)", "逐元素 × U", "attention 与 gate 合流"] },
      { title: "Scale system", lines: ["Jagged kernel", "Stochastic Length", "M-FALCON amortization"] },
    ],
    formula: "Y(X) = f₂( Norm( SiLU(QKᵀ + rabᵖ,ᵗ)V ) ⊙ U )",
    footer: "Source: HSTU Equations 1–3；pointwise attention 不是 softmax attention 的同义改名。",
  }),
  "16-hstu/evidence.svg": bars({
    title: "HSTU 规模越大，公共数据收益越明显 / NDCG@10",
    subtitle: "同样的 sequential setting；HSTU-large 使用 4× 层数、2× heads。",
    max: 0.23,
    groups: [
      { label: "MovieLens-1M", values: [{ label: "SAS", value: 0.1603, color: c.line }, { label: "HSTU", value: 0.172, color: c.blue }, { label: "Large", value: 0.1893, color: c.violet }] },
      { label: "MovieLens-20M", values: [{ label: "SAS", value: 0.1621, color: c.line }, { label: "HSTU", value: 0.1878, color: c.blue }, { label: "Large", value: 0.2106, color: c.violet }] },
      { label: "Amazon Books", values: [{ label: "SAS", value: 0.0156, color: c.line }, { label: "HSTU", value: 0.0219, color: c.blue }, { label: "Large", value: 0.0257, color: c.violet }] },
    ],
    footer: "Source: HSTU Table 4. 工业结果另报告 1.5T 参数、ranking 主任务线上 +12.4%。",
  }),

  "17-onerec/paradigm-shift.svg": compare({
    title: "从三级漏斗到一次生成 / Unify Retrieve and Rank",
    subtitle: "OneRec 让同一个模型生成最终 session，并把 session 级业务偏好直接对齐到输出分布。",
    left: {
      kicker: "Cascade",
      title: "Retrieve → Pre-rank → Rank",
      rows: ["每级只看上一级候选", "前级漏掉后级无法挽回", "多个目标分别调参", "列表靠规则拼接"],
    },
    right: {
      kicker: "OneRec",
      title: "History → Session",
      rows: ["Balanced Semantic IDs", "Sparse MoE 扩容量", "一次写 5–10 个视频", "RM + DPO 对齐业务奖励"],
    },
    bridge: ["单阶段", "列表生成"],
    footer: "统一优化减少级联上限，但在线仍需要 codebook、beam、奖励模型与增量训练系统。",
  }),
  "17-onerec/toy-session.svg": cards({
    title: "不是预测下一件，而是学习一段高价值 session",
    subtitle: "训练目标是一批真实请求返回且被充分消费的列表；列表内部顺序成为条件。",
    items: [
      { kicker: "历史", title: "用户行为", lines: ["I01 网球拍", "I03 攀岩鞋", "I05 羽毛球拍", "I07 泳帽"] },
      { kicker: "高价值筛选", title: "Session label", lines: ["实际观看 ≥ 5 个", "总观看时长过阈值", "点赞 / 收藏 / 分享", "不是任意曝光列表"] },
      { kicker: "逐件写", title: "Target session", lines: ["I06 泳镜", "I08 运动毛巾", "I04 镁粉袋", "I02 网球"] },
      { kicker: "上下文", title: "列表依赖", lines: ["第 2 件看见第 1 件", "学习连贯与多样", "BOS 分隔每件商品", "整段 NTP loss"] },
    ],
    footer: "session-wise generation 提供列表依赖结构；所谓“高价值”仍由日志筛选规则和奖励模型定义。",
  }),
  "17-onerec/mechanism.svg": flow({
    title: "OneRec：造码、生成、对齐 / The complete loop",
    subtitle: "Balanced K-means 防止 hourglass，Sparse MoE 增容量，IPA 从模型自己的候选中挖 hard preference pairs。",
    steps: [
      { title: "Balanced IDs", lines: ["multi-modal embedding", "residual K-means", "每簇 |V|/K"] },
      { title: "History encoder", lines: ["正向观看 / 互动", "三层 Semantic ID", "T5-like encoder"] },
      { title: "Sparse MoE decoder", lines: ["24 experts", "Top-2 activated", "5–10 item session"] },
      { title: "Reward model", lines: ["beam 128 responses", "watch / view / like", "best vs worst"] },
      { title: "IPA / DPO", lines: ["1% samples", "NTP + λDPO", "iterate snapshot"] },
    ],
    formula: "L = L_NTP + λ · [−log σ(βΔ log πθ/πref)]",
    footer: "实验配置：1B 模型、24 experts、每 token 激活 2 个；偏好采样只占训练样本的 1%。",
  }),
  "17-onerec/evidence.svg": bars({
    title: "列表生成与 IPA 都带来增益 / Offline reward-model metrics",
    subtitle: "工业测试集上的 maximum session watch time (swt) 与 like probability (ltr)。",
    max: 0.21,
    groups: [
      { label: "Max session watch time", values: [{ label: "TIGER", value: 0.1368, color: c.line }, { label: "OneRec", value: 0.1529, color: c.blue }, { label: "+IPA", value: 0.1933, color: c.violet }] },
      { label: "Max like probability", values: [{ label: "TIGER", value: 0.0579, color: c.line }, { label: "OneRec", value: 0.066, color: c.blue }, { label: "+IPA", value: 0.1203, color: c.violet }] },
    ],
    footer: "Source: OneRec Table 1. 线上对多阶段系统：总观看时长 +1.68%，平均单视频观看时长 +6.56%。",
  }),

  "18-mtgr/paradigm-shift.svg": compare({
    title: "规模化不该以丢掉交叉特征为代价 / MTGR",
    subtitle: "Meituan 发现纯 next-token 路线丢失 candidate-aware statistics，放大模型也补不回来。",
    left: {
      kicker: "Pure GRM",
      title: "只把行为排成序列",
      rows: ["长历史可端到端建模", "用户计算可复用", "去掉候选交叉特征", "scale 后仍有质量缺口"],
    },
    right: {
      kicker: "MTGR",
      title: "Token 序列 + Cross features",
      rows: ["用户与序列 token", "候选 token 带 ctr / pv", "一个用户聚合多候选", "判别 loss 输出每项分数"],
    },
    bridge: ["保留先验", "重新排布"],
    footer: "工程结论不是回到旧 DLRM，而是把已验证有效的特征装进可规模化的 HSTU 表示。",
  }),
  "18-mtgr/toy-compression.svg": cards({
    title: "用户级压缩：用户算一次，候选共享 / User aggregation",
    subtitle: "同一请求的候选被聚合到一个 token 序列，避免为每个候选重复编码历史。",
    items: [
      { kicker: "共享用户", title: "Static tokens", lines: ["年龄 / 城市", "I01,I03,I05 历史", "请求前长期序列", "所有候选可见"] },
      { kicker: "实时行为", title: "Dynamic tokens", lines: ["I07 刚刚收藏", "按时间因果可见", "禁止看见未来事件", "dynamic mask"] },
      { kicker: "候选 I06", title: "Cross token", lines: ["商品 ID / 类目", "user×item CTR", "历史曝光次数", "输出 click score"] },
      { kicker: "候选 I08", title: "Cross token", lines: ["另一组 cross stats", "与 I06 同批计算", "候选间按规则遮罩", "输出 conversion score"] },
    ],
    footer: "传统 K 个 pair samples 变为 1 个 user-level sample；推理成本对候选数呈次线性增长。",
  }),
  "18-mtgr/mechanism.svg": flow({
    title: "MTGR 的异构 token 编码 / GLN + Dynamic Mask",
    subtitle: "不同语义域先各自归一化，再由定制可见性规则进入 HSTU，候选位置输出判别 logits。",
    steps: [
      { title: "Feature tokenize", lines: ["User / Seq / RT", "Candidate + Cross", "MLP → d_model"] },
      { title: "Group-Layer Norm", lines: ["每个 domain 单独 LN", "对齐数值分布", "不混淆语义空间"] },
      { title: "Dynamic mask", lines: ["static: full visible", "realtime: causal", "candidate: self-only rules"] },
      { title: "HSTU encoder", lines: ["SiLU attention", "U gate", "15 layers at large"] },
      { title: "Discriminative heads", lines: ["candidate outputs", "CTR / CTCVR", "cross feature retained"] },
    ],
    formula: "D = [U, S, R, [C,I]₁, …, [C,I]ₖ]  →  logits₁…ₖ",
    footer: "“Generative”在 MTGR 中主要体现为用户级序列训练与计算复用；最终 ranking 仍输出判别分数。",
  }),
  "18-mtgr/evidence.svg": bars({
    title: "交叉特征不是可有可无 / CTCVR GAUC",
    subtitle: "10-day Meituan dataset；纵轴从 0.645 截断以显示工业指标的细小但重要差异。",
    min: 0.645,
    max: 0.667,
    groups: [
      { label: "Strong DLRM", values: [{ label: "UserTower", value: 0.655, color: c.line }] },
      { label: "MTGR scale", values: [{ label: "Small", value: 0.6603, color: c.blue }, { label: "Medium", value: 0.6625, color: c.violet }, { label: "Large", value: 0.6646, color: c.green }] },
      { label: "Ablation", values: [{ label: "w/o cross", value: 0.6514, color: c.red }, { label: "Full small", value: 0.6603, color: c.blue }] },
    ],
    footer: "Source: MTGR Tables 3–4. Large 线上 PV_CTR +1.90%、UV_CTCVR +1.02%，已部署主流量。",
  }),

  "19-onerec-think/paradigm-shift.svg": compare({
    title: "生成 ID 不等于会推理 / In-text Reasoning",
    subtitle: "OneRec-Think 把 itemic tokens 接入语言空间，并在答案前显式生成可读 reasoning trace。",
    left: {
      kicker: "Implicit GR",
      title: "History → Item tokens",
      rows: ["行为 code 缺少可读语义", "一次黑箱预测", "难处理自然语言约束", "业务奖励只看最终列表"],
    },
    right: {
      kicker: "OneRec-Think",
      title: "History → Think → Item",
      rows: ["Itemic–text alignment", "相关行为压缩", "文本 CoT + item evidence", "Rollout-Beam reward"],
    },
    bridge: ["先解释", "再生成"],
    footer: "显式文字提供可控接口，但文字流畅本身不能证明中间推理对答案有因果贡献。",
  }),
  "19-onerec-think/toy-reasoning.svg": cards({
    title: "八件商品 + 一句实时意图 / Dialogue-aware recommendation",
    subtitle: "模型需要在历史兴趣和当前约束冲突时，先重组证据再生成商品地址。",
    items: [
      { kicker: "历史", title: "Mixed interests", lines: ["网球 / 羽毛球", "攀岩", "游泳", "近期收藏泳帽"] },
      { kicker: "用户说", title: "Current request", lines: ["“今天肩膀不舒服”", "“想要轻松恢复”", "自然语言优先约束", "覆盖旧的激烈运动偏好"] },
      { kicker: "Think", title: "Reasoning trace", lines: ["近期水上兴趣", "排除高冲击运动", "补充训练后恢复", "选择速干毛巾"] },
      { kicker: "Answer", title: "Itemic tokens", lines: ["<item_begin>", "<a_…><b_…><c_…>", "映射 I08", "合法目录解码"] },
    ],
    footer: "训练 trace 必须引用历史证据而不能泄露 target；否则模型只是在答案后补一个故事。",
  }),
  "19-onerec-think/mechanism.svg": flow({
    title: "OneRec-Think 三阶段 + Think-Ahead 部署",
    subtitle: "先让 item code 可被语言理解，再用 SFT 激活 reasoning，最后用多有效答案奖励增强。",
    steps: [
      { title: "Itemic Alignment", lines: ["persona interleaving", "sequence prediction", "dense caption + text"] },
      { title: "Reasoning Activation", lines: ["prune relevant history", "teacher rationale", "raw noisy history SFT"] },
      { title: "Reasoning Enhancement", lines: ["16 CoT rollouts", "beam width 32", "GRPO rollout-beam"] },
      { title: "Offline think-ahead", lines: ["sample reasoning paths", "generate first 2 codes", "cache prefix set Cᵤ"] },
      { title: "Online finalize", lines: ["real-time OneRec", "prefix-constrained", "decode final code"] },
    ],
    formula: "R_Rollout-Beam = max_{ŝ ∈ Beam_K} Σ_l 𝟙[ŝ_l = s*_l]",
    footer: "Think-Ahead 把慢推理变成离线语义先验，线上只在缓存前缀内完成最后一级选择。",
  }),
  "19-onerec-think/evidence.svg": bars({
    title: "Itemic Alignment 打地基，Reasoning 再增益 / Beauty",
    subtitle: "Ablation on Recall and NDCG；Base 使用原始 itemic sequence 调优。",
    max: 0.09,
    groups: [
      { label: "Recall@5", values: [{ label: "Base", value: 0.046, color: c.line }, { label: "+IA", value: 0.0532, color: c.blue }, { label: "+R", value: 0.0563, color: c.violet }] },
      { label: "Recall@10", values: [{ label: "Base", value: 0.0654, color: c.line }, { label: "+IA", value: 0.0735, color: c.blue }, { label: "+R", value: 0.0791, color: c.violet }] },
      { label: "NDCG@10", values: [{ label: "Base", value: 0.0377, color: c.line }, { label: "+IA", value: 0.0402, color: c.blue }, { label: "+R", value: 0.0471, color: c.violet }] },
    ],
    footer: "Source: OneRec-Think Table 2. Kuaishou 线上 App Stay Time +0.159%，Watch Time +0.169%。",
  }),

  "20-onereason/paradigm-shift.svg": compare({
    title: "会写 CoT，不等于 CoT 帮助了推荐 / OneReason",
    subtitle: "OneReason 从失败现象出发：SFT 后 thinking mode 反而弱于直接回答，问题在 perception 与 cognition。",
    left: {
      kicker: "Surface reasoning",
      title: "Fluent but hollow",
      rows: ["itemic token 没被真正理解", "历史只做表面复述", "CoT 漂移或泄露 target", "think 结果弱于 non-think"],
    },
    right: {
      kicker: "Grounded reasoning",
      title: "Perception + Cognition",
      rows: ["R0：读懂商品 token", "R1：建立 item 关系", "R2：建模兴趣演化", "R3：压缩后作推荐决策"],
    },
    bridge: ["先看懂", "再思考"],
    footer: "推荐推理是溯因：从多种可能未来中假设潜在兴趣，而非像数学题那样推导唯一答案。",
  }),
  "20-onereason/toy-reasoning.svg": cards({
    title: "从八件商品走完 R0 → R3 / A diagnostic ladder",
    subtitle: "每一层都可独立评测；高层失败不再被笼统归因于“模型不够大”。",
    items: [
      { kicker: "R0", title: "Perception", lines: ["读出 I07 是泳帽", "理解三级 code", "内容 / 图像 / 音频对齐", "否则历史不可读"] },
      { kicker: "R1", title: "Derivation", lines: ["泳帽 → 泳镜", "攀岩鞋 → 镁粉袋", "常识 + 协同 bridge", "从单件抽潜在兴趣"] },
      { kicker: "R2", title: "Evolution", lines: ["球拍兴趣反复出现", "近期转向游泳", "区分长期 / 短期", "识别细化与饱和"] },
      { kicker: "R3", title: "Recommendation", lines: ["压缩 persona", "压缩兴趣演化", "比较多条候选方向", "生成 I06 / I08"] },
    ],
    footer: "R3 trace 禁止直接出现 ground-truth target；目标只在 <think> 之后作为监督答案。",
  }),
  "20-onereason/mechanism.svg": flow({
    title: "OneReason：深对齐、结构化 CoT、先专后合",
    subtitle: "四粒度预训练解决 perception，R0–R3 SFT 解决 cognition，单域 RL 与 RFT/MOPD 解决跨域冲突。",
    steps: [
      { title: "Itemic tokenizer", lines: ["ViT + LLM + audio", "3×8192 RQ-KMeans", "domain + 3 codes"] },
      { title: "4-granularity PT", lines: ["token / item", "relation / user", "general data retained"] },
      { title: "R0–R3 SFT", lines: ["perceive → derive", "evolve → recommend", "two-axis compression"] },
      { title: "Single-domain RL", lines: ["N CoTs × K answers", "hit × diversity reward", "GRPO specialists"] },
      { title: "Unify + deploy", lines: ["RFT or MOPD", "fast–slow pipeline", "Thinking Token distill"] },
    ],
    formula: "R_{u,i,j} = 𝟙[c_{u,i,j} ∈ C⁺_u] · max(0, mᵢ⁽¹⁾−1)/(K−1)",
    footer: "工作仍标注 Work in progress；结果应视作一份大规模技术报告，而不是推理因果性的最终定论。",
  }),
  "20-onereason/evidence.svg": bars({
    title: "SFT 后“多想”更差；RFT 后才稳定反转 / Recall@64 (%)",
    subtitle: "四个 Kuaishou cross-domain benchmarks；每组依次为 SFT non-think、SFT think、RFT non-think、RFT think。",
    max: 20,
    groups: [
      { label: "Cross-Video", values: [{ label: "SFT-N", value: 0.11, display: "0.11", color: c.line }, { label: "SFT-T", value: 0.06, display: "0.06", color: c.amber }, { label: "RFT-N", value: 0.19, display: "0.19", color: c.blue }, { label: "RFT-T", value: 0.24, display: "0.24", color: c.violet }] },
      { label: "Cross-Product", values: [{ label: "SFT-N", value: 2.96, color: c.line }, { label: "SFT-T", value: 1.65, color: c.amber }, { label: "RFT-N", value: 3.96, color: c.blue }, { label: "RFT-T", value: 4.19, color: c.violet }] },
      { label: "Cross-Ad", values: [{ label: "SFT-N", value: 6.49, color: c.line }, { label: "SFT-T", value: 3.41, color: c.amber }, { label: "RFT-N", value: 7.26, color: c.blue }, { label: "RFT-T", value: 7.5, color: c.violet }] },
      { label: "Cross-Live", values: [{ label: "SFT-N", value: 15.52, color: c.line }, { label: "SFT-T", value: 14.32, color: c.amber }, { label: "RFT-N", value: 18.17, color: c.blue }, { label: "RFT-T", value: 18.35, color: c.violet }] },
    ],
    footer: "Source: OneReason Table 9. 线上 fast+slow 联合：Impressions +10.332%、Revenue +8.234%。",
  }),
};

for (const [relative, contents] of Object.entries(files)) {
  const target = path.join(root, relative);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, contents);
}

console.log(`Generated ${Object.keys(files).length} industrial/reasoning figures.`);
