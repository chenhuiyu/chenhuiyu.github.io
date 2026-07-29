import fs from "node:fs";
import path from "node:path";

const root = path.resolve("public/blog/generative-recommendation");
const W = 1600;
const H = 900;
const palette = {
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

const text = (x, y, value, size = 24, fill = palette.ink, weight = 400, anchor = "start") =>
  `<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${esc(value)}</text>`;

const rect = (x, y, width, height, fill, stroke = "none", rx = 22, strokeWidth = 2) =>
  `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;

const arrow = (x1, y1, x2, y2, color = palette.blue) => {
  const head = 16;
  return `<path d="M${x1} ${y1} H${x2 - head}" stroke="${color}" stroke-width="5" stroke-linecap="round"/>
  <path d="M${x2 - head - 4} ${y2 - 13} L${x2} ${y2} L${x2 - head - 4} ${y2 + 13}" fill="none" stroke="${color}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>`;
};

const titleBlock = (title, subtitle) =>
  `${text(80, 82, title, 42, palette.ink, 650)}
  ${text(80, 124, subtitle, 21, palette.muted, 400)}`;

const svg = (title, desc, body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">
  <title id="title">${esc(title)}</title>
  <desc id="desc">${esc(desc)}</desc>
  <rect width="${W}" height="${H}" fill="${palette.bg}"/>
  <g font-family="Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">
    ${body}
  </g>
</svg>
`;

const itemNames = [
  ["I01", "网球拍", "racket"],
  ["I02", "吸汗带", "sweatband"],
  ["I03", "攀岩鞋", "climbing shoes"],
  ["I04", "粉袋", "chalk bag"],
  ["I05", "羽毛球", "shuttlecock"],
  ["I06", "泳镜", "goggles"],
  ["I07", "瑜伽垫", "yoga mat"],
  ["I08", "蛋白棒", "protein bar"],
];

function itemStrip(highlights = {}) {
  return itemNames
    .map(([id, zh, en], index) => {
      const x = 80 + index * 185;
      const style = highlights[id] ?? {};
      const fill = style.fill ?? "#ffffff";
      const stroke = style.stroke ?? palette.line;
      return `${rect(x, 180, 160, 105, fill, stroke, 18, 3)}
      ${text(x + 80, 215, id, 17, style.color ?? palette.muted, 700, "middle")}
      ${text(x + 80, 248, zh, 23, palette.ink, 650, "middle")}
      ${text(x + 80, 273, en, 14, palette.muted, 400, "middle")}`;
    })
    .join("\n");
}

function shiftFigure({ title, subtitle, left, right, bridge, footer }) {
  const body = `${titleBlock(title, subtitle)}
    ${rect(90, 235, 570, 400, palette.blueBg, palette.blue, 30)}
    ${text(135, 295, left.kicker, 19, palette.blueDark, 700)}
    ${text(135, 350, left.title, 35, palette.ink, 650)}
    ${left.lines.map((line, index) => text(135, 410 + index * 48, line, 23, palette.muted)).join("\n")}
    ${arrow(690, 435, 895, 435, palette.violet)}
    ${rect(705, 360, 170, 70, palette.violetBg, palette.violet, 35)}
    ${text(790, 404, bridge, 21, palette.violetDark, 700, "middle")}
    ${rect(925, 235, 585, 400, palette.greenBg, palette.green, 30)}
    ${text(970, 295, right.kicker, 19, palette.greenDark, 700)}
    ${text(970, 350, right.title, 35, palette.ink, 650)}
    ${right.lines.map((line, index) => text(970, 410 + index * 48, line, 23, palette.muted)).join("\n")}
    <line x1="90" y1="710" x2="1510" y2="710" stroke="${palette.line}" stroke-width="2" stroke-dasharray="10 10"/>
    ${text(800, 786, footer, 25, palette.amberDark, 650, "middle")}`;
  return svg(title, subtitle, body);
}

function bprToy() {
  const body = `${titleBlock("一次点击怎样变成一个训练三元组 / BPR toy example", "固定商品不变；改变的是模型如何解释“小林没有点击”。")}
    ${itemStrip({
      I03: { fill: palette.greenBg, stroke: palette.green, color: palette.greenDark },
      I06: { fill: palette.roseBg, stroke: palette.rose, color: palette.rose },
    })}
    ${rect(120, 365, 410, 245, "#ffffff", palette.green, 28)}
    ${text(160, 415, "已观察 / observed", 20, palette.greenDark, 700)}
    ${text(160, 470, "小林点击了 I03 攀岩鞋", 29, palette.ink, 650)}
    ${text(160, 520, "(u, i) ∈ S", 27, palette.greenDark, 600)}
    ${text(160, 566, "这只是行为，不是五星评分", 19, palette.muted)}
    ${rect(1070, 365, 410, 245, "#ffffff", palette.rose, 28)}
    ${text(1110, 415, "未观察 / unobserved", 20, palette.rose, 700)}
    ${text(1110, 470, "小林没有点击 I06 泳镜", 29, palette.ink, 650)}
    ${text(1110, 520, "(u, j) ∉ S", 27, palette.rose, 600)}
    ${text(1110, 566, "可能没看见，也可能不喜欢", 19, palette.muted)}
    ${arrow(555, 490, 1035, 490, palette.violet)}
    ${rect(620, 435, 350, 110, palette.violetBg, palette.violet, 26)}
    ${text(795, 478, "(小林, I03, I06)", 27, palette.violetDark, 700, "middle")}
    ${text(795, 515, "假设 I03 ≻ᵤ I06", 21, palette.muted, 500, "middle")}
    ${rect(320, 690, 960, 105, palette.amberBg, palette.amber, 26)}
    ${text(800, 735, "关键假设：observed ≻ unobserved", 28, palette.amberDark, 700, "middle")}
    ${text(800, 772, "BPR 没有证明“没点 = 不喜欢”；它把沉默变成可学习的相对顺序。", 20, palette.muted, 400, "middle")}`;
  return svg("BPR toy example", "An observed and an unobserved item become a pairwise training triple.", body);
}

function sequenceToy({ title, subtitle, mode }) {
  const highlights = {
    I01: { fill: palette.blueBg, stroke: palette.blue },
    I03: { fill: palette.violetBg, stroke: palette.violet },
    I05: { fill: palette.amberBg, stroke: palette.amber },
    I06: { fill: palette.greenBg, stroke: palette.green },
  };
  const strip = itemStrip(highlights);
  let lower = "";
  if (mode === "gru") {
    const xs = [255, 585, 915, 1245];
    const ids = ["I01 网球拍", "I03 攀岩鞋", "I05 羽毛球", "I06 泳镜"];
    lower = ids
      .map((label, i) => `${rect(xs[i] - 120, 430, 240, 100, [palette.blueBg, palette.violetBg, palette.amberBg, palette.greenBg][i], [palette.blue, palette.violet, palette.amber, palette.green][i], 24)}
        ${text(xs[i], 475, `t=${i + 1}`, 17, palette.muted, 600, "middle")}
        ${text(xs[i], 508, label, 23, palette.ink, 650, "middle")}
        ${i < ids.length - 1 ? arrow(xs[i] + 125, 480, xs[i + 1] - 125, 480, palette.violet) : ""}`)
      .join("\n");
    lower += `${rect(275, 665, 1050, 105, "#ffffff", palette.line, 24)}
      ${text(800, 708, "训练标签 / shifted targets", 20, palette.muted, 650, "middle")}
      ${text(800, 750, "I01 → I03　　I03 → I05　　I05 → I06", 29, palette.ink, 700, "middle")}`;
  } else if (mode === "sas") {
    const labels = [
      ["I01 网球拍", "0.10", palette.blue],
      ["I03 攀岩鞋", "0.58", palette.violet],
      ["I05 羽毛球", "0.32", palette.amber],
    ];
    lower = `${text(800, 382, "预测下一件：I06 泳镜（示意注意力，非论文实测权重）", 25, palette.ink, 650, "middle")}
      ${labels
        .map(([label, value, color], i) => {
          const y = 440 + i * 95;
          const width = Number(value) * 900;
          return `${text(180, y + 31, label, 22, palette.ink, 600)}
          ${rect(430, y, 900, 48, "#eee9df", "none", 24)}
          ${rect(430, y, width, 48, color, "none", 24)}
          ${text(1360, y + 32, value, 21, palette.muted, 700)}`;
        })
        .join("\n")}
      ${rect(260, 750, 1080, 70, palette.violetBg, palette.violet, 24)}
      ${text(800, 794, "同一条历史里，不是每一步都同等重要；因果 mask 保证模型看不到未来。", 22, palette.violetDark, 650, "middle")}`;
  } else {
    lower = `${text(800, 385, "随机遮住历史中的两个位置 / randomly mask two positions", 24, palette.ink, 650, "middle")}
      ${rect(160, 440, 260, 105, palette.blueBg, palette.blue, 24)}
      ${text(290, 500, "I01 网球拍", 26, palette.ink, 650, "middle")}
      ${arrow(430, 493, 535, 493, palette.violet)}
      ${rect(545, 440, 260, 105, palette.violetBg, palette.violet, 24)}
      ${text(675, 500, "[MASK]₁", 27, palette.violetDark, 700, "middle")}
      ${arrow(815, 493, 920, 493, palette.violet)}
      ${rect(930, 440, 260, 105, palette.amberBg, palette.amber, 24)}
      ${text(1060, 500, "I05 羽毛球", 26, palette.ink, 650, "middle")}
      ${arrow(1200, 493, 1305, 493, palette.violet)}
      ${rect(1315, 440, 190, 105, palette.greenBg, palette.green, 24)}
      ${text(1410, 500, "[MASK]₂", 27, palette.greenDark, 700, "middle")}
      ${rect(310, 655, 980, 125, "#ffffff", palette.line, 28)}
      ${text(800, 703, "监督信号 / labels", 20, palette.muted, 650, "middle")}
      ${text(800, 751, "[MASK]₁ = I03 攀岩鞋　　[MASK]₂ = I06 泳镜", 28, palette.ink, 700, "middle")}`;
  }
  return svg(title, subtitle, `${titleBlock(title, subtitle)}${strip}${lower}`);
}

function pipelineFigure({ title, subtitle, steps, formula, note }) {
  const widths = 275;
  const gap = 55;
  const start = 85;
  const colors = [
    [palette.blueBg, palette.blue, palette.blueDark],
    [palette.violetBg, palette.violet, palette.violetDark],
    [palette.amberBg, palette.amber, palette.amberDark],
    [palette.greenBg, palette.green, palette.greenDark],
  ];
  const pipeline = steps
    .map((step, index) => {
      const x = start + index * (widths + gap);
      const [fill, stroke, dark] = colors[index % colors.length];
      return `${rect(x, 235, widths, 245, fill, stroke, 28)}
      ${text(x + 28, 278, `0${index + 1}`, 18, dark, 750)}
      ${text(x + 28, 327, step.title, 28, palette.ink, 650)}
      ${step.lines.map((line, i) => text(x + 28, 374 + i * 37, line, 18, palette.muted)).join("\n")}
      ${index < steps.length - 1 ? arrow(x + widths + 8, 355, x + widths + gap - 8, 355, colors[(index + 1) % colors.length][1]) : ""}`;
    })
    .join("\n");
  return svg(
    title,
    subtitle,
    `${titleBlock(title, subtitle)}
    ${pipeline}
    ${rect(130, 570, 1340, 135, "#ffffff", palette.line, 26)}
    ${text(800, 618, "核心公式 / central equation", 19, palette.muted, 650, "middle")}
    ${text(800, 670, formula, 31, palette.ink, 650, "middle")}
    ${rect(230, 755, 1140, 68, palette.amberBg, palette.amber, 22)}
    ${text(800, 798, note, 20, palette.amberDark, 650, "middle")}`,
  );
}

function evidenceBars({ title, subtitle, groups, max, suffix = "" }) {
  const chartX = 310;
  const chartW = 1120;
  const maxSeries = Math.max(...groups.map((group) => group.values.length));
  const groupH = maxSeries > 2 ? 175 : 135;
  const barH = 28;
  const body = groups
    .map((group, gi) => {
      const top = 205 + gi * groupH;
      const bars = group.values
        .map((series, si) => {
          const y = top + 38 + si * 42;
          const width = (series.value / max) * chartW;
          return `${rect(chartX, y, chartW, barH, "#eee9df", "none", 14)}
          ${rect(chartX, y, width, barH, series.color, "none", 14)}
          ${text(chartX - 20, y + 21, series.label, 17, palette.muted, 600, "end")}
          ${text(Math.min(chartX + width + 14, 1515), y + 21, `${series.value}${suffix}`, 17, palette.ink, 700)}`;
        })
        .join("\n");
      return `${text(90, top + 24, group.label, 21, palette.ink, 700)}
      ${bars}`;
    })
    .join("\n");
  return svg(
    title,
    subtitle,
    `${titleBlock(title, subtitle)}
    ${body}
    ${text(1510, 842, "Source: original paper tables; values redrawn", 16, palette.muted, 500, "end")}`,
  );
}

function evidenceCards({ title, subtitle, cards, footer }) {
  const body = cards
    .map((card, index) => {
      const x = 90 + index * 495;
      const colors = [
        [palette.blueBg, palette.blue, palette.blueDark],
        [palette.violetBg, palette.violet, palette.violetDark],
        [palette.greenBg, palette.green, palette.greenDark],
      ][index];
      return `${rect(x, 230, 440, 390, colors[0], colors[1], 30)}
      ${text(x + 40, 285, `0${index + 1}`, 18, colors[2], 750)}
      ${text(x + 40, 340, card.title, 29, palette.ink, 650)}
      ${card.lines.map((line, i) => text(x + 40, 405 + i * 46, line, 20, palette.muted)).join("\n")}`;
    })
    .join("\n");
  return svg(
    title,
    subtitle,
    `${titleBlock(title, subtitle)}
    ${body}
    ${rect(250, 700, 1100, 105, palette.amberBg, palette.amber, 26)}
    ${text(800, 745, footer, 23, palette.amberDark, 700, "middle")}
    ${text(800, 780, "证据方向来自原论文 Figure 5–6；未对图中曲线做伪精确数字化。", 17, palette.muted, 400, "middle")}`,
  );
}

const files = {
  "01-bpr/paradigm-shift.svg": shiftFigure({
    title: "从预测数值到比较顺序 / BPR’s paradigm shift",
    subtitle: "隐式反馈没有评分；BPR 把问题改写成“同一用户更偏好哪一个”。",
    left: {
      kicker: "POINTWISE · 逐点",
      title: "拟合一个绝对分数",
      lines: ["输入：(u, i)", "目标：点击=1，未点=0", "风险：把沉默当作明确负反馈"],
    },
    right: {
      kicker: "PAIRWISE · 成对",
      title: "学习谁排在谁前面",
      lines: ["输入：(u, i, j)", "目标：s(u,i) > s(u,j)", "输出：个性化商品顺序"],
    },
    bridge: "BPR-Opt",
    footer: "改变的不是矩阵分解本身，而是训练目标终于与“排序”一致。",
  }),
  "01-bpr/toy-triple.svg": bprToy(),
  "01-bpr/mechanism.svg": pipelineFigure({
    title: "BPR-MF 从三元组到一次参数更新 / Data flow",
    subtitle: "以 batch size B、隐向量维度 d 为例；原论文的 LearnBPR 每次随机抽一个三元组。",
    steps: [
      { title: "采样", lines: ["(u, i, j)", "i：已观察", "j：未观察"] },
      { title: "查向量", lines: ["wᵤ ∈ Rᵈ", "hᵢ,hⱼ ∈ Rᵈ", "参数共享"] },
      { title: "比分差", lines: ["x̂ᵤᵢ = wᵤ·hᵢ", "x̂ᵤⱼ = wᵤ·hⱼ", "Δ = x̂ᵤᵢ−x̂ᵤⱼ"] },
      { title: "更新", lines: ["最大化 log σ(Δ)", "加入 L2 prior", "bootstrap SGD"] },
    ],
    formula: "BPR-Opt = Σ(u,i,j)∈Dₛ log σ(x̂ᵤᵢ − x̂ᵤⱼ) − λ‖Θ‖²",
    note: "单次 MF 更新约 O(d)；推理仍需给候选商品打分并取 Top-K。",
  }),
  "01-bpr/evidence.svg": evidenceCards({
    title: "BPR 论文真正证明了什么 / Evidence, not mythology",
    subtitle: "论文把同一优化准则应用到 MF 与 adaptive kNN；重点是“目标函数可迁移”。",
    cards: [
      { title: "目标对齐", lines: ["直接优化 pairwise ranking", "与 AUC 的比较结构一致", "不是先回归评分再排序"] },
      { title: "学习可行", lines: ["随机抽 (u,i,j)", "避免完整 O(|S||I|) 梯度", "bootstrap 比 user-wise 更快收敛"] },
      { title: "跨模型有效", lines: ["BPR-MF 与 BPR-kNN", "在论文数据上领先对应基线", "收益来自 criterion，不只架构"] },
    ],
    footer: "结论边界：它证明了优化“比较”很重要，没有证明所有未点击商品都是真负样本。",
  }),
  "02-gru4rec/paradigm-shift.svg": shiftFigure({
    title: "从静态用户向量到会更新的 session 状态 / GRU4Rec",
    subtitle: "同一个人此刻在逛什么，可能比她长期是谁更重要。",
    left: {
      kicker: "BPR-MF · 静态",
      title: "用户是一个固定坐标",
      lines: ["wᵤ 概括长期偏好", "事件顺序被压扁", "匿名新 session 没有 wᵤ"],
    },
    right: {
      kicker: "GRU4REC · 动态",
      title: "用户是正在变化的状态",
      lines: ["hₜ 读取当前点击", "门控决定保留与更新", "每一步预测下一件商品"],
    },
    bridge: "hₜ",
    footer: "推荐第一次认真回答：用户刚刚做过什么，会怎样改变下一步。",
  }),
  "02-gru4rec/toy-sequence.svg": sequenceToy({
    title: "把一次逛店过程展开成监督信号 / Session toy example",
    subtitle: "一条四步 session 可以产生三组 next-item 训练目标。",
    mode: "gru",
  }),
  "02-gru4rec/mechanism.svg": pipelineFigure({
    title: "GRU4Rec 的一步训练 / One recurrent step",
    subtitle: "原始实现使用 1-of-N 输入，并让同一 mini-batch 的其他正样本充当负样本。",
    steps: [
      { title: "并行 session", lines: ["B 条活跃 session", "xₜ ∈ Rᴮˣᴺ", "结束就换下一条"] },
      { title: "更新状态", lines: ["hₜ₋₁ ∈ Rᴮˣᵈ", "reset / update gates", "hₜ ∈ Rᴮˣᵈ"] },
      { title: "候选打分", lines: ["正确 next item", "batch 内其他 item", "score ∈ Rᴮˣᴮ"] },
      { title: "排序损失", lines: ["BPR 或 TOP1", "正样本高于负样本", "Adagrad 更新"] },
    ],
    formula: "hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ",
    note: "zₜ 决定这一步写入多少新兴趣；reset gate 决定构造候选状态时忘掉多少旧信息。",
  }),
  "02-gru4rec/evidence.svg": evidenceBars({
    title: "GRU4Rec 对 item-KNN 的关键提升 / ICLR 2016 Table 3",
    subtitle: "选择论文中 TOP1、1000 hidden units；指标均为越高越好。",
    max: 0.72,
    groups: [
      { label: "RSC15 Recall@20", values: [{ label: "Item-KNN", value: 0.5065, color: palette.line }, { label: "GRU4Rec", value: 0.6206, color: palette.violet }] },
      { label: "RSC15 MRR@20", values: [{ label: "Item-KNN", value: 0.2048, color: palette.line }, { label: "GRU4Rec", value: 0.2693, color: palette.violet }] },
      { label: "VIDEO Recall@20", values: [{ label: "Item-KNN", value: 0.5508, color: palette.line }, { label: "GRU4Rec", value: 0.6624, color: palette.violet }] },
      { label: "VIDEO MRR@20", values: [{ label: "Item-KNN", value: 0.3381, color: palette.line }, { label: "GRU4Rec", value: 0.3891, color: palette.violet }] },
    ],
  }),
  "03-sasrec/paradigm-shift.svg": shiftFigure({
    title: "从逐步传话到直接回看历史 / SASRec",
    subtitle: "RNN 把过去压进一个状态；self-attention 让当前位置直接选择相关历史。",
    left: {
      kicker: "RNN · RECURRENCE",
      title: "信息沿时间逐步传递",
      lines: ["h₁ → h₂ → … → hₜ", "最长依赖路径 O(n)", "训练难以在时间维并行"],
    },
    right: {
      kicker: "SELF-ATTENTION",
      title: "每一步直接查看过去",
      lines: ["query 与所有历史 key 比较", "因果 mask 遮住未来", "相关动作获得更大权重"],
    },
    bridge: "QKᵀ",
    footer: "SASRec 不假设“越近越重要”；它让数据决定哪些过去与下一步有关。",
  }),
  "03-sasrec/toy-attention.svg": sequenceToy({
    title: "历史很长，但真正相关的可能只有几步 / Toy attention",
    subtitle: "颜色条仅解释机制，不是论文中的真实个体 attention 权重。",
    mode: "sas",
  }),
  "03-sasrec/mechanism.svg": pipelineFigure({
    title: "SASRec 从 Item ID 到下一件商品 / Causal Transformer",
    subtitle: "n 为最大序列长度，d 为表示维度；每个位置都预测它后面的真实商品。",
    steps: [
      { title: "序列嵌入", lines: ["Item ID → M", "加 learned position", "E ∈ Rⁿˣᵈ"] },
      { title: "因果注意力", lines: ["Q,K,V ∈ Rⁿˣᵈ", "遮住 j > t", "A ∈ Rⁿˣⁿ"] },
      { title: "残差 + FFN", lines: ["逐位置非线性", "LayerNorm", "堆叠 2 blocks"] },
      { title: "下一项打分", lines: ["hₜ·mᵢ", "正项 + sampled negative", "BCE 训练"] },
    ],
    formula: "Attention(Q,K,V) = softmax(QKᵀ / √d + causal mask) V",
    note: "时间复杂度含 O(n²d)，但所有位置可并行；从任意历史到当前输出的路径长度是 O(1)。",
  }),
  "03-sasrec/evidence.svg": evidenceBars({
    title: "SASRec 在稀疏与稠密数据上都有效 / ICDM 2018 Table III",
    subtitle: "NDCG@10；灰色为论文中最强神经基线，紫色为 SASRec。",
    max: 0.7,
    groups: [
      { label: "Amazon Beauty", values: [{ label: "best neural", value: 0.2556, color: palette.line }, { label: "SASRec", value: 0.3219, color: palette.violet }] },
      { label: "Amazon Games", values: [{ label: "best neural", value: 0.4759, color: palette.line }, { label: "SASRec", value: 0.536, color: palette.violet }] },
      { label: "Steam", values: [{ label: "best neural", value: 0.5595, color: palette.line }, { label: "SASRec", value: 0.6306, color: palette.violet }] },
      { label: "MovieLens-1M", values: [{ label: "best neural", value: 0.5538, color: palette.line }, { label: "SASRec", value: 0.5905, color: palette.violet }] },
    ],
  }),
  "04-bert4rec/paradigm-shift.svg": shiftFigure({
    title: "从只看左边到在上下文中补空 / BERT4Rec",
    subtitle: "训练时随机遮住历史行为，让每个被遮位置同时利用左右两侧。",
    left: {
      kicker: "SASREC · LEFT-TO-RIGHT",
      title: "位置 t 只看 1…t",
      lines: ["目标：预测 t+1", "严格符合在线因果性", "同一表示看不到右侧上下文"],
    },
    right: {
      kicker: "BERT4REC · CLOZE",
      title: "在完整历史里恢复缺失项",
      lines: ["随机替换多个 [MASK]", "左右行为共同提供线索", "一条序列产生更多监督"],
    },
    bridge: "[MASK]",
    footer: "双向发生在训练历史内部；线上预测仍是在序列末尾追加一个 [MASK]。",
  }),
  "04-bert4rec/toy-cloze.svg": sequenceToy({
    title: "购物记录变成一道完形填空 / Cloze toy example",
    subtitle: "模型必须从左右上下文恢复被遮住的 Item ID。",
    mode: "bert",
  }),
  "04-bert4rec/mechanism.svg": pipelineFigure({
    title: "BERT4Rec 的训练与推理并不完全相同 / Train–serve flow",
    subtitle: "训练随机 mask 多个历史位置；推理只在序列末尾添加 [MASK] 预测下一项。",
    steps: [
      { title: "构造输入", lines: ["随机 mask 比例 ρ", "保留左右上下文", "S′ ∈ Rⁿ"] },
      { title: "双向编码", lines: ["L 层 Transformer", "multi-head attention", "Hᴸ ∈ Rⁿˣᵈ"] },
      { title: "恢复目标", lines: ["只读取 mask 位置", "softmax over items", "多位置 NLL"] },
      { title: "线上推荐", lines: ["历史末尾加 [MASK]", "读取最后 hidden", "Top-K item logits"] },
    ],
    formula: "L = − Σₘ∈masked log P(vₘ* | S′)",
    note: "ρ 太小监督不足，太大上下文被破坏；论文在四个数据集上分别调节最佳 mask 比例。",
  }),
  "04-bert4rec/evidence.svg": evidenceBars({
    title: "收益来自双向表示，也来自多位置 Cloze / CIKM 2019 Table 3",
    subtitle: "HR@10；“1 mask”隔离双向结构，“full”再加入多 mask 训练收益。",
    max: 0.8,
    groups: [
      { label: "Beauty", values: [{ label: "SASRec", value: 0.2653, color: palette.line }, { label: "BERT 1 mask", value: 0.294, color: palette.blue }, { label: "BERT full", value: 0.3025, color: palette.violet }] },
      { label: "MovieLens-1M", values: [{ label: "SASRec", value: 0.6629, color: palette.line }, { label: "BERT 1 mask", value: 0.6869, color: palette.blue }, { label: "BERT full", value: 0.697, color: palette.violet }] },
    ],
  }),
};

for (const [relative, contents] of Object.entries(files)) {
  const target = path.join(root, relative);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, contents);
}

console.log(`Generated ${Object.keys(files).length} foundation figures.`);
