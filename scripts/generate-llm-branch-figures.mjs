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
  `${text(80, 82, title, 42, c.ink, 650)}
  ${text(80, 124, subtitle, 21, c.muted)}`;

const svg = (title, desc, body) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc">
  <title id="title">${esc(title)}</title>
  <desc id="desc">${esc(desc)}</desc>
  <rect width="${W}" height="${H}" fill="${c.bg}"/>
  <g font-family="Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif">
    ${body}
  </g>
</svg>
`;

function shift({ title, subtitle, left, right, bridge, footer }) {
  return svg(
    title,
    subtitle,
    `${titleBlock(title, subtitle)}
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
    ${text(800, 785, footer, 24, c.amberDark, 650, "middle")}`,
  );
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
  const body = values
    .map((card, i) => {
      const x = start + i * (width + gap);
      const [fill, stroke, dark] = styles[i];
      return `${rect(x, 220, width, 420, fill, stroke, 28)}
      ${text(x + 34, 270, `0${i + 1}`, 17, dark, 750)}
      ${text(x + 34, 325, card.title, 27, c.ink, 650)}
      ${card.lines.map((value, j) => text(x + 34, 385 + j * 45, value, 19, c.muted)).join("\n")}`;
    })
    .join("\n");
  return svg(
    title,
    subtitle,
    `${titleBlock(title, subtitle)}
    ${body}
    ${rect(235, 705, 1130, 100, "#ffffff", c.line, 25)}
    ${text(800, 765, footer, 22, c.ink, 650, "middle")}`,
  );
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
  const flow = steps
    .map((step, i) => {
      const x = start + i * (width + gap);
      const [fill, stroke, dark] = styles[i];
      return `${rect(x, 220, width, 270, fill, stroke, 28)}
      ${text(x + 28, 265, `0${i + 1}`, 17, dark, 750)}
      ${text(x + 28, 317, step.title, 27, c.ink, 650)}
      ${step.lines.map((value, j) => text(x + 28, 365 + j * 39, value, 18, c.muted)).join("\n")}
      ${i < steps.length - 1 ? arrow(x + width + 8, 355, x + width + gap - 8, 355, styles[i + 1][1]) : ""}`;
    })
    .join("\n");
  return svg(
    title,
    subtitle,
    `${titleBlock(title, subtitle)}
    ${flow}
    ${rect(130, 565, 1340, 135, "#ffffff", c.line, 26)}
    ${text(800, 612, "核心公式 / central equation", 18, c.muted, 650, "middle")}
    ${text(800, 665, formula, 30, c.ink, 650, "middle")}
    ${rect(230, 750, 1140, 70, c.amberBg, c.amber, 22)}
    ${text(800, 794, note, 20, c.amberDark, 650, "middle")}`,
  );
}

function bars({ title, subtitle, groups, max, suffix = "", footer = "Source: original paper tables; values redrawn" }) {
  const chartX = 335;
  const chartW = 1060;
  const maxSeries = Math.max(...groups.map((group) => group.values.length));
  const groupH = maxSeries > 2 ? 176 : 142;
  const barH = 29;
  const body = groups
    .map((group, gi) => {
      const top = 188 + gi * groupH;
      return `${text(90, top + 29, group.label, 20, c.ink, 700)}
      ${group.values
        .map((series, si) => {
          const y = top + 42 + si * 43;
          const width = Math.max(3, (series.value / max) * chartW);
          return `${rect(chartX, y, chartW, barH, "#eee9df", "none", 15)}
          ${rect(chartX, y, width, barH, series.color, "none", 15)}
          ${text(chartX - 18, y + 21, series.label, 16, c.muted, 600, "end")}
          ${text(Math.min(chartX + width + 13, 1505), y + 21, `${series.value}${suffix}`, 16, c.ink, 700)}`;
        })
        .join("\n")}`;
    })
    .join("\n");
  return svg(
    title,
    subtitle,
    `${titleBlock(title, subtitle)}
    ${body}
    ${text(1510, 850, footer, 15, c.muted, 500, "end")}`,
  );
}

const files = {
  "05-p5/paradigm-shift.svg": shift({
    title: "从四个模型到一种语言接口 / P5",
    subtitle: "P5 把推荐的任务边界改写成 prompt 边界：同一模型、同一损失、不同问法。",
    left: {
      kicker: "TASK-SPECIFIC · 各做各的",
      title: "一个任务一个模型头",
      lines: ["评分：回归", "下一项：分类 / 排序", "解释：文本生成", "知识难以跨任务流动"],
    },
    right: {
      kicker: "TEXT-TO-TEXT · 统一接口",
      title: "输入文本，输出文本",
      lines: ["评分 → “4 stars”", "下一项 → “item_6”", "解释 → 一句话", "共享参数与语言目标"],
    },
    bridge: "prompt",
    footer: "P5 的关键贡献不是“会聊天”，而是把五类推荐任务放进同一个条件生成问题。",
  }),
  "05-p5/toy-prompts.svg": cards({
    title: "同一位用户，四种问法 / P5 toy prompts",
    subtitle: "历史固定为网球拍 → 攀岩鞋 → 羽毛球 → 泳镜；模板决定当前任务。",
    cards: [
      { title: "评分 / Rating", lines: ["小林会给 I03", "多少星？", "→ “5”"] },
      { title: "下一项 / Next", lines: ["看过 I01,I03,I05", "接下来是什么？", "→ “I06”"] },
      { title: "解释 / Explain", lines: ["为什么推荐 I03？", "结合用户与商品", "→ 一句话"] },
      { title: "直接推荐 / Direct", lines: ["I06 与 I07 中", "哪件更适合？", "→ “I06”"] },
    ],
    footer: "原始行为没有变；P5 用 personalized prompt 把它重组为 input–target token pairs。",
  }),
  "05-p5/mechanism.svg": pipeline({
    title: "P5：一个模型、一种损失、五个任务族 / Data flow",
    subtitle: "以 T5 式 encoder–decoder 为骨架；Item ID 也会被 SentencePiece 拆成子词。",
    steps: [
      { title: "模板实例化", lines: ["用户 / 商品 / 历史", "填入 prompt", "x ∈ tokensⁿ"] },
      { title: "双向编码", lines: ["token + position", "+ whole-word emb", "T ∈ Rⁿˣᵈ"] },
      { title: "自回归解码", lines: ["读取 encoder", "逐 token 生成", "P(yⱼ|y<ⱼ,x)"] },
      { title: "任务解释", lines: ["数字 / Yes-No", "Item ID / 文本", "greedy 或 beam"] },
    ],
    formula: "Lᴾ⁵ = − Σⱼ log Pθ(yⱼ | y<ⱼ, x)",
    note: "统一的是数据格式与 token-level NLL；不同任务仍需要不同模板、解码与评估协议。",
  }),
  "05-p5/evidence.svg": bars({
    title: "统一生成并未牺牲下一项推荐 / RecSys 2022 Table 3",
    subtitle: "NDCG@5；P5-B 使用 Prompt 2-3，与论文中此前最强结果比较。",
    max: 0.065,
    groups: [
      { label: "Sports", values: [{ label: "S³-Rec", value: 0.0161, color: c.line }, { label: "P5-B", value: 0.0296, color: c.violet }] },
      { label: "Beauty", values: [{ label: "SASRec", value: 0.0249, color: c.line }, { label: "P5-B", value: 0.0379, color: c.violet }] },
      { label: "Toys", values: [{ label: "SASRec", value: 0.0306, color: c.line }, { label: "P5-S", value: 0.0567, color: c.violet }] },
    ],
  }),

  "06-m6-rec/paradigm-shift.svg": shift({
    title: "从学术统一到工业开放任务 / M6-Rec",
    subtitle: "统一模型要进入真实系统，必须同时回答任务覆盖、样本效率与在线延迟。",
    left: {
      kicker: "P5 · BENCHMARK",
      title: "五个任务族共享 T5",
      lines: ["公开数据集", "离线生成与排序", "主要验证统一范式", "服务代价不是主角"],
    },
    right: {
      kicker: "M6-REC · PRODUCTION",
      title: "召回到创作都要落地",
      lines: ["检索 / CTR / 解释", "对话 / 内容生成", "云端与手机部署", "缓存、蒸馏、早退"],
    },
    bridge: "serve",
    footer: "“foundation model”在工业里不是参数越多越好，而是更多任务能以可接受成本复用。",
  }),
  "06-m6-rec/toy-open-tasks.svg": cards({
    title: "一份用户文本，贯穿推荐漏斗 / M6-Rec toy tasks",
    subtitle: "小林的体育行为被写成普通文本；任务改变，底座不变。",
    cards: [
      { title: "召回", lines: ["用户文本 → 向量 x", "商品文本 → 向量 y", "kNN 找候选"] },
      { title: "排序", lines: ["历史 + 候选 I06", "输出点击概率", "Yes / No option"] },
      { title: "解释", lines: ["推荐泳镜，因为…", "生成个性化理由", "语言质量可评估"] },
      { title: "创作", lines: ["根据小众兴趣", "生成搜索词 / 标题", "甚至接图像生成"] },
    ],
    footer: "同一个底座并不意味着同一种头部：检索用向量，CTR 用分类，开放任务用生成。",
  }),
  "06-m6-rec/mechanism.svg": pipeline({
    title: "M6-Rec 怎样把大模型塞进实时链路 / Late interaction",
    subtitle: "请求拆成可复用 segment；前 21 层离线缓存，在线只运行最后几层交互。",
    steps: [
      { title: "文本分段", lines: ["画像 / 单次点击", "候选商品", "segment tokens"] },
      { title: "深层预计算", lines: ["前 L′ 层", "各段独立编码", "结果可缓存"] },
      { title: "在线拼接", lines: ["加入 segment emb", "读取最新行为", "candidate 组合"] },
      { title: "浅层交互", lines: ["最后 L−L′ 层", "soft options", "实时预测"] },
    ],
    formula: "Lretrieval = −log exp(xᵀy/τ) / [exp(xᵀy/τ)+Σy′ exp(xᵀy′/τ)]",
    note: "检索用对比学习；排序与生成沿用语言建模。统一底座之上仍保留任务适配。",
  }),
  "06-m6-rec/evidence.svg": bars({
    title: "性能与延迟必须一起看 / M6-Rec Tables 2, 3, 6",
    subtitle: "上两组为效果；最后一组为毫秒延迟，刻度仅用于同图比较。",
    max: 0.82,
    groups: [
      { label: "TaoProduct AUC", values: [{ label: "DIN", value: 0.7611, color: c.line }, { label: "M6-Rec", value: 0.7995, color: c.violet }] },
      { label: "MiniApp HR@100", values: [{ label: "TwinBERT", value: 0.696, color: c.line }, { label: "M6-Rec", value: 0.741, color: c.violet }] },
      { label: "Latency / 70 ms", values: [{ label: "full 57ms", value: 0.57, color: c.blue }, { label: "late 16ms", value: 0.16, color: c.green }] },
    ],
    footer: "Source: original paper Tables 2, 3, 6; latency normalized by 100 for plotting",
  }),

  "07-tallrec/paradigm-shift.svg": shift({
    title: "会语言，不等于会判断偏好 / TALLRec",
    subtitle: "通用 LLM 的预训练目标与推荐任务错位；少量 recommendation instruction 才是桥。",
    left: {
      kicker: "ZERO-SHOT LLM",
      title: "懂电影，却不懂点击规律",
      lines: ["世界知识丰富", "能理解自然语言", "推荐 AUC 接近随机", "训练语料缺少偏好监督"],
    },
    right: {
      kicker: "REC-TUNED LLM",
      title: "把偏好变成可学习指令",
      lines: ["喜欢 / 不喜欢历史", "目标商品文本", "输出 Yes 或 No", "LoRA 只更新低秩参数"],
    },
    bridge: "SFT",
    footer: "TALLRec 的核心证据：能力不是靠 prompt 自动“唤醒”，而是靠少量域内监督完成对齐。",
  }),
  "07-tallrec/toy-rec-tuning.svg": cards({
    title: "一条推荐日志怎样变成 instruction / Toy sample",
    subtitle: "标签必须来自行为或评分，模型不能自己编“喜欢”。",
    cards: [
      { title: "历史分组", lines: ["喜欢：I01 网球拍", "喜欢：I03 攀岩鞋", "不喜欢：I07 瑜伽垫"] },
      { title: "目标商品", lines: ["Target: I06 泳镜", "标题 + 简介", "用户尚未见过"] },
      { title: "指令输入", lines: ["根据历史判断", "会不会喜欢 I06？", "只回答 Yes / No"] },
      { title: "监督输出", lines: ["真实行为：点击", "Target = “Yes”", "token NLL 训练"] },
    ],
    footer: "输入是自然语言，但监督仍是推荐数据；这正是 language–recommendation gap 的落点。",
  }),
  "07-tallrec/mechanism.svg": pipeline({
    title: "TALLRec 的两阶段轻量对齐 / Alpaca + rec tuning",
    subtitle: "先保持通用指令跟随，再用推荐样本对齐；两阶段都只训练 LoRA 参数。",
    steps: [
      { title: "通用指令", lines: ["Alpaca data", "(x,y) 文本对", "学习遵循任务"] },
      { title: "LoRA 更新", lines: ["冻结 Φ", "ΔΦ = BA", "rank r ≪ d"] },
      { title: "推荐指令", lines: ["历史 + target", "Yes / No label", "少量 rec samples"] },
      { title: "偏好概率", lines: ["读取 Yes/No", "用于二分类", "AUC 评估"] },
    ],
    formula: "maxΘ Σ(x,y) Σₜ log PΦ+Θ(yₜ | x, y<ₜ)",
    note: "论文报告 LLaMA-7B 可在单张 RTX 3090 上训练；效率来自 PEFT，不是小模型。",
  }),
  "07-tallrec/evidence.svg": bars({
    title: "64 条样本已经改变推荐能力 / RecSys 2023 Table 3",
    subtitle: "AUC×100；传统模型和 TALLRec 使用相同 few-shot 样本量。",
    max: 75,
    groups: [
      { label: "Movie · 64-shot", values: [{ label: "best baseline", value: 51.71, color: c.line }, { label: "TALLRec", value: 67.48, color: c.violet }] },
      { label: "Book · 64-shot", values: [{ label: "best baseline", value: 50.06, color: c.line }, { label: "TALLRec", value: 60.39, color: c.violet }] },
      { label: "Movie · 256-shot", values: [{ label: "best baseline", value: 54.2, color: c.line }, { label: "TALLRec", value: 71.98, color: c.violet }] },
    ],
  }),

  "08-rella/paradigm-shift.svg": shift({
    title: "上下文窗口够长，注意力仍可能失焦 / ReLLa",
    subtitle: "LLM 在数百 token 就出现“终身序列不理解”，问题不是塞不下，而是相关信号太稀。",
    left: {
      kicker: "RECENT-K",
      title: "把最近历史全部塞进去",
      lines: ["顺序自然", "主题可能高度混杂", "长度增加后 AUC 反降", "仍远低于 2048 token 上限"],
    },
    right: {
      kicker: "RELEVANT-K",
      title: "围绕目标检索历史",
      lines: ["目标商品作 query", "语义相似度选 Top-K", "输入长度基本不变", "信噪比更高"],
    },
    bridge: "SUBR",
    footer: "ReLLa 把问题从“能装多少历史”改成“为当前候选挑哪段历史”。",
  }),
  "08-rella/toy-retrieval.svg": cards({
    title: "推荐泳镜时，哪些历史值得带进 prompt？ / SUBR toy",
    subtitle: "完整历史包含八件商品；预算只允许保留四件。",
    cards: [
      { title: "目标 / Query", lines: ["I06 泳镜", "游泳 · 防雾", "编码成 vtarget"] },
      { title: "最近四件", lines: ["蛋白棒 / 粉袋", "羽毛球 / 瑜伽垫", "主题混杂"] },
      { title: "语义检索", lines: ["I06 与每件历史", "计算 cosine", "按相关度排序"] },
      { title: "相关四件", lines: ["泳帽 / 运动毛巾", "羽毛球 / 网球拍", "再写入 prompt"] },
    ],
    footer: "示例只解释机制；原论文用 LLM hidden states + PCA 得到 512 维商品语义向量。",
  }),
  "08-rella/mechanism.svg": pipeline({
    title: "ReLLa：先改数据，再调模型 / SUBR + ReiT",
    subtitle: "零样本只做检索；少样本把原始与检索增强样本混合后 instruction tuning。",
    steps: [
      { title: "商品编码", lines: ["描述文本 → LLM", "last-layer mean pool", "PCA → v ∈ R⁵¹²"] },
      { title: "行为检索", lines: ["target 作 query", "cosine similarity", "Top-K relevant"] },
      { title: "混合数据", lines: ["N 原始样本", "+ N 检索样本", "ReiT 共 2N"] },
      { title: "Yes / No", lines: ["causal LM tuning", "读取两 token logits", "得到 CTR score"] },
    ],
    formula: "ŷ = exp(sYes) / [exp(sYes) + exp(sNo)]",
    note: "SUBR 不增加上下文长度；它改变的是选择函数。ReiT 的 pattern enrichment 还充当正则化。",
  }),
  "08-rella/evidence.svg": bars({
    title: "不到 10% 数据，超过全量传统模型 / WWW 2024 Table 2",
    subtitle: "AUC；灰色为全量训练的最强传统基线 SIM，紫色为 ReLLa (<10%)。",
    max: 0.9,
    groups: [
      { label: "BookCrossing", values: [{ label: "SIM full", value: 0.7541, color: c.line }, { label: "ReLLa", value: 0.7575, color: c.violet }] },
      { label: "MovieLens-1M", values: [{ label: "SIM full", value: 0.7992, color: c.line }, { label: "ReLLa", value: 0.8033, color: c.violet }] },
      { label: "MovieLens-25M", values: [{ label: "SIM full", value: 0.8344, color: c.line }, { label: "ReLLa", value: 0.8477, color: c.violet }] },
    ],
  }),

  "09-llara/paradigm-shift.svg": shift({
    title: "商品有两种知识：它是什么，谁会喜欢 / LLaRA",
    subtitle: "文本 token 调用世界知识；传统推荐器的 ID embedding 保存协同行为。二者缺一不可。",
    left: {
      kicker: "TEXT ONLY",
      title: "LLM 知道商品的含义",
      lines: ["“泳镜”与游泳有关", "可解释标题与描述", "冷商品也有语义", "难学匿名共现规律"],
    },
    right: {
      kicker: "HYBRID TOKEN",
      title: "语义与行为并排输入",
      lines: ["title tokens", "+ projected ID embedding", "LLM 空间内融合", "候选与历史同样表示"],
    },
    bridge: "align",
    footer: "LLaRA 不是用 LLM 替换推荐器，而是把推荐器当作一种新模态接入 LLM。",
  }),
  "09-llara/toy-hybrid.svg": cards({
    title: "“泳镜”后面多出的一个 token 是什么？ / Hybrid item",
    subtitle: "文本 token 与行为 token 不相加覆盖，而是拼接后共同进入 prompt。",
    cards: [
      { title: "文本侧", lines: ["“I06 泳镜”", "LLM tokenizer", "embᵗ：世界知识"] },
      { title: "行为侧", lines: ["SASRec item emb", "eˢ ∈ Rᵈ", "共现与顺序模式"] },
      { title: "投影器", lines: ["two-layer MLP", "Proj(eˢ)", "映射到 LLM 维度"] },
      { title: "混合表示", lines: ["[embᵗ ; embˢ]", "写入历史与候选", "生成一个候选标题"] },
    ],
    footer: "投影解决维度与模态接口；它不能自动保证两种知识在语义上完美对齐。",
  }),
  "09-llara/mechanism.svg": pipeline({
    title: "LLaRA 为什么要从易到难 / Curriculum prompt tuning",
    subtitle: "直接注入行为 token 会让 LLM 同时学习推荐任务和陌生模态；课程学习拆开难度。",
    steps: [
      { title: "预训练推荐器", lines: ["GRU / Caser / SAS", "得到 eᵢˢ", "保留行为规律"] },
      { title: "Easy prompt", lines: ["商品标题 + [PH]", "纯文本输入", "先学任务格式"] },
      { title: "渐进替换", lines: ["p(τ)=τ/T", "[PH] → Proj(eᵢˢ)", "LoRA + projector"] },
      { title: "Hard prompt", lines: ["全量 hybrid tokens", "候选集合生成", "next item title"] },
    ],
    formula: "p(hard at τ) = τ / T,   Lτ = (1−Iτ)Leasy + IτLhard",
    note: "训练越往后，采样 hard prompt 的概率越高；它是平滑迁移，不是固定两阶段切换。",
  }),
  "09-llara/evidence.svg": bars({
    title: "融合后，行为与世界知识都能贡献 / SIGIR 2024 Table 2",
    subtitle: "HitRatio@1；LLaRA 取各数据集最佳变体，对比最强传统或 LLM-based baseline。",
    max: 0.55,
    groups: [
      { label: "MovieLens", values: [{ label: "TALLRec", value: 0.3895, color: c.line }, { label: "LLaRA", value: 0.4737, color: c.violet }] },
      { label: "Steam", values: [{ label: "TALLRec", value: 0.4637, color: c.line }, { label: "LLaRA", value: 0.4949, color: c.violet }] },
      { label: "LastFM", values: [{ label: "TALLRec", value: 0.418, color: c.line }, { label: "LLaRA", value: 0.4508, color: c.violet }] },
    ],
  }),
};

for (const [relative, contents] of Object.entries(files)) {
  const target = path.join(root, relative);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, contents);
}

console.log(`Generated ${Object.keys(files).length} LLM-branch figures.`);
