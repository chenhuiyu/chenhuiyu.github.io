#!/usr/bin/env node
// Original SVG illustrations. Run from any directory; no graphics dependency.
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'public/blog/gated-deltanet');
const p = {
  paper: '#f7f5f0', ink: '#16262b', muted: '#657477', line: '#d5dcd8',
  teal: '#087f83', tealBg: '#e0eeeb', rose: '#b84e66', roseBg: '#f7e5e8',
  blue: '#4565ad', blueBg: '#e8edf8', amber: '#a16a27', amberBg: '#f3ead9',
  white: '#fffefa', gray: '#9aa6a5',
};
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const t = (x, y, s, size = 28, fill = p.ink, weight = 400, anchor = 'start', extra = '') =>
  `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" font-weight="${weight}" text-anchor="${anchor}" ${extra}>${esc(s)}</text>`;
const lines = (x, y, ss, size = 28, fill = p.muted, gap = 42, weight = 400) =>
  ss.map((s, i) => t(x, y + i * gap, s, size, fill, weight)).join('');
const box = (x, y, w, h, fill = p.white, stroke = p.line, r = 18) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`;
const line = (x1, y1, x2, y2, color = p.line, width = 2, extra = '') =>
  `<path d="M${x1},${y1} L${x2},${y2}" fill="none" stroke="${color}" stroke-width="${width}" ${extra}/>`;
const arrow = (x1, y1, x2, y2, color = p.teal, width = 3) =>
  line(x1, y1, x2, y2, color, width, `marker-end="url(#arrow-${Object.keys(p).find(k => p[k] === color) ?? 'ink'})"`);
const choose = (lang, zh, en) => lang === 'zh' ? zh : en;
const formula = (x, y, s, size = 31, color = p.ink, anchor = 'start') =>
  t(x, y, s, size, color, 400, anchor, 'font-family="DejaVu Sans, Arial, sans-serif"');

function page(lang, id, title, subtitle, body, height = 950, note = '') {
  const marker = Object.entries(p).map(([k, v]) => `<marker id="arrow-${k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="${v}"/></marker>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="${height}" viewBox="0 0 1200 ${height}" role="img" aria-labelledby="title desc" lang="${lang === 'zh' ? 'zh-CN' : 'en'}">
<title id="title">${esc(title)}</title><desc id="desc">${esc(subtitle + ' ' + note)}</desc>
<defs>${marker}<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="${p.line}" opacity=".5"/></pattern></defs>
<rect width="1200" height="${height}" fill="${p.paper}"/><rect width="1200" height="${height}" fill="url(#grid)"/>
<g font-family="Noto Sans CJK SC, PingFang SC, Microsoft YaHei, DejaVu Sans, sans-serif">
${t(56, 53, 'HUIYU CHEN  /  PAPER NOTES', 18, p.muted, 600, 'start', 'letter-spacing="2"')}
${t(1144, 53, id, 18, p.teal, 600, 'end', 'letter-spacing="2"')}
${t(56, 116, title, lang === 'zh' ? 42 : 40, p.ink, 700)}
${t(56, 162, subtitle, 25, p.muted)}
${body}
${line(56, height - 76, 1144, height - 76)}
${t(56, height - 35, note || choose(lang, '机制示意 · 非模型激活 / 非新增实验', 'Mechanism schematic · no model activations or new experiments'), 19, p.muted)}
${t(1144, height - 35, 'GATED DELTANET', 18, p.muted, 600, 'end', 'letter-spacing="2"')}
</g></svg>\n`;
}

function matrix(x, y, values, { cell = 68, fill = p.tealBg, color = p.teal, labels = true } = {}) {
  const cols = values[0].length;
  let s = '';
  values.forEach((row, i) => row.forEach((v, j) => {
    const alpha = Math.min(1, Math.abs(v));
    s += `<rect x="${x + j * cell}" y="${y + i * cell}" width="${cell - 5}" height="${cell - 5}" rx="8" fill="${fill}"/>`;
    if (!labels) s += `<rect x="${x + j * cell + 6}" y="${y + i * cell + 6}" width="${cell - 17}" height="${cell - 17}" rx="5" fill="${color}" opacity="${.08 + .82 * alpha}"/>`;
    if (labels) s += formula(x + j * cell + (cell - 5) / 2, y + i * cell + cell / 2 + 9, Number.isInteger(v) ? String(v) : v.toFixed(1), 29, v < 0 ? p.rose : p.ink, 'middle');
  }));
  return { svg: s, width: cols * cell, height: values.length * cell };
}

function cover(lang) {
  const title = choose(lang, 'Gated DeltaNet：记忆如何更新', 'Gated DeltaNet: how memory updates');
  let s = box(56, 215, 1088, 590, p.ink, p.ink, 26);
  s += t(96, 270, 'ICLR 2025  ·  VISUAL PAPER GUIDE', 20, '#b2c5c4', 600, 'start', 'letter-spacing="2"');
  s += lines(96, 360, choose(lang, ['先遗忘。', '再修正。', '然后继续。'], ['Decay.', 'Correct.', 'Continue.']), lang === 'zh' ? 68 : 72, '#fffefa', 94, 700);
  s += lines(98, 687, choose(lang, ['一个固定大小的状态矩阵，', '两种互补的记忆控制。'], ['One fixed-size state.', 'Two kinds of memory control.']), 27, '#b2c5c4', 44);
  const cells = Array.from({ length: 8 }, (_, i) => Array.from({ length: 7 }, (_, j) => .12 + .85 * Math.abs(Math.sin(i * 1.63 + j * .81))));
  s += `<g transform="translate(737 334) rotate(-8 150 160)">${matrix(0, 0, cells, { cell: 43, labels: false, fill: '#284146', color: '#6ec3b6' }).svg}</g>`;
  s += `<path d="M650 650 C790 650 725 460 865 460 C1010 460 900 300 1080 300" fill="none" stroke="#df91a1" stroke-width="5"/>`;
  s += box(648, 685, 442, 76, '#294247', '#426067', 12);
  s += formula(869, 735, 'Sₜ = αₜSₜ₋₁ + βₜ(vₜ − αₜSₜ₋₁kₜ)kₜᵀ', 22, '#fffefa', 'middle');
  return page(lang, '00 / COVER', title, choose(lang, '从线性注意力到门控 Delta 更新，再到 GPU 上的分块训练。', 'From linear attention to gated delta updates and chunkwise training.'), s, 940);
}

function storage(lang) {
  let s = '';
  s += box(56, 213, 1088, 255);
  s += t(84, 261, choose(lang, '逐个保存历史 K / V', 'Keep every historical K / V'), 32, p.blue, 700);
  for (let i = 0; i < 8; i++) {
    const x = 89 + i * 99;
    s += box(x, 296, 76, 68, p.blueBg, 'none', 10) + formula(x + 38, 339, `k${i + 1}`, 27, p.blue, 'middle');
    s += box(x, 372, 76, 56, p.roseBg, 'none', 10) + formula(x + 38, 408, `v${i + 1}`, 27, p.rose, 'middle');
  }
  s += t(922, 333, '…', 46, p.muted) + formula(922, 397, 'T(dₖ+dᵥ)', 27, p.ink);
  s += box(56, 498, 1088, 300);
  s += t(84, 548, choose(lang, '将历史写入一个状态矩阵', 'Write history into one state matrix'), 32, p.teal, 700);
  for (let i = 0; i < 4; i++) {
    const x = 90 + i * 105;
    s += box(x, 594, 78, 95, i === 3 ? p.roseBg : p.blueBg, 'none', 10);
    s += formula(x + 39, 634, `k${i + 1}`, 26, p.blue, 'middle') + formula(x + 39, 674, `v${i + 1}`, 26, p.rose, 'middle');
  }
  s += arrow(520, 644, 641, 644);
  const m = Array.from({ length: 4 }, (_, i) => Array.from({ length: 5 }, (_, j) => Math.abs(Math.sin(i + j * .6))));
  s += matrix(681, 565, m, { cell: 43, labels: false }).svg;
  s += formula(1001, 636, 'Sₜ', 42, p.teal, 'middle') + formula(1001, 681, 'dᵥ × dₖ', 27, p.ink, 'middle');
  s += t(90, 775, choose(lang, '状态大小固定；能保留的信息仍受维度和更新规则约束。', 'Fixed storage; retained information still depends on capacity and updates.'), 26, p.muted);
  return page(lang, '01 / STORAGE', choose(lang, '历史越来越长，状态仍然一样大', 'Longer history, same-sized state'), choose(lang, '按单层、单头计算元素数；忽略其他缓存、归一化项和精度差异。', 'Elements per layer and head; other state and normalization terms omitted.'), s, 930);
}

function update(lang) {
  let s = box(56, 211, 1088, 103, p.ink, p.ink, 16);
  s += formula(600, 255, 'k = q = [1, 0]ᵀ     v = [0, 1]ᵀ     α = 0.8     β = 0.5', 29, p.white, 'middle');
  s += t(600, 293, choose(lang, '手算示例 · 单头、二维状态、单位长度 key', 'Worked arithmetic · one head, two-dimensional state, unit key'), 22, '#bed1d0', 400, 'middle');
  const rows = [
    { y: 348, n: '01', title: choose(lang, '衰减旧状态', 'Decay the old state'), sub: choose(lang, '所有方向先乘 α', 'Every direction is scaled by α'), left: [[1, 0], [0, 1]], right: [[.8, 0], [0, .8]], op: '× 0.8', fill: p.blueBg, color: p.blue },
    { y: 569, n: '02', title: choose(lang, '构造校正项', 'Build the correction'), sub: 'e = v − αSₜ₋₁k', left: [[-.8], [1]], right: [[-.4, 0], [.5, 0]], op: '× βkᵀ', fill: p.roseBg, color: p.rose },
    { y: 790, n: '03', title: choose(lang, '得到新状态', 'Obtain the new state'), sub: choose(lang, '用 q 读出当前关联', 'Read the current association with q'), left: [[.4, 0], [.5, .8]], right: [[.4], [.5]], op: '× q', fill: p.tealBg, color: p.teal },
  ];
  for (const row of rows) {
    s += box(56, row.y, 1088, 195);
    s += t(84, row.y + 48, row.n, 22, row.color, 700);
    s += t(144, row.y + 48, row.title, 31, p.ink, 700);
    s += t(144, row.y + 97, row.sub, 24, p.muted);
    s += matrix(598, row.y + 31, row.left, { cell: 65, fill: row.fill, color: row.color }).svg;
    s += formula(806, row.y + 101, row.op, 28, row.color, 'middle');
    s += arrow(773, row.y + 128, 840, row.y + 128, row.color, 2);
    s += matrix(898, row.y + 31, row.right, { cell: 65, fill: row.fill, color: row.color }).svg;
  }
  return page(lang, '03 / UPDATE', choose(lang, '把一次状态更新拆成三个动作', 'One state update, three actions'), choose(lang, '校正项针对当前 key；与 key 正交的方向仅受全局衰减影响。', 'The correction targets the current key; orthogonal directions only decay.'), s, 1110, choose(lang, '原创手算示例 · 数值来自给定公式，不是训练结果', 'Original worked example · arithmetic from the recurrence, not trained results'));
}

function rules(lang) {
  let s = '';
  const cards = [
    { x: 56, y: 218, label: 'Linear attention', color: p.gray, bg: '#e9eeeb', f: 'S ← S + vkᵀ', text: choose(lang, ['累加新的关联', '每次都继续写入'], ['Accumulate associations', 'Keep adding each new write']), decay: 1, delta: false },
    { x: 616, y: 218, label: 'Mamba2 · simplified', color: p.blue, bg: p.blueBg, f: 'S ← αS + vkᵀ', text: choose(lang, ['先整体衰减', '再加上新的关联'], ['Decay the entire state', 'Add the incoming association']), decay: .25, delta: false },
    { x: 56, y: 560, label: 'DeltaNet', color: p.rose, bg: p.roseBg, f: 'S ← S + β(v − Sk)kᵀ', text: choose(lang, ['根据当前预测误差', '修正 key 方向的关联'], ['Use prediction errors', 'Correct a key association']), decay: 1, delta: true },
    { x: 616, y: 560, label: 'Gated DeltaNet', color: p.teal, bg: p.tealBg, f: 'S ← αS + β(v − αSk)kᵀ', text: choose(lang, ['整体衰减 + 定向校正', '两种控制各司其职'], ['Decay globally', 'Correct along the current key']), decay: .25, delta: true },
  ];
  for (const c of cards) {
    s += box(c.x, c.y, 528, 310, p.white, p.line, 20);
    s += `<path d="M${c.x + 20} ${c.y} H${c.x + 508}" stroke="${c.color}" stroke-width="5"/>`;
    s += t(c.x + 28, c.y + 52, c.label, 29, c.color, 700);
    s += formula(c.x + 28, c.y + 108, c.f, 26, p.ink);
    const vals = Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => c.delta && j === 1 ? .95 : c.decay * (.25 + Math.abs(Math.sin(i + j)) * .65)));
    s += matrix(c.x + 32, c.y + 146, vals, { cell: 30, labels: false, fill: c.bg, color: c.color }).svg;
    s += lines(c.x + 178, c.y + 188, c.text, 23, p.muted, 40);
  }
  return page(lang, '02 / MECHANISMS', choose(lang, '两个旋钮，解决两种不同的问题', 'Two controls, two different jobs'), choose(lang, 'α 控制整体遗忘；β 控制沿 key 方向的写入强度。', 'α controls global decay; β controls the strength of the key-directed write.'), s, 1010, choose(lang, '矩阵色块为机制示意 · Mamba2 省略完整参数化', 'Matrix colors are schematic · full Mamba2 parameterization omitted'));
}

function chunks(lang) {
  let s = box(56, 212, 1088, 168);
  s += t(84, 261, choose(lang, '跨块：传递固定大小的状态', 'Across chunks: propagate a fixed-size state'), 30, p.teal, 700);
  for (let i = 0; i < 3; i++) {
    const x = 95 + i * 350;
    s += box(x, 288, 250, 60, i === 1 ? p.tealBg : p.blueBg, 'none', 10);
    s += formula(x + 125, 328, `chunk ${i + 1}`, 29, i === 1 ? p.teal : p.blue, 'middle');
    if (i < 2) s += arrow(x + 265, 319, x + 332, 319);
  }
  s += box(56, 414, 1088, 475);
  s += t(84, 468, choose(lang, '块内：把顺序依赖改写为三角系统', 'Inside a chunk: encode dependencies in a triangular system'), 29, p.ink, 700);
  const startX = 99, startY = 521, cell = 40;
  for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
    s += box(startX + j * cell, startY + i * cell, cell - 4, cell - 4, j > i ? '#ecefe9' : i === j ? p.teal : p.tealBg, 'none', 6);
    if (i === j) s += t(startX + j * cell + 18, startY + i * cell + 26, '1', 21, p.white, 600, 'middle');
  }
  s += formula(219, 801, 'A ∈ ℝᶜˣᶜ', 28, p.teal, 'middle');
  s += arrow(366, 641, 448, 641);
  s += box(477, 548, 596, 150, p.blueBg, 'none', 16);
  s += formula(775, 602, 'A = I + strictLower[B(Γ ⊙ KKᵀ)]', 28, p.blue, 'middle');
  s += formula(775, 658, 'AŨ = BV       B = diag(β)', 30, p.blue, 'middle');
  s += lines(478, 755, choose(lang, ['求解变换后的写入，再以矩阵乘法计算块输出。', '初始状态的贡献另行合并；因果顺序仍然保留。'], ['Solve transformed writes; use matrix multiplications.', 'Merge the incoming state separately; causality is preserved.']), 24, p.muted, 42);
  return page(lang, '04 / TRAINING', choose(lang, '递推公式，也可以高效训练', 'A recurrence can still train efficiently'), choose(lang, '分块改写保留同一更新规则；不必构造全序列的稠密注意力矩阵。', 'Chunkwise reformulation preserves the update without dense full-sequence attention.'), s, 1020, choose(lang, '根据论文 §3.3 重绘 · C 为固定块长，Γ 为块内衰减掩码', 'Redrawn from §3.3 · C is fixed chunk length; Γ is the decay-aware mask'));
}

// Author-reported Table 3 Avg. and Table 4 Avg.; arXiv v3, 2025-03-06.
// Scores are transcribed, not generated by the illustration script.
const scores = [
  { name: 'Mamba2', reasoning: 54.89, retrieval: 29.8, color: p.gray },
  { name: 'DeltaNet', reasoning: 52.14, retrieval: 26.2, color: p.gray },
  { name: 'Gated DeltaNet', reasoning: 55.32, retrieval: 30.6, color: p.teal },
  { name: 'Transformer++', reasoning: 52.25, retrieval: 37.0, color: p.blue },
  { name: 'Gated DeltaNet-H1', reasoning: 56.40, retrieval: 39.0, color: p.rose },
  { name: 'Gated DeltaNet-H2', reasoning: 56.18, retrieval: 40.1, color: p.rose },
];

function evidence(lang) {
  let s = '';
  const panels = [
    { y: 216, key: 'reasoning', max: 60, title: choose(lang, '常识推理 · 表 3 Avg.', 'Commonsense reasoning · Table 3 Avg.'), decimals: 2 },
    { y: 660, key: 'retrieval', max: 45, title: choose(lang, '真实文本检索 · 表 4 Avg.', 'Real-world retrieval · Table 4 Avg.'), decimals: 1 },
  ];
  for (const panel of panels) {
    s += box(56, panel.y, 1088, 410);
    s += t(84, panel.y + 48, panel.title, 31, p.ink, 700);
    const x0 = 430, width = 605, top = panel.y + 90;
    for (let tick = 0; tick <= panel.max; tick += 15) {
      const x = x0 + width * tick / panel.max;
      s += line(x, top - 5, x, top + 269, p.line, 1);
      s += t(x, top + 304, String(tick), 22, p.muted, 400, 'middle');
    }
    for (let i = 0; i < scores.length; i++) {
      const row = scores[i], y = top + i * 45;
      s += t(84, y + 25, row.name, 25, row.color === p.gray ? p.muted : row.color, row.color === p.gray ? 400 : 600);
      const w = width * row[panel.key] / panel.max;
      s += box(x0, y + 4, w, 25, row.color, 'none', 5);
      s += t(x0 + w + 12, y + 26, row[panel.key].toFixed(panel.decimals), 23, p.ink, 600);
    }
  }
  s += lines(84, 1122, choose(lang, ['固定训练设置：1.3B 参数，100B FineWeb-Edu tokens，训练长度 4K。', '检索评测输入截断至 2K。两种均分属于不同任务集合，分别解读。'], ['Training setup: 1.3B parameters, 100B FineWeb-Edu tokens, 4K training length.', 'Retrieval inputs truncated to 2K. The two averages cover different task sets.']), 24, p.muted, 40);
  return page(lang, '05 / EVIDENCE', choose(lang, '看见进步，也看见混合架构的作用', 'Read the gains, then read the hybrids'), choose(lang, '原作者报告的分数，按零起点重绘；两项指标均为越高越好。', 'Author-reported scores, redrawn from zero; higher is better for both metrics.'), s, 1300, choose(lang, '数据来源：论文 v3，表 3 / 表 4 · 非本站复现', 'Source: paper v3, Tables 3 / 4 · no independent reproduction'));
}

await mkdir(out, { recursive: true });
for (const lang of ['zh', 'en']) {
  for (const [name, render] of Object.entries({ cover, storage, update, rules, chunks, evidence })) {
    await writeFile(path.join(out, `${name}-${lang}.svg`), render(lang));
  }
}
await writeFile(path.join(out, 'sources.json'), JSON.stringify({
  paper: 'Gated Delta Networks: Improving Mamba2 with Delta Rule',
  authors: ['Songlin Yang', 'Jan Kautz', 'Ali Hatamizadeh'],
  version: 'arXiv:2412.06464v3 (2025-03-06)',
  url: 'https://arxiv.org/html/2412.06464v3',
  checked: '2026-11-02',
  illustrations: 'Original schematics and worked arithmetic; not measured model activations.',
  evidence: {
    provenance: 'Author-reported values; not independently reproduced.',
    setup: '1.3B parameters; 100B FineWeb-Edu tokens; training length 4K; hybrid SWA window 2K.',
    reasoning: 'Table 3 Avg.; eight accuracy-based commonsense tasks, excluding perplexity columns.',
    retrieval: 'Table 4 Avg.; six retrieval tasks, inputs truncated to 2K tokens.',
    values: scores.map(({ name, reasoning, retrieval }) => ({ name, reasoning, retrieval })),
  },
}, null, 2) + '\n');
console.log('Generated 12 localized SVG figures and their source manifest.');
