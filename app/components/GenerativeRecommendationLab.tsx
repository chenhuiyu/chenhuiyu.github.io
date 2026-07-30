"use client";

import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";

type Locale = "zh-CN" | "en";
type Localized = Record<Locale, string>;
type SemanticId = [number, number, number, number];
type Point = [number, number];

type CatalogItem = {
  id: string;
  emoji: string;
  name: Localized;
  sid: SemanticId;
};

type Beam = {
  path: number[];
  logProbability: number;
};

const catalog: CatalogItem[] = [
  {
    id: "T01",
    emoji: "🎾",
    name: { "zh-CN": "网球拍", en: "Tennis racket" },
    sid: [0, 0, 0, 0],
  },
  {
    id: "T02",
    emoji: "🟡",
    name: { "zh-CN": "网球", en: "Tennis balls" },
    sid: [0, 0, 1, 0],
  },
  {
    id: "C01",
    emoji: "🧗",
    name: { "zh-CN": "攀岩鞋", en: "Climbing shoes" },
    sid: [0, 1, 0, 0],
  },
  {
    id: "C02",
    emoji: "◻",
    name: { "zh-CN": "粉袋", en: "Chalk bag" },
    sid: [0, 1, 1, 1],
  },
  {
    id: "B01",
    emoji: "🏸",
    name: { "zh-CN": "羽毛球拍", en: "Badminton racket" },
    sid: [1, 0, 0, 1],
  },
  {
    id: "B02",
    emoji: "○",
    name: { "zh-CN": "羽毛球", en: "Shuttlecocks" },
    sid: [1, 0, 1, 0],
  },
  {
    id: "S01",
    emoji: "🥽",
    name: { "zh-CN": "泳镜", en: "Swim goggles" },
    sid: [1, 1, 0, 1],
  },
  {
    id: "S02",
    emoji: "🏊",
    name: { "zh-CN": "泳帽", en: "Swim cap" },
    sid: [1, 1, 1, 0],
  },
];

const codeVectors: Point[][] = [
  [
    [0.22, 0.12],
    [0.5, 0.1],
  ],
  [
    [0.04, 0.1],
    [0.08, 0.4],
  ],
  [
    [0.04, 0.02],
    [0.11, 0.08],
  ],
  [
    [0, 0],
    [0.02, 0.01],
  ],
];

const copy = {
  "zh-CN": {
    sectionLabel: "COMPANION LAB · 交互实验",
    semanticTitle: "先把商品变成一种可以生成的语言。",
    semanticBody:
      "选择一件商品。图上的位置来自四层 code vector 相加；共享前缀的商品会先落在同一片粗粒度区域，再被后续 token 逐步分开。",
    selectItem: "选择商品",
    mapLabel: "八件玩具商品的二维语义空间",
    mapDescription:
      "商品按第一个 Semantic ID token 分为左右两组，当前商品以橙色高亮。",
    address: "生成地址",
    sharedPrefix: "共享前缀",
    sharedPrefixEmpty: "没有其他商品共享前三位前缀",
    rqLabel: "02 · RESIDUAL QUANTIZATION",
    rqTitle: "每一层只负责解释上一层没解释完的部分。",
    rqBody:
      "橙色终点是商品向量。四段箭头依次加入 code vector；越靠后的 token，修正越细。图中是教学用二维投影，不代表真实 RQ-VAE 只有两个维度。",
    target: "目标向量",
    reconstruction: "重构结果",
    residual: "剩余残差",
    decoderLabel: "03 · AUTOREGRESSIVE DECODING",
    decoderTitle: "生成一个 ID，就是在商品目录树里一步步选路。",
    decoderBody:
      "下面的概率来自同一个固定 toy model。Temperature 改变分布的尖锐程度，beam width 决定保留多少条路，decode depth 决定走到第几位。",
    temperature: "Temperature",
    beamWidth: "Beam width",
    decodeDepth: "Decode depth",
    trie: "只允许目录 Trie 中存在的前缀",
    token: "Token",
    validItem: "目录内商品",
    invalidItem: "目录中不存在",
    partial: "尚未生成完整 ID",
    comparisonTitle: "同一个模型，两种解码结果",
    unconstrained: "无约束",
    constrained: "Trie 约束",
    comparisonBody:
      "无约束模型最喜欢的完整地址并不一定对应真实商品。Trie 在每一步屏蔽非法前缀，让概率只能流向目录中的路径。",
    pythonLabel: "04 · EDITABLE PYTHON",
    pythonTitle: "现在改代码，而不只是看动画。",
    pythonBody:
      "代码会在你的浏览器里真实执行。第一次运行需要加载约 12 MB 的本地 Python 核心；之后再次运行会复用已经启动的环境。",
    run: "运行 Python",
    running: "运行中…",
    restore: "恢复示例",
    output: "输出",
    idleOutput: "点击“运行 Python”查看 beam search 结果。",
    loadingOutput: "正在准备浏览器内的 Python 环境…",
    emptyOutput: "代码执行完成，没有打印输出。",
    errorPrefix: "执行失败：",
    toyNote:
      "边界说明：这个 lab 用 8 件商品和手写 logits 教机制；完整复现还需要真实商品 encoder、RQ-VAE 训练、序列数据与离线评测。",
  },
  en: {
    sectionLabel: "COMPANION LAB · INTERACTIVE",
    semanticTitle: "First, turn products into a language the model can generate.",
    semanticBody:
      "Choose an item. Its position is the sum of four code vectors. Items with a shared prefix first occupy the same coarse region, then later tokens separate them.",
    selectItem: "Choose an item",
    mapLabel: "A two-dimensional semantic space for eight toy products",
    mapDescription:
      "The first Semantic ID token splits products into left and right groups; the selected item is highlighted in orange.",
    address: "Generated address",
    sharedPrefix: "Shared prefix",
    sharedPrefixEmpty: "No other item shares the first three tokens",
    rqLabel: "02 · RESIDUAL QUANTIZATION",
    rqTitle: "Each level explains only what the previous level left behind.",
    rqBody:
      "The orange endpoint is the item vector. Four arrows add code vectors from coarse to fine. This is a teaching projection, not a claim that a real RQ-VAE has only two dimensions.",
    target: "Target vector",
    reconstruction: "Reconstruction",
    residual: "Residual left",
    decoderLabel: "03 · AUTOREGRESSIVE DECODING",
    decoderTitle: "Generating an ID means choosing a path through the catalog tree.",
    decoderBody:
      "All probabilities come from one fixed toy model. Temperature changes sharpness, beam width controls how many paths survive, and decode depth controls how far generation proceeds.",
    temperature: "Temperature",
    beamWidth: "Beam width",
    decodeDepth: "Decode depth",
    trie: "Allow only prefixes present in the catalog Trie",
    token: "Token",
    validItem: "Catalog item",
    invalidItem: "Not in catalog",
    partial: "The ID is not complete yet",
    comparisonTitle: "One model, two decoding outcomes",
    unconstrained: "Unconstrained",
    constrained: "Trie-constrained",
    comparisonBody:
      "The model’s favorite complete address need not name a real item. The Trie masks invalid prefixes at every step, so probability can flow only into catalog paths.",
    pythonLabel: "04 · EDITABLE PYTHON",
    pythonTitle: "Now edit the code instead of merely watching it.",
    pythonBody:
      "This code genuinely executes in your browser. The first run loads roughly 12 MB of locally hosted Python core files; later runs reuse the initialized environment.",
    run: "Run Python",
    running: "Running…",
    restore: "Restore example",
    output: "Output",
    idleOutput: "Select “Run Python” to inspect the beam-search result.",
    loadingOutput: "Preparing the in-browser Python environment…",
    emptyOutput: "The code finished without printing output.",
    errorPrefix: "Execution failed: ",
    toyNote:
      "Boundary: this lab uses eight products and hand-written logits to teach the mechanism. A full reproduction still needs a real item encoder, RQ-VAE training, sequence data, and offline evaluation.",
  },
} satisfies Record<Locale, Record<string, string>>;

const defaultPython = `from math import exp, log

CATALOG = {
    "tennis_racket": (0, 0, 0, 0),
    "tennis_balls":  (0, 0, 1, 0),
    "climbing_shoes":(0, 1, 0, 0),
    "chalk_bag":     (0, 1, 1, 1),
    "badminton":     (1, 0, 0, 1),
    "shuttlecocks":  (1, 0, 1, 0),
    "swim_goggles":  (1, 1, 0, 1),
    "swim_cap":      (1, 1, 1, 0),
}

# Scores emitted by a tiny autoregressive model.
LOGITS = {
    ():       (0.15, 1.35),
    (1,):     (0.25, 1.15),
    (1, 1):   (0.30, 0.95),
    (1, 1, 1):(0.10, 1.50),  # token 1 creates an invalid ID
    (1, 1, 0):(0.20, 1.20),
}

def valid_prefix(prefix):
    return any(sid[:len(prefix)] == prefix for sid in CATALOG.values())

def probabilities(prefix, temperature, allowed):
    scores = LOGITS.get(prefix, (0.80, 0.55))
    scaled = {token: scores[token] / temperature for token in allowed}
    peak = max(scaled.values())
    weights = {token: exp(score - peak) for token, score in scaled.items()}
    total = sum(weights.values())
    return {token: weight / total for token, weight in weights.items()}

def beam_search(temperature=0.8, beam_width=3, constrained=True):
    beams = [((), 0.0)]
    for _ in range(4):
        candidates = []
        for prefix, log_probability in beams:
            allowed = [
                token for token in (0, 1)
                if not constrained or valid_prefix(prefix + (token,))
            ]
            for token, probability in probabilities(
                prefix, temperature, allowed
            ).items():
                candidates.append(
                    (prefix + (token,), log_probability + log(probability))
                )
        beams = sorted(candidates, key=lambda row: row[1], reverse=True)
        beams = beams[:beam_width]
    return beams

reverse_catalog = {sid: name for name, sid in CATALOG.items()}
for constrained in (False, True):
    label = "TRIE CONSTRAINED" if constrained else "UNCONSTRAINED"
    print(f"\\n{label}")
    for sid, log_probability in beam_search(constrained=constrained):
        item = reverse_catalog.get(sid, "INVALID ID")
        print(f"  {sid}  p={exp(log_probability):.3f}  -> {item}")
`;

function addPoints(left: Point, right: Point): Point {
  return [left[0] + right[0], left[1] + right[1]];
}

function subtractPoints(left: Point, right: Point): Point {
  return [left[0] - right[0], left[1] - right[1]];
}

function itemSteps(item: CatalogItem) {
  const steps: Point[] = [[0, 0]];
  item.sid.forEach((token, level) => {
    steps.push(addPoints(steps[steps.length - 1], codeVectors[level][token]));
  });
  return steps;
}

function itemPoint(item: CatalogItem) {
  return itemSteps(item).at(-1) as Point;
}

function pathKey(path: number[]) {
  return path.join("");
}

function isValidPrefix(path: number[]) {
  return catalog.some((item) =>
    path.every((token, index) => item.sid[index] === token),
  );
}

function logitsFor(path: number[]): [number, number] {
  const known: Record<string, [number, number]> = {
    "": [0.15, 1.35],
    "0": [0.85, 0.45],
    "00": [0.95, 0.5],
    "01": [0.55, 0.9],
    "000": [1.0, 0.35],
    "001": [0.9, 0.45],
    "010": [1.0, 0.4],
    "011": [0.35, 0.95],
    "1": [0.25, 1.15],
    "10": [0.65, 0.85],
    "11": [0.3, 0.95],
    "100": [0.3, 0.95],
    "101": [0.9, 0.4],
    "110": [0.2, 1.2],
    "111": [0.1, 1.5],
  };
  return known[pathKey(path)] ?? [0.8, 0.55];
}

function tokenProbabilities(
  path: number[],
  temperature: number,
  constrained: boolean,
) {
  const allowed = [0, 1].filter(
    (token) => !constrained || isValidPrefix([...path, token]),
  );
  const logits = logitsFor(path);
  const scaled = allowed.map((token) => logits[token] / temperature);
  const peak = Math.max(...scaled);
  const weights = scaled.map((score) => Math.exp(score - peak));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  return allowed.map((token, index) => ({
    token,
    probability: weights[index] / total,
  }));
}

function beamSearch(
  depth: number,
  beamWidth: number,
  temperature: number,
  constrained: boolean,
) {
  let beams: Beam[] = [{ path: [], logProbability: 0 }];
  const steps: Beam[][] = [];

  for (let level = 0; level < depth; level += 1) {
    const candidates = beams.flatMap((beam) =>
      tokenProbabilities(beam.path, temperature, constrained).map(
        ({ token, probability }) => ({
          path: [...beam.path, token],
          logProbability: beam.logProbability + Math.log(probability),
        }),
      ),
    );
    beams = candidates
      .sort((left, right) => right.logProbability - left.logProbability)
      .slice(0, beamWidth);
    steps.push(beams);
  }

  return { beams, steps };
}

function itemForPath(path: number[]) {
  if (path.length !== 4) return undefined;
  return catalog.find((item) =>
    item.sid.every((token, index) => token === path[index]),
  );
}

function formatPath(path: number[]) {
  return `〈${path.join(" · ")}${path.length < 4 ? " · …" : ""}〉`;
}

export function GenerativeRecommendationLab({ locale }: { locale: Locale }) {
  const text = copy[locale];
  const [selectedId, setSelectedId] = useState("S01");
  const [temperature, setTemperature] = useState(0.8);
  const [beamWidth, setBeamWidth] = useState(3);
  const [decodeDepth, setDecodeDepth] = useState(4);
  const [trieConstrained, setTrieConstrained] = useState(true);
  const [pythonCode, setPythonCode] = useState(defaultPython);
  const [pythonOutput, setPythonOutput] = useState(text.idleOutput);
  const [pythonStatus, setPythonStatus] = useState<
    "idle" | "loading" | "running" | "ready" | "error"
  >("idle");
  const workerRef = useRef<Worker | null>(null);
  const activeRunRef = useRef(0);
  const rawArrowId = useId();
  const arrowId = `rq-arrow-${rawArrowId.replaceAll(":", "")}`;

  const selectedItem =
    catalog.find((item) => item.id === selectedId) ?? catalog[0];
  const selectedSteps = itemSteps(selectedItem);
  const selectedPoint = selectedSteps.at(-1) as Point;
  const reconstructed = selectedSteps.at(-1) as Point;
  const residual = subtractPoints(selectedPoint, reconstructed);
  const sharedPrefixItems = catalog.filter(
    (item) =>
      item.id !== selectedItem.id &&
      item.sid.slice(0, 3).every((token, index) => token === selectedItem.sid[index]),
  );

  const decoding = useMemo(
    () =>
      beamSearch(
        decodeDepth,
        beamWidth,
        temperature,
        trieConstrained,
      ),
    [beamWidth, decodeDepth, temperature, trieConstrained],
  );
  const comparison = useMemo(
    () => ({
      unconstrained: beamSearch(4, beamWidth, temperature, false).beams[0],
      constrained: beamSearch(4, beamWidth, temperature, true).beams[0],
    }),
    [beamWidth, temperature],
  );

  useEffect(
    () => () => {
      workerRef.current?.terminate();
    },
    [],
  );

  function ensureWorker() {
    if (workerRef.current) return workerRef.current;

    const worker = new Worker("/workers/generative-lab-worker.mjs", {
      type: "module",
    });
    worker.addEventListener("message", (event) => {
      const {
        runId,
        type,
        output,
      }: { runId: number; type: string; output?: string } = event.data ?? {};
      if (runId !== activeRunRef.current) return;

      if (type === "loading") {
        setPythonStatus("loading");
        setPythonOutput(text.loadingOutput);
      } else if (type === "result") {
        setPythonStatus("ready");
        setPythonOutput(output?.trim() || text.emptyOutput);
      } else if (type === "error") {
        setPythonStatus("error");
        setPythonOutput(`${text.errorPrefix}${output ?? "Unknown error"}`);
      }
    });
    workerRef.current = worker;
    return worker;
  }

  function runPython() {
    const runId = activeRunRef.current + 1;
    activeRunRef.current = runId;
    setPythonStatus("running");
    setPythonOutput(text.loadingOutput);
    ensureWorker().postMessage({ code: pythonCode, runId });
  }

  const mapX = (point: Point) => 55 + point[0] * 850;
  const mapY = (point: Point) => 310 - point[1] * 420;
  const rqX = (point: Point) => 58 + point[0] * 830;
  const rqY = (point: Point) => 300 - point[1] * 405;

  return (
    <div className="generative-lab">
      <section className="lab-section" id="semantic-id">
        <div className="lab-heading">
          <p>{text.sectionLabel}</p>
          <h2>{text.semanticTitle}</h2>
          <span>{text.semanticBody}</span>
        </div>

        <div className="lab-item-picker" aria-label={text.selectItem}>
          {catalog.map((item) => (
            <button
              aria-pressed={item.id === selectedItem.id}
              key={item.id}
              onClick={() => setSelectedId(item.id)}
              type="button"
            >
              <span aria-hidden="true">{item.emoji}</span>
              {item.name[locale]}
            </button>
          ))}
        </div>

        <div className="semantic-lab-grid">
          <div className="semantic-map">
            <svg
              aria-labelledby="semantic-map-title semantic-map-description"
              role="img"
              viewBox="0 0 720 330"
            >
              <title id="semantic-map-title">{text.mapLabel}</title>
              <desc id="semantic-map-description">{text.mapDescription}</desc>
              <line className="map-axis" x1="42" x2="690" y1="294" y2="294" />
              <line className="map-axis" x1="42" x2="42" y1="30" y2="294" />
              <line className="map-divider" x1="395" x2="395" y1="38" y2="294" />
              <text className="map-cluster-label" x="68" y="52">
                c₁ = 0
              </text>
              <text className="map-cluster-label" x="424" y="52">
                c₁ = 1
              </text>
              {catalog.map((item) => {
                const point = itemPoint(item);
                const active = item.id === selectedItem.id;
                return (
                  <g
                    className={active ? "is-active" : ""}
                    key={item.id}
                    transform={`translate(${mapX(point)} ${mapY(point)})`}
                  >
                    <circle r={active ? 18 : 13} />
                    <text textAnchor="middle" y="4">
                      {item.id}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          <aside className="semantic-address-card">
            <p>
              {selectedItem.emoji} {selectedItem.name[locale]}{" "}
              <small>{selectedItem.id}</small>
            </p>
            <span>{text.address}</span>
            <div className="semantic-token-row" aria-label={formatPath(selectedItem.sid)}>
              {selectedItem.sid.map((token, index) => (
                <strong key={`${selectedItem.id}-${index}`}>{token}</strong>
              ))}
            </div>
            <dl>
              <div>
                <dt>{text.sharedPrefix}</dt>
                <dd>
                  {sharedPrefixItems.length
                    ? sharedPrefixItems
                        .map((item) => item.name[locale])
                        .join(" · ")
                    : text.sharedPrefixEmpty}
                </dd>
              </div>
            </dl>
          </aside>
        </div>
      </section>

      <section className="lab-section lab-rq-section" id="residual-quantization">
        <div className="lab-heading">
          <p>{text.rqLabel}</p>
          <h2>{text.rqTitle}</h2>
          <span>{text.rqBody}</span>
        </div>

        <div className="rq-visual">
          <svg
            aria-label={`${text.rqTitle} ${selectedItem.name[locale]}`}
            role="img"
            viewBox="0 0 720 330"
          >
            <defs>
              <marker
                id={arrowId}
                markerHeight="7"
                markerWidth="7"
                orient="auto-start-reverse"
                refX="6"
                refY="3.5"
                viewBox="0 0 7 7"
              >
                <path d="M0 0L7 3.5L0 7Z" />
              </marker>
            </defs>
            <line className="rq-axis" x1="48" x2="684" y1="300" y2="300" />
            <line className="rq-axis" x1="48" x2="48" y1="32" y2="300" />
            {selectedSteps.slice(1).map((point, index) => {
              const previous = selectedSteps[index];
              return (
                <g className={`rq-level rq-level-${index + 1}`} key={index}>
                  <line
                    markerEnd={`url(#${arrowId})`}
                    x1={rqX(previous)}
                    x2={rqX(point)}
                    y1={rqY(previous)}
                    y2={rqY(point)}
                  />
                  <circle cx={rqX(point)} cy={rqY(point)} r="7" />
                  <text x={rqX(point) + 10} y={rqY(point) - 10}>
                    c{index + 1}={selectedItem.sid[index]}
                  </text>
                </g>
              );
            })}
            <circle
              className="rq-target"
              cx={rqX(selectedPoint)}
              cy={rqY(selectedPoint)}
              r="15"
            />
          </svg>
          <div className="rq-legend">
            {selectedItem.sid.map((token, index) => (
              <span className={`rq-legend-${index + 1}`} key={index}>
                <i aria-hidden="true" />
                L{index + 1}: code {token}
              </span>
            ))}
          </div>
        </div>

        <div className="rq-metrics" aria-label={text.reconstruction}>
          <p>
            <span>{text.target}</span>
            <strong>
              [{selectedPoint.map((value) => value.toFixed(2)).join(", ")}]
            </strong>
          </p>
          <p>
            <span>{text.reconstruction}</span>
            <strong>
              [{reconstructed.map((value) => value.toFixed(2)).join(", ")}]
            </strong>
          </p>
          <p>
            <span>{text.residual}</span>
            <strong>
              [{residual.map((value) => value.toFixed(2)).join(", ")}]
            </strong>
          </p>
        </div>
      </section>

      <section className="lab-section" id="constrained-decoding">
        <div className="lab-heading">
          <p>{text.decoderLabel}</p>
          <h2>{text.decoderTitle}</h2>
          <span>{text.decoderBody}</span>
        </div>

        <div className="decoder-controls">
          <label>
            <span>
              {text.temperature} <strong>{temperature.toFixed(1)}</strong>
            </span>
            <input
              max="1.8"
              min="0.4"
              onChange={(event) => setTemperature(Number(event.target.value))}
              step="0.1"
              type="range"
              value={temperature}
            />
          </label>
          <label>
            <span>
              {text.beamWidth} <strong>{beamWidth}</strong>
            </span>
            <input
              max="5"
              min="1"
              onChange={(event) => setBeamWidth(Number(event.target.value))}
              step="1"
              type="range"
              value={beamWidth}
            />
          </label>
          <label>
            <span>
              {text.decodeDepth} <strong>{decodeDepth}/4</strong>
            </span>
            <input
              max="4"
              min="1"
              onChange={(event) => setDecodeDepth(Number(event.target.value))}
              step="1"
              type="range"
              value={decodeDepth}
            />
          </label>
          <label className="trie-switch">
            <input
              checked={trieConstrained}
              onChange={(event) => setTrieConstrained(event.target.checked)}
              type="checkbox"
            />
            <span aria-hidden="true" />
            <strong>{text.trie}</strong>
          </label>
        </div>

        <div className="decode-flow" aria-live="polite">
          {decoding.steps.map((step, stepIndex) => {
            const peak = Math.max(...step.map((beam) => beam.logProbability));
            return (
              <div className="decode-column" key={stepIndex}>
                <p>
                  {text.token} {stepIndex + 1}
                </p>
                {step.map((beam) => {
                  const item = itemForPath(beam.path);
                  const invalid =
                    beam.path.length === 4 && !isValidPrefix(beam.path);
                  const relative = Math.exp(beam.logProbability - peak);
                  return (
                    <div
                      className={`decode-beam${invalid ? " is-invalid" : ""}`}
                      key={pathKey(beam.path)}
                    >
                      <span>{formatPath(beam.path)}</span>
                      <i style={{ width: `${Math.max(8, relative * 100)}%` }} />
                      <small>
                        {item
                          ? `${item.emoji} ${item.name[locale]}`
                          : invalid
                            ? `× ${text.invalidItem}`
                            : text.partial}
                      </small>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>

        <div className="decode-comparison">
          <div>
            <p>{text.comparisonTitle}</p>
            <span>{text.comparisonBody}</span>
          </div>
          {(
            [
              ["unconstrained", text.unconstrained],
              ["constrained", text.constrained],
            ] as const
          ).map(([key, label]) => {
            const beam = comparison[key];
            const item = itemForPath(beam.path);
            return (
              <article className={item ? "is-valid" : "is-invalid"} key={key}>
                <span>{label}</span>
                <strong>{formatPath(beam.path)}</strong>
                <small>
                  {item
                    ? `✓ ${text.validItem}: ${item.name[locale]}`
                    : `× ${text.invalidItem}`}
                </small>
              </article>
            );
          })}
        </div>
      </section>

      <section className="lab-section lab-python-section" id="browser-python">
        <div className="lab-heading">
          <p>{text.pythonLabel}</p>
          <h2>{text.pythonTitle}</h2>
          <span>{text.pythonBody}</span>
        </div>

        <div className="python-toolbar">
          <button
            className="python-run"
            disabled={pythonStatus === "loading" || pythonStatus === "running"}
            onClick={runPython}
            type="button"
          >
            <span aria-hidden="true">▶</span>
            {pythonStatus === "loading" || pythonStatus === "running"
              ? text.running
              : text.run}
          </button>
          <button
            className="python-restore"
            disabled={pythonCode === defaultPython}
            onClick={() => setPythonCode(defaultPython)}
            type="button"
          >
            {text.restore}
          </button>
        </div>

        <div className="python-workbench">
          <label>
            <span className="sr-only">Python code</span>
            <textarea
              aria-label="Python code"
              onChange={(event) => setPythonCode(event.target.value)}
              spellCheck={false}
              value={pythonCode}
            />
          </label>
          <div className="python-output">
            <p>
              {text.output}
              <span className={`python-status is-${pythonStatus}`}>
                {pythonStatus}
              </span>
            </p>
            <pre aria-live="polite">{pythonOutput}</pre>
          </div>
        </div>

        <p className="lab-boundary-note">{text.toyNote}</p>
      </section>
    </div>
  );
}
