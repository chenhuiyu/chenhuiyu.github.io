import { SiteFooter, SiteHeader } from "@/app/components/SiteHeader";
import seriesData from "@/content/generative-recommendation-series.json";
import { posts } from "@/lib/posts";
import { absoluteUrl, serializeJsonLd, SITE_URL } from "@/lib/site";

type Locale = "zh-CN" | "en";

type Localized = Record<Locale, string>;

type Arc = {
  id: string;
  range: string;
  label: Localized;
  description: Localized;
};

type Episode = {
  order: number;
  paper: string;
  arc: string;
  title: Localized;
  role: Localized;
};

const series = seriesData as {
  id: string;
  title: Localized;
  subtitle: Localized;
  question: Localized;
  arcs: Arc[];
  episodes: Episode[];
};

const pageCopy = {
  "zh-CN": {
    eyebrow: "专题系列 · 00–20",
    language: "Read in English",
    languageHref: "/series/generative-recommendation/en",
    prologueLabel: "序章 · 已发布",
    prologueTitle: "生成式推荐到底在生成什么？",
    prologueCopy:
      "在进入论文之前，先分清模型是在生成语言、商品 ID、完整列表，还是一个看起来像推理的过程。",
    prologueAction: "从序章开始",
    mapKicker: "一张图看懂主线",
    mapTitle: "同一条问题，分成两条支线，最后重新汇合。",
    score: "打分与排序",
    sequence: "行为序列",
    question: "模型究竟生成什么？",
    languageBranch: "生成语言答案",
    idBranch: "生成商品标识",
    convergence: "统一、规模化与推荐推理",
    roadmapKicker: "20 篇论文路线图",
    roadmapTitle: "每一篇都由上一篇未解决的问题推出来。",
    upcoming: "写作中",
    read: "阅读本篇",
    labKicker: "Companion lab · 动手实验",
    labTitle: "读完论文，再亲手生成一次 Semantic ID。",
    labText:
      "从 BPR 商品向量、残差量化到 beam search 与 Trie 约束；既能在页面里即时改 Python，也能进入完整 Jupyter / Colab Notebook。",
    labAction: "打开交互教程与 Notebook",
    labSteps: ["BPR", "Semantic ID", "Decode", "Evaluate"],
    methodKicker: "统一阅读方法",
    methodTitle: "零基础能进入，工程师能复现，读论文的人能质疑。",
    layers: [
      {
        index: "01",
        title: "零基础层",
        text: "生活例子、上一代模型的困难，以及这一篇的一句话答案。",
      },
      {
        index: "02",
        title: "工程层",
        text: "输入输出、tensor shape、训练数据、推理过程和复杂度。",
      },
      {
        index: "03",
        title: "论文层",
        text: "一个核心公式、一张关键实验、一个重要 ablation，以及论文没有证明什么。",
      },
    ],
    visualKicker: "贯穿全系列的玩具世界",
    visualTitle: "同一个用户，同一组商品，看模型如何一代代改变。",
    visualText:
      "固定行为序列是“网球 → 攀岩 → 羽毛球 → 游泳”。每篇都画前后对比、数据流、公式拆解和证据图，让变化发生在同一份数据上。",
    sideNoteTitle: "同名，但不同宗",
    sideNote:
      "VAE、GAN、Diffusion 也曾被称为生成式推荐，但不是本系列 autoregressive ID generation 的直接主线，因此放在序章的知识框中，而不占用 20 篇主线名额。",
  },
  en: {
    eyebrow: "Editorial series · 00–20",
    language: "阅读中文版",
    languageHref: "/series/generative-recommendation",
    prologueLabel: "Prologue · published",
    prologueTitle: "What Does Generative Recommendation Actually Generate?",
    prologueCopy:
      "Before the papers, distinguish language, item IDs, complete lists, and a process that merely looks like reasoning.",
    prologueAction: "Start with the prologue",
    mapKicker: "The story in one map",
    mapTitle: "One question splits into two branches, then converges again.",
    score: "Scoring & ranking",
    sequence: "Behavior sequences",
    question: "What does the model generate?",
    languageBranch: "Generate language answers",
    idBranch: "Generate item identifiers",
    convergence: "Unification, scale & recommendation reasoning",
    roadmapKicker: "The 20-paper roadmap",
    roadmapTitle: "Every paper is motivated by what the previous one could not solve.",
    upcoming: "In progress",
    read: "Read this episode",
    labKicker: "Companion lab · hands-on",
    labTitle: "After the papers, generate a Semantic ID yourself.",
    labText:
      "Move from BPR item vectors and residual quantization to beam search and Trie constraints. Edit Python on the page or run the complete Jupyter / Colab notebook.",
    labAction: "Open the tutorial and notebook",
    labSteps: ["BPR", "Semantic ID", "Decode", "Evaluate"],
    methodKicker: "One reading protocol",
    methodTitle:
      "Approachable to beginners, reproducible for engineers, and debatable for paper readers.",
    layers: [
      {
        index: "01",
        title: "Beginner layer",
        text: "A familiar example, the previous model’s limitation, and the new idea in one sentence.",
      },
      {
        index: "02",
        title: "Engineering layer",
        text: "Inputs, outputs, tensor shapes, training data, inference, and complexity.",
      },
      {
        index: "03",
        title: "Paper layer",
        text: "One central equation, one decisive result, one ablation, and what remains unproven.",
      },
    ],
    visualKicker: "One toy world across the series",
    visualTitle: "The same user and items reveal how each generation changes.",
    visualText:
      "The fixed history is “tennis → climbing → badminton → swimming.” Every article uses a before-and-after diagram, a data flow, an equation map, and an evidence figure.",
    sideNoteTitle: "The same name, a different lineage",
    sideNote:
      "VAEs, GANs, and diffusion models have also been called generative recommendation. They are not the direct lineage of autoregressive ID generation, so they appear in the prologue rather than taking one of the 20 main episodes.",
  },
} satisfies Record<Locale, object>;

export function SeriesPage({ locale }: { locale: Locale }) {
  const copy = pageCopy[locale];
  const publishedByOrder = new Map(
    posts
      .filter(
        (post) =>
          post.series === series.id &&
          post.language === locale &&
          typeof post.seriesOrder === "number",
      )
      .map((post) => [post.seriesOrder as number, post]),
  );
  const prologue = publishedByOrder.get(0);
  const labPost = posts.find(
    (post) =>
      post.pairKey === "generative-recommendation-hands-on" &&
      post.language === locale,
  );
  const seriesPath =
    locale === "zh-CN"
      ? "/series/generative-recommendation"
      : "/series/generative-recommendation/en";
  const seriesUrl = absoluteUrl(seriesPath);
  const seriesJsonLd = {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    "@id": `${seriesUrl}#series`,
    url: seriesUrl,
    name: series.title[locale],
    description: series.subtitle[locale],
    inLanguage: locale,
    author: {
      "@id": `${SITE_URL}/#person`,
    },
    isPartOf: {
      "@id": `${SITE_URL}/#website`,
    },
    mainEntity: {
      "@type": "ItemList",
      numberOfItems: series.episodes.length,
      itemListOrder: "https://schema.org/ItemListOrderAscending",
      itemListElement: series.episodes.map((episode) => {
        const post = publishedByOrder.get(episode.order);
        return {
          "@type": "ListItem",
          position: episode.order,
          item: {
            "@type": "Article",
            name: episode.title[locale],
            headline: episode.title[locale],
            ...(post
              ? {
                  url: absoluteUrl(`/blog/${post.slug}`),
                  datePublished: post.date,
                  dateModified: post.updated || post.date,
                }
              : {}),
          },
        };
      }),
    },
  };

  return (
    <main className="site-shell series-shell" id="top">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: serializeJsonLd(seriesJsonLd) }}
      />
      <SiteHeader />

      <section className="series-hero">
        <div>
          <p className="eyebrow">{copy.eyebrow}</p>
          <h1>{series.title[locale]}</h1>
          <p className="series-subtitle">{series.subtitle[locale]}</p>
        </div>
        <div className="series-question-card">
          <span aria-hidden="true">?</span>
          <p>{series.question[locale]}</p>
          <a className="language-switch" href={copy.languageHref}>
            {copy.language} →
          </a>
        </div>
      </section>

      <a
        className="series-prologue"
        href={
          prologue
            ? `/blog/${prologue.slug}`
            : locale === "zh-CN"
              ? "/blog"
              : "/blog"
        }
      >
        <div>
          <p>{copy.prologueLabel}</p>
          <h2>{copy.prologueTitle}</h2>
        </div>
        <p>{copy.prologueCopy}</p>
        <span>{copy.prologueAction} →</span>
      </a>

      <section className="series-map-section" aria-labelledby="series-map-title">
        <div className="series-section-heading">
          <p className="section-kicker">{copy.mapKicker}</p>
          <h2 id="series-map-title">{copy.mapTitle}</h2>
        </div>

        <div className="series-map" aria-label={copy.mapTitle}>
          <div className="series-map-node">
            <small>BPR</small>
            <strong>{copy.score}</strong>
          </div>
          <span className="series-map-arrow" aria-hidden="true">
            →
          </span>
          <div className="series-map-node">
            <small>GRU4Rec → SASRec → BERT4Rec</small>
            <strong>{copy.sequence}</strong>
          </div>
          <span className="series-map-arrow" aria-hidden="true">
            →
          </span>
          <div className="series-map-question">
            <small>05–15</small>
            <strong>{copy.question}</strong>
          </div>
          <div className="series-map-branches">
            <div>
              <small>P5 → LLaRA</small>
              <strong>{copy.languageBranch}</strong>
            </div>
            <div>
              <small>DSI → ETEGRec</small>
              <strong>{copy.idBranch}</strong>
            </div>
          </div>
          <div className="series-map-merge" aria-hidden="true">
            ↘︎ &nbsp; ↙︎
          </div>
          <div className="series-map-final">
            <small>HSTU → OneRec → OneReason</small>
            <strong>{copy.convergence}</strong>
          </div>
        </div>
      </section>

      <section className="series-roadmap" aria-labelledby="series-roadmap-title">
        <div className="series-section-heading">
          <p className="section-kicker">{copy.roadmapKicker}</p>
          <h2 id="series-roadmap-title">{copy.roadmapTitle}</h2>
        </div>

        <div className="series-arcs">
          {series.arcs.map((arc) => (
            <section className={`series-arc series-arc-${arc.id}`} key={arc.id}>
              <header>
                <p>{arc.range}</p>
                <div>
                  <h3>{arc.label[locale]}</h3>
                  <p>{arc.description[locale]}</p>
                </div>
              </header>
              <div className="series-episode-list">
                {series.episodes
                  .filter((episode) => episode.arc === arc.id)
                  .map((episode) => {
                    const post = publishedByOrder.get(episode.order);
                    const card = (
                      <>
                        <div className="series-episode-number">
                          {String(episode.order).padStart(2, "0")}
                        </div>
                        <div className="series-episode-copy">
                          <p>{episode.paper}</p>
                          <h4>{episode.title[locale]}</h4>
                          <span>{episode.role[locale]}</span>
                        </div>
                        <div className="series-episode-status">
                          {post ? `${copy.read} →` : copy.upcoming}
                        </div>
                      </>
                    );

                    return post ? (
                      <a
                        className="series-episode is-published"
                        href={`/blog/${post.slug}`}
                        key={episode.order}
                      >
                        {card}
                      </a>
                    ) : (
                      <article className="series-episode" key={episode.order}>
                        {card}
                      </article>
                    );
                  })}
              </div>
            </section>
          ))}
        </div>
      </section>

      {labPost ? (
        <a className="series-lab-callout" href={`/blog/${labPost.slug}`}>
          <div className="series-lab-copy">
            <p className="section-kicker">{copy.labKicker}</p>
            <h2>{copy.labTitle}</h2>
            <span>{copy.labText}</span>
          </div>
          <div className="series-lab-path" aria-hidden="true">
            {copy.labSteps.map((step, index) => (
              <div key={step}>
                <span>{step}</span>
                {index < copy.labSteps.length - 1 ? <i>→</i> : null}
              </div>
            ))}
          </div>
          <strong>
            {copy.labAction} <span aria-hidden="true">↗︎</span>
          </strong>
        </a>
      ) : null}

      <section className="series-method" aria-labelledby="series-method-title">
        <div className="series-section-heading">
          <p className="section-kicker">{copy.methodKicker}</p>
          <h2 id="series-method-title">{copy.methodTitle}</h2>
        </div>
        <div className="series-layer-grid">
          {copy.layers.map((layer) => (
            <article key={layer.index}>
              <span>{layer.index}</span>
              <h3>{layer.title}</h3>
              <p>{layer.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="series-visual-system">
        <div>
          <p className="section-kicker">{copy.visualKicker}</p>
          <h2>{copy.visualTitle}</h2>
          <p>{copy.visualText}</p>
        </div>
        <div className="series-toy-sequence" aria-label={copy.visualText}>
          {["🎾", "🧗", "🏸", "🏊"].map((item, index) => (
            <div key={item}>
              <span>{item}</span>
              {index < 3 ? <small aria-hidden="true">→</small> : null}
            </div>
          ))}
        </div>
        <aside>
          <p>{copy.sideNoteTitle}</p>
          <span>{copy.sideNote}</span>
        </aside>
      </section>

      <SiteFooter />
    </main>
  );
}
