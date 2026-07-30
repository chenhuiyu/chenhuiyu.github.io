import { SiteFooter, SiteHeader } from "@/app/components/SiteHeader";
import seriesData from "@/content/generative-recommendation-2026-series.json";
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

const copy = {
  "zh-CN": {
    eyebrow: "生成式推荐 · 第二季 · 2026 前沿",
    switchLabel: "Read in English",
    switchHref: "/series/generative-recommendation-2026/en",
    mapKicker: "这一季追问什么",
    mapTitle: "模型已经会生成，真正困难的部分才刚刚开始。",
    representation: "表示",
    reasoning: "推理",
    generation: "生成",
    systems: "系统",
    audit: "审计",
    roadmapKicker: "七篇论文路线图",
    roadmapTitle: "一篇论文解决一个问题，再把更棘手的问题递给下一篇。",
    read: "阅读本篇",
    upcoming: "写作中",
    seasonOne: "返回第一季",
    seasonOneHref: "/series/generative-recommendation",
    closing:
      "第二季不再讨论“生成式推荐是什么”，而是检查它怎样学习商品语言、怎样控制生成路径、怎样满足工业约束，以及怎样证明自己没有把记忆误写成泛化。",
  },
  en: {
    eyebrow: "Generative Recommendation · Season 2 · 2026 Frontier",
    switchLabel: "阅读中文版",
    switchHref: "/series/generative-recommendation-2026",
    mapKicker: "The question of this season",
    mapTitle: "The model can already generate. The difficult part starts now.",
    representation: "Representation",
    reasoning: "Reasoning",
    generation: "Generation",
    systems: "Systems",
    audit: "Audit",
    roadmapKicker: "A seven-paper roadmap",
    roadmapTitle: "Each paper solves one problem and hands a harder one to the next.",
    read: "Read this paper",
    upcoming: "In progress",
    seasonOne: "Back to Season 1",
    seasonOneHref: "/series/generative-recommendation/en",
    closing:
      "Season 2 no longer asks what generative recommendation is. It examines how item languages learn, how decoding paths are controlled, how industrial constraints are met, and how we avoid mistaking memorization for generalization.",
  },
} satisfies Record<Locale, object>;

export function SeriesPage({ locale }: { locale: Locale }) {
  const pageCopy = copy[locale];
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

  const seriesPath =
    locale === "zh-CN"
      ? "/series/generative-recommendation-2026"
      : "/series/generative-recommendation-2026/en";
  const seriesUrl = absoluteUrl(seriesPath);
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    "@id": `${seriesUrl}#series`,
    url: seriesUrl,
    name: series.title[locale],
    description: series.subtitle[locale],
    inLanguage: locale,
    author: { "@id": `${SITE_URL}/#person` },
    isPartOf: { "@id": `${SITE_URL}/#website` },
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
        dangerouslySetInnerHTML={{ __html: serializeJsonLd(jsonLd) }}
      />
      <SiteHeader />

      <section className="series-hero">
        <div>
          <p className="eyebrow">{pageCopy.eyebrow}</p>
          <h1>{series.title[locale]}</h1>
          <p className="series-subtitle">{series.subtitle[locale]}</p>
        </div>
        <div className="series-question-card">
          <span aria-hidden="true">?</span>
          <p>{series.question[locale]}</p>
          <a className="language-switch" href={pageCopy.switchHref}>
            {pageCopy.switchLabel} →
          </a>
        </div>
      </section>

      <section className="series-map-section" aria-labelledby="frontier-map-title">
        <div className="series-section-heading">
          <p className="section-kicker">{pageCopy.mapKicker}</p>
          <h2 id="frontier-map-title">{pageCopy.mapTitle}</h2>
        </div>
        <div className="series-map" aria-label={pageCopy.mapTitle}>
          <div className="series-map-node">
            <small>DIGER</small>
            <strong>{pageCopy.representation}</strong>
          </div>
          <span className="series-map-arrow" aria-hidden="true">→</span>
          <div className="series-map-node">
            <small>CARE</small>
            <strong>{pageCopy.reasoning}</strong>
          </div>
          <span className="series-map-arrow" aria-hidden="true">→</span>
          <div className="series-map-node">
            <small>ContRec · DualFashion</small>
            <strong>{pageCopy.generation}</strong>
          </div>
          <span className="series-map-arrow" aria-hidden="true">→</span>
          <div className="series-map-node">
            <small>GenRec · GR4AD</small>
            <strong>{pageCopy.systems}</strong>
          </div>
          <span className="series-map-arrow" aria-hidden="true">→</span>
          <div className="series-map-final">
            <small>Generalization</small>
            <strong>{pageCopy.audit}</strong>
          </div>
        </div>
      </section>

      <section className="series-roadmap" aria-labelledby="frontier-roadmap-title">
        <div className="series-section-heading">
          <p className="section-kicker">{pageCopy.roadmapKicker}</p>
          <h2 id="frontier-roadmap-title">{pageCopy.roadmapTitle}</h2>
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
                          {post ? `${pageCopy.read} →` : pageCopy.upcoming}
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

      <section className="series-method">
        <div className="series-section-heading">
          <p className="section-kicker">Season boundary</p>
          <h2>{pageCopy.closing}</h2>
        </div>
        <a className="language-switch" href={pageCopy.seasonOneHref}>
          {pageCopy.seasonOne} →
        </a>
      </section>

      <SiteFooter />
    </main>
  );
}
