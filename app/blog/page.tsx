import type { Metadata } from "next";
import { SiteFooter, SiteHeader } from "@/app/components/SiteHeader";
import { posts } from "@/lib/posts";
import { RSS_ALTERNATE, SITE_SOCIAL_IMAGE } from "@/lib/site";
import { BlogExplorer } from "./BlogExplorer";

const storyCount = new Set(posts.map((post) => post.pairKey)).size;
const blogDescription = `${storyCount} stories in complete English and Chinese editions, covering LLMs, NLP, engineering, code, life, and travel.`;

export const metadata: Metadata = {
  title: "Blog — Huiyu Chen",
  description: blogDescription,
  alternates: {
    canonical: "/blog",
    types: RSS_ALTERNATE,
  },
  openGraph: {
    type: "website",
    url: "/blog",
    title: "Writing Archive — Huiyu Chen",
    description: blogDescription,
    images: [SITE_SOCIAL_IMAGE],
  },
  twitter: {
    card: "summary",
    title: "Writing Archive — Huiyu Chen",
    description: blogDescription,
    images: [SITE_SOCIAL_IMAGE],
  },
};

export default function BlogPage() {
  const pairCount = storyCount;
  const categoryCount = new Set(posts.map((post) => post.category)).size;
  const cards = posts.map(
    ({ slug, title, date, category, language, excerpt, tags }) => ({
      slug,
      title,
      date,
      category,
      language,
      excerpt,
      tags,
    }),
  );

  return (
    <main className="site-shell blog-shell" id="top">
      <SiteHeader />

      <section className="blog-hero">
        <p className="eyebrow">
          <span>Writing archive</span>
          <span aria-hidden="true">·</span>
          黑头呆鱼进化之旅
        </p>
        <h1>
          Notes on models,
          <br />
          systems, and <span className="word-mark">being human</span>.
        </h1>
        <p>
          Every story has a complete English and Chinese edition—not just a
          bilingual label. New published notes join this archive and the
          homepage automatically.
        </p>
        <div className="blog-stats" aria-label="Blog archive summary">
          <span>
            <strong>{posts.length}</strong> complete editions
          </span>
          <span>
            <strong>{pairCount}</strong> EN ↔ 中文 pairs
          </span>
          <span>
            <strong>{categoryCount}</strong> shelves
          </span>
        </div>
      </section>

      <section className="blog-series-feature">
        <div className="blog-series-index">
          <span>00</span>
          <small>→</small>
          <span>20</span>
        </div>
        <div>
          <p className="section-kicker">New editorial series · 新专题</p>
          <h2>推荐系统如何学会“说出答案”</h2>
          <p>
            从 BPR 的打分器，到 OneReason 的推荐推理。不是 20
            篇互不相干的摘要，而是一条从排序、序列建模、语言任务到 Semantic
            ID 与推荐推理的完整故事线。
          </p>
        </div>
        <a href="/series/generative-recommendation">
          Explore the roadmap
          <span aria-hidden="true">↗</span>
        </a>
      </section>

      <BlogExplorer posts={cards} />
      <SiteFooter />
    </main>
  );
}
