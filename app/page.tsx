import { SiteFooter, SiteHeader } from "@/app/components/SiteHeader";
import { countryOrder, travelLocations } from "@/content/travel-data";
import { posts } from "@/lib/posts";

const focusAreas = [
  {
    number: "01",
    title: "Multimodal content understanding",
    copy: "At Meta, I work on LLMs that understand content across language, images, and video—connecting model behavior to real product experiences.",
    note: "text · image · video",
  },
  {
    number: "02",
    title: "Large language model systems",
    copy: "I care about training, alignment, evaluation, and production systems that stay measurable, efficient, and understandable at real scale.",
    note: "train · evaluate · ship",
  },
  {
    number: "03",
    title: "Clear technical writing",
    copy: "I translate papers, experiments, and hard-won debugging lessons into notes that another engineer can actually use.",
    note: "read · question · explain",
  },
];

const featuredPosts = [
  posts.find(
    (post) =>
      post.language === "en" && post.title.includes("Sparse-Linear Attention"),
  ),
  posts.find(
    (post) =>
      post.language === "en" &&
      post.title.includes("Redefining Memory Management"),
  ),
  posts.find(
    (post) =>
      post.language === "zh-CN" &&
      post.title.includes("推理 LLM 的可视化指南"),
  ),
].filter((post) => post !== undefined);

export default function Home() {
  return (
    <main className="site-shell" id="top">
      <SiteHeader />

      <section className="hero personal-hero">
        <div className="hero-copy">
          <p className="eyebrow">
            <span>Huiyu (Yvette) Chen</span>
            <span aria-hidden="true">·</span>
            Machine Learning Engineer at Meta
          </p>

          <h1>
            Hi, I’m Yvette. I build multimodal LLMs that understand{" "}
            <span className="word-mark">content</span>.
          </h1>

          <p className="hero-description">
            I’m a machine learning engineer at Meta, based in Singapore. My
            current work focuses on multimodal content understanding across
            language, images, and video—plus the training, evaluation, and
            production systems that make it useful.
          </p>

          <div className="hero-actions">
            <a
              className="text-link text-link-primary"
              href="https://www.linkedin.com/in/yvette-huiyu-chen"
              target="_blank"
              rel="noreferrer"
            >
              Meet me on LinkedIn <span aria-hidden="true">→</span>
            </a>
            <a className="text-link" href="/blog">
              Explore the blog <span aria-hidden="true">→</span>
            </a>
          </div>

          <p className="tiny-note">
            research brain, product hands{" "}
            <span aria-hidden="true">♡</span>
          </p>
        </div>

        <div className="orbit-stage">
          <div className="orbit-decor" aria-hidden="true">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="orbit orbit-three" />
            <div className="axis axis-x" />
            <div className="axis axis-y" />

            <div className="memory-node node-one">
              <span>see</span>
            </div>
            <div className="memory-node node-two">
              <span>read</span>
            </div>
            <div className="memory-node node-three">
              <span>reason</span>
            </div>
            <div className="memory-node node-four">
              <span>ship</span>
            </div>

            <div className="fish-mark">
              <span>models in motion</span>
            </div>
            <div className="spark-cluster spark-one">· · ✦ ·</div>
            <div className="spark-cluster spark-two">✦ · ·</div>
            <p className="diagram-note">curiosity → systems → impact</p>
          </div>
          <figure className="hero-portrait">
            <img
              alt="Portrait of Huiyu Chen in warm afternoon light"
              src="/photos/profile/yvette-portrait.jpeg"
            />
            <figcaption>
              hello from Singapore <span aria-hidden="true">♡</span>
            </figcaption>
          </figure>
        </div>
      </section>

      <section className="now-strip" aria-label="Professional introduction">
        <p className="section-kicker">Now · 现在</p>
        <p>
          Building multimodal content-understanding LLMs at{" "}
          <strong>Meta</strong>—across language, images, and video—while
          staying interested in the whole path from how a model learns to how a
          person experiences it.
        </p>
        <a
          href="https://www.linkedin.com/in/yvette-huiyu-chen"
          target="_blank"
          rel="noreferrer"
        >
          Full profile ↗︎
        </a>
      </section>

      <section className="focus-section" id="focus">
        <div className="section-intro">
          <p className="section-kicker">What I do · 研究方向</p>
          <div>
            <h2>Model work, systems work, and the bridge between them.</h2>
            <p className="section-summary">
              I’m happiest in the messy middle—where a promising idea has to
              become a reliable, measurable experience.
            </p>
          </div>
        </div>

        <div className="focus-grid">
          {focusAreas.map((area) => (
            <article className="focus-card" key={area.number}>
              <div className="card-topline">
                <span>{area.number}</span>
                <p>{area.note}</p>
              </div>
              <h3>{area.title}</h3>
              <p>{area.copy}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="journey-section">
        <div className="journey-title">
          <p className="section-kicker">A short timeline · 简历</p>
          <h2>
            The current chapter,
            <br />
            plus just enough backstory.
          </h2>
          <p className="about-handnote">details belong on LinkedIn 〰</p>
        </div>

        <div className="journey-list">
          <article>
            <p className="journey-date">Now</p>
            <div>
              <h3>Machine Learning Engineer · Meta</h3>
              <p>
                Building multimodal content-understanding LLM systems across
                text, images, and video.
              </p>
            </div>
          </article>
          <article>
            <p className="journey-date">Previously</p>
            <div>
              <h3>Senior Machine Learning Engineer · Shopee</h3>
              <p>
                Built production AI assistants for e-commerce across multiple
                markets. That chapter taught me how to carry an LLM idea from
                training to real users—without turning this homepage into a
                quarterly report.
              </p>
            </div>
          </article>
          <article>
            <p className="journey-date">Before that</p>
            <div>
              <h3>Computer Science · CASIA</h3>
              <p>
                Research training in NLP and machine learning, with a lasting
                habit of reading the appendix.
              </p>
            </div>
          </article>
          <article className="journey-personal">
            <p className="journey-date">Off screen</p>
            <div>
              <h3>
                Bouldering, tennis, hiking, swimming—and too many stairs.
              </h3>
              <p>Still learning, still moving, still evolving.</p>
            </div>
          </article>
        </div>
      </section>

      <section className="speaking-section" id="speaking">
        <div className="speaking-heading">
          <p className="section-kicker">Speaking · 演讲</p>
          <h2>One stage, one production LLM story, twenty-seven slides.</h2>
          <p>
            A practical talk about taking chatbot assistants from architecture
            decisions and model training to alignment and production scale.
          </p>
        </div>

        <article className="talk-card">
          <a
            aria-label="Open the 27-slide APAC Data Innovation Summit presentation"
            className="talk-cover"
            href="/talks/llm-powered-chatbot-assistants-apac-dis-2026.pdf"
            target="_blank"
            rel="noreferrer"
          >
            <img
              alt="Title slide for LLM-Powered Chatbot Assistants"
              src="/talks/apac-dis-2026-cover.png"
            />
            <span>Open 27 slides ↗︎</span>
          </a>

          <div className="talk-copy">
            <div className="talk-meta">
              <span>12 Mar 2026</span>
              <span>Singapore</span>
              <span>ML &amp; Generative AI Stage</span>
            </div>
            <p className="talk-event">APAC Data Innovation Summit 2026</p>
            <h3>
              LLM-Powered Chatbot Assistants: Elevating the Customer Journey
            </h3>
            <p>
              I shared the system and model choices behind production-grade
              conversational AI—from retrieval and fine-tuning to alignment,
              evaluation, and deployment.
            </p>
            <div className="talk-actions">
              <a
                className="text-link text-link-primary"
                href="/talks/llm-powered-chatbot-assistants-apac-dis-2026.pdf"
                target="_blank"
                rel="noreferrer"
              >
                View the slides <span aria-hidden="true">→</span>
              </a>
              <a
                className="text-link"
                href="https://www.linkedin.com/in/yvette-huiyu-chen/recent-activity/all/"
                target="_blank"
                rel="noreferrer"
              >
                LinkedIn highlight <span aria-hidden="true">↗︎</span>
              </a>
            </div>
          </div>
        </article>
      </section>

      <section className="writing-section" id="writing">
        <div className="section-intro writing-intro">
          <p className="section-kicker">Writing · 写作</p>
          <div>
            <h2>All the old notes, now in one much better-dressed archive.</h2>
            <a className="text-link text-link-primary" href="/blog">
              Browse all 102 posts <span aria-hidden="true">→</span>
            </a>
          </div>
        </div>

        <div className="writing-list">
          {featuredPosts.map((article, index) => (
            <a
              className="writing-card"
              href={`/blog/${article.slug}`}
              key={article.slug}
            >
              <p className="writing-number">0{index + 1}</p>
              <div className="writing-meta">
                <span>{article.category}</span>
                <span>{article.date}</span>
              </div>
              <h3>{article.title}</h3>
              <p>{article.excerpt}</p>
              <span className="card-arrow" aria-hidden="true">
                ↗︎
              </span>
            </a>
          ))}
        </div>
      </section>

      <section className="little-life-section">
        <div className="section-intro">
          <p className="section-kicker">Elsewhere · 生活支线</p>
          <div>
            <h2>There is life outside the terminal, allegedly.</h2>
            <p className="section-summary">
              A spinning planet of places I’ve been, plus the less polished
              thoughts that escape onto Xiaohongshu.
            </p>
          </div>
        </div>
        <div className="little-life-grid">
          <a className="little-life-card travel-life-card" href="/travel">
            <span className="life-icon" aria-hidden="true">◎</span>
            <p>
              {countryOrder.length} countries &amp; regions ·{" "}
              {travelLocations.length} places
            </p>
            <h3>Spin my travel globe</h3>
            <span>Footprints, photos, and tiny postcards →</span>
          </a>
          <a
            className="little-life-card red-life-card"
            href="/xiaohongshu"
          >
            <span className="life-icon" aria-hidden="true">♡</span>
            <p>@乌节路陈女士</p>
            <h3>See my Xiaohongshu shelf</h3>
            <span>攀岩、坡漂生活与伤痛文学 →</span>
          </a>
        </div>
      </section>

      <SiteFooter />
    </main>
  );
}
