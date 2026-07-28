import type { Metadata } from "next";
import { SiteFooter, SiteHeader } from "@/app/components/SiteHeader";
import profile from "@/content/xiaohongshu.json";

export const metadata: Metadata = {
  title: "小红书 — 乌节路陈女士",
  description: "乌节路陈女士的小红书公开笔记精选。",
};

export default function XiaohongshuPage() {
  return (
    <main className="site-shell redbook-shell" id="top">
      <SiteHeader />

      <section className="redbook-hero">
        <div className="redbook-stamp" aria-hidden="true">
          <span>RED</span>
          <span>小红书</span>
        </div>
        <div>
          <p className="eyebrow">
            <span>Life, but in red · 生活切片</span>
            <span aria-hidden="true">·</span>
            Singapore
          </p>
          <h1>
            乌节路陈女士，
            <br />
            <span className="word-mark">新加坡伤痛文学</span>博主。
          </h1>
          <p className="redbook-intro">
            工作日训练大模型，业余时间训练攀岩、写段子，以及观察坡漂生活的九十九种结局。
          </p>
        </div>
      </section>

      <section className="redbook-profile">
        <div>
          <p className="section-kicker">On Xiaohongshu · 小红书</p>
          <h2>@{profile.nickname}</h2>
          <p>{profile.description}</p>
          <span>IP · {profile.location}</span>
        </div>
        <div className="redbook-metrics" aria-label="Xiaohongshu profile summary">
          <p>
            <strong>{profile.followers}</strong>
            <span>粉丝</span>
          </p>
          <p>
            <strong>{profile.likesAndCollects}</strong>
            <span>获赞与收藏</span>
          </p>
          <p>
            <strong>{profile.following}</strong>
            <span>关注</span>
          </p>
        </div>
        <a
          className="redbook-button"
          href={profile.profileUrl}
          rel="noreferrer"
          target="_blank"
        >
          去小红书找我 <span aria-hidden="true">↗︎</span>
        </a>
      </section>

      <section className="redbook-notes">
        <div className="redbook-section-title">
          <p className="section-kicker">Public notes · 公开笔记</p>
          <h2>最近在小红书胡说八道的我。</h2>
          <p>
            去掉广告，只留下生活、坡漂、攀岩和那些值得认真胡说八道的瞬间。
          </p>
        </div>

        <div className="redbook-grid">
          {profile.notes.map((note, index) => (
            <a
              className={`redbook-card redbook-card-${(index % 3) + 1}`}
              href={note.url}
              key={note.id}
              rel="noreferrer"
              target="_blank"
            >
              <div className="redbook-image-wrap">
                {note.image ? (
                  <img alt={note.title} src={note.image} />
                ) : (
                  <div className="redbook-text-cover" aria-hidden="true">
                    <span>{note.coverMark}</span>
                    <p>{note.title}</p>
                    <small>乌节路陈女士 · 小红书</small>
                  </div>
                )}
                {note.sticky ? <span>置顶</span> : null}
              </div>
              <div className="redbook-card-copy">
                <p>{String(index + 1).padStart(2, "0")}</p>
                <h3>{note.title}</h3>
                <p className="redbook-card-excerpt">{note.excerpt}</p>
                {note.likes !== null ? (
                  <div>
                    <span>♡ {note.likes}</span>
                    <span>⌁ {note.comments}</span>
                    <span>☆ {note.collects}</span>
                  </div>
                ) : (
                  <div className="redbook-open-note">
                    <span>打开小红书阅读全文</span>
                    <span aria-hidden="true">↗︎</span>
                  </div>
                )}
              </div>
            </a>
          ))}
        </div>
      </section>

      <SiteFooter />
    </main>
  );
}
