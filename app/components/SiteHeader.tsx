export function SiteHeader() {
  return (
    <header className="site-header" aria-label="Main navigation">
      <a className="brand" href="/" aria-label="Huiyu Chen, home">
        Huiyu Chen
        <span className="brand-spark" aria-hidden="true">
          ✦
        </span>
      </a>

      <nav className="nav-links" aria-label="Primary">
        <a href="/#focus">Focus</a>
        <a href="/blog">Blog</a>
        <a href="/travel">Travel</a>
        <a href="/xiaohongshu">小红书</a>
        <a
          href="https://www.linkedin.com/in/yvette-huiyu-chen"
          target="_blank"
          rel="noreferrer"
        >
          LinkedIn
        </a>
      </nav>

      <p className="status-note">
        <span className="status-dot" aria-hidden="true" />
        Meta · Multimodal LLMs
      </p>
    </header>
  );
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div>
        <p className="footer-name">Huiyu Chen</p>
        <p>Building thoughtful multimodal AI in Singapore.</p>
      </div>
      <p className="footer-evolving">
        <span className="mini-fish" aria-hidden="true" />
        黑头呆鱼，还在进化中 ♡
      </p>
      <a href="#top">Back to top ↑</a>
    </footer>
  );
}
