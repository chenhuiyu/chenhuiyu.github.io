"use client";

import { useEffect, useRef } from "react";

function counterNumber(element: HTMLElement | null) {
  if (!element) return null;
  const value = Number(element.textContent?.replace(/[^\d.]/g, ""));
  return Number.isFinite(value) && value > 0 ? value : null;
}

export function VisitorPulse() {
  const ratioRef = useRef<HTMLElement>(null);
  const hostRef = useRef<HTMLElement>(null);
  const scopeRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    const pageViews = document.getElementById("busuanzi_value_site_pv");
    const visitors = document.getElementById("busuanzi_value_site_uv");
    const canonicalHost = window.location.hostname === "chenhuiyu.github.io";

    if (hostRef.current) {
      hostRef.current.textContent = canonicalHost
        ? "GitHub Pages · 旧站续计"
        : `${window.location.hostname} · 独立镜像`;
    }

    if (scopeRef.current) {
      scopeRef.current.textContent = canonicalHost
        ? "口径：累计数据沿用旧 Hexo 站的不蒜子同域统计；“新站”数据由 Vercount 从 2026 改版后单独记录。"
        : "口径：你正在访问独立计数的 Sites 镜像；旧 Hexo 的历史累计以 chenhuiyu.github.io 主站为准。";
    }

    const syncRatio = () => {
      const pv = counterNumber(pageViews);
      const uv = counterNumber(visitors);
      if (!ratioRef.current || !pv || !uv) return;
      const ratio = pv / uv;
      ratioRef.current.textContent = `${ratio.toFixed(ratio >= 10 ? 0 : 1)}×`;
    };

    const observer = new MutationObserver(syncRatio);
    [pageViews, visitors].forEach((element) => {
      if (element) {
        observer.observe(element, {
          characterData: true,
          childList: true,
          subtree: true,
        });
      }
    });
    syncRatio();

    return () => observer.disconnect();
  }, []);

  return (
    <section className="visitor-pulse-section" id="visitor-pulse">
      <div className="visitor-pulse-heading">
        <p className="section-kicker">Visitor pulse · 访问脉搏</p>
        <div>
          <h2>Every visit leaves a small ripple.</h2>
          <p>
            旧站的历史没有消失：累计数据重新接回 Hexo
            时期使用的同域计数器；改版后的实时数据则独立保留，两个口径不再混为一谈。
          </p>
        </div>
      </div>

      <div className="visitor-pulse-panel">
        <div
          className="visitor-orbit-visual"
          aria-label="网站累计阅读可视化"
        >
          <span className="visitor-orbit visitor-orbit-one" aria-hidden="true" />
          <span className="visitor-orbit visitor-orbit-two" aria-hidden="true" />
          <span
            className="visitor-orbit visitor-orbit-three"
            aria-hidden="true"
          />
          <span className="visitor-orbit-dot dot-coral" aria-hidden="true" />
          <span className="visitor-orbit-dot dot-sage" aria-hidden="true" />
          <span className="visitor-orbit-dot dot-gold" aria-hidden="true" />

          <div className="visitor-pulse-center">
            <span>同域累计阅读</span>
            <strong id="busuanzi_value_site_pv" suppressHydrationWarning>
              —
            </strong>
            <small>lifetime page views</small>
          </div>
        </div>

        <div className="visitor-pulse-copy">
          <div className="visitor-live-line">
            <span className="visitor-live-dot" aria-hidden="true" />
            <span>Live counters</span>
            <span aria-hidden="true">·</span>
            <strong ref={hostRef}>正在识别统计域名</strong>
          </div>

          <h3>旧站历史，终于接回来了。</h3>
          <p>
            主页过去显示的“全站阅读”其实只统计这次改版之后，而且 GitHub
            Pages 与 Sites 镜像各算各的。现在主数字恢复为旧站延续至今的同域累计值，右侧仍保留新站数据，方便核对增长。
          </p>

          <div className="visitor-metric-grid" aria-live="polite">
            <div>
              <strong id="busuanzi_value_site_uv" suppressHydrationWarning>
                —
              </strong>
              <span>累计访客</span>
              <small>legacy + current</small>
            </div>
            <div>
              <strong ref={ratioRef}>—</strong>
              <span>人均阅读</span>
              <small>views / visitor</small>
            </div>
            <div>
              <strong id="vercount_value_site_pv" suppressHydrationWarning>
                —
              </strong>
              <span>新站阅读</span>
              <small>since redesign</small>
            </div>
            <div>
              <strong id="vercount_value_site_uv" suppressHydrationWarning>
                —
              </strong>
              <span>新站访客</span>
              <small>since redesign</small>
            </div>
          </div>

          <p className="visitor-scope-note" ref={scopeRef}>
            口径：正在识别当前访问域名。
          </p>
        </div>
      </div>
    </section>
  );
}
