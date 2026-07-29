type CounterProps = {
  locale?: "en" | "zh-CN";
};

export function PageViewCounter({ locale = "en" }: CounterProps) {
  return (
    <span
      className="view-counter view-counter-page"
      title={locale === "zh-CN" ? "本文浏览次数" : "Article views"}
    >
      <span
        className="view-counter-value"
        id="vercount_value_page_pv"
        suppressHydrationWarning
      >
        —
      </span>
      <span>{locale === "zh-CN" ? "次阅读" : "views"}</span>
    </span>
  );
}

export function SiteTrafficCounter({ locale = "en" }: CounterProps) {
  return (
    <div
      className="site-traffic-counter"
      aria-label={locale === "zh-CN" ? "全站访问统计" : "Site traffic"}
    >
      <span>
        <strong id="vercount_value_site_pv" suppressHydrationWarning>
          —
        </strong>
        {locale === "zh-CN" ? "全站阅读" : "site views"}
      </span>
      <span>
        <strong id="vercount_value_site_uv" suppressHydrationWarning>
          —
        </strong>
        {locale === "zh-CN" ? "位访客" : "visitors"}
      </span>
    </div>
  );
}
