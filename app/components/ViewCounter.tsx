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
