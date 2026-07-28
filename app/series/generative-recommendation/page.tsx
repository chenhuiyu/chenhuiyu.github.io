import type { Metadata } from "next";
import { SeriesPage } from "./SeriesPage";

export const metadata: Metadata = {
  title: "推荐系统如何学会说出答案 — Huiyu Chen",
  description:
    "从 BPR 的打分器，到 OneReason 的推荐推理：20 篇论文串起生成式推荐的完整故事。",
};

export default function GenerativeRecommendationSeriesPage() {
  return <SeriesPage locale="zh-CN" />;
}
