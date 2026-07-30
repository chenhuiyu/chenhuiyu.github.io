import { posts } from "@/lib/posts";

type TopicDefinition = {
  id: string;
  eyebrow: string;
  title: string;
  description: string;
  href: string;
  labels: [string, string, string];
  theme: "sage" | "sand" | "ink";
  seriesId?: string;
  pairKey?: string;
  excludePairKeys?: string[];
  unit: string;
};

const topicDefinitions: TopicDefinition[] = [
  {
    id: "generative-recommendation",
    eyebrow: "FOUNDATION SERIES · 基础主线",
    title: "20 篇论文读懂生成式推荐",
    description:
      "从 BPR 的打分器，到 OneReason 的推荐推理：沿着同一个问题，读懂排序、序列、语言任务、Semantic ID 与统一生成。",
    href: "/series/generative-recommendation",
    labels: ["BPR", "Semantic ID", "OneReason"],
    theme: "sage",
    seriesId: "generative-recommendation",
    excludePairKeys: ["generative-recommendation-hands-on"],
    unit: "章",
  },
  {
    id: "generative-recommendation-2026",
    eyebrow: "SEASON 2 · 2026 前沿",
    title: "生成式推荐进入深水区",
    description:
      "从可学习 ID、级联推理和连续 Token，到页面生成、广告部署与泛化审计：追踪 2026 年真正改变问题边界的七篇论文。",
    href: "/series/generative-recommendation-2026",
    labels: ["Learnable ID", "Diffusion", "At scale"],
    theme: "sand",
    seriesId: "generative-recommendation-2026",
    unit: "篇",
  },
  {
    id: "generative-recommendation-lab",
    eyebrow: "INTERACTIVE LAB · 动手实验",
    title: "亲手走一遍生成式推荐",
    description:
      "从 BPR 到 Semantic ID、beam search 与 Trie 约束：可在页面里改 Python，也可下载 Notebook 或直接进入 Colab。",
    href: "/blog/generative-recommendation-hands-on-zh",
    labels: ["BPR", "Semantic ID", "Colab"],
    theme: "ink",
    pairKey: "generative-recommendation-hands-on",
    unit: "个实验",
  },
];

export const homeTopics = topicDefinitions.map((topic) => {
  const topicPosts = posts.filter((post) => {
    if (topic.pairKey) return post.pairKey === topic.pairKey;
    if (post.series !== topic.seriesId) return false;
    return !topic.excludePairKeys?.includes(post.pairKey);
  });
  const storyCount = new Set(topicPosts.map((post) => post.pairKey)).size;

  return {
    ...topic,
    storyCount,
    editionCount: topicPosts.length,
  };
});
