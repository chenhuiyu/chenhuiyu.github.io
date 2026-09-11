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
  standalone?: boolean;
};

const topicDefinitions: TopicDefinition[] = [
  {id: "transformer-observatory", eyebrow: "MODEL OBSERVATORY · 交互实验", title: "打开 Transformer 的黑箱", description: "旋转真实激活的三维矩阵，追踪注意力，切断一个头，观察模型会忘记什么。一个可检查、可干预的微型 Transformer。", href: "/lab/transformer", labels: ["3D tensors", "Attention", "Ablation"], theme: "ink", standalone: true, unit: "个实验"},
  {"id": "llm-foundations", "eyebrow": "MODEL LEARNING · 双语学习路线", "title": "从第一个 token 学懂大模型", "description": "从张量和 Attention，到训练、模型输入输出、RAG 与前沿推理；真实模型与可编辑 Python 陪你动手。", "href": "/learn/llm-foundations", "labels": ["Tensors", "Attention", "Model I/O"], "theme": "ink", "seriesId": "llm-foundations", "unit": "章"},
  {"id": "llm-infra", "eyebrow": "MODEL LEARNING · 双语学习路线", "title": "LLM Infra：从显存到吞吐", "description": "五大训练/推理框架、GPU/TPU、kernel、分片与 profiling，把性能问题变成可验证的实验。", "href": "/learn/llm-infra", "labels": ["vLLM", "Megatron", "XProf"], "theme": "sand", "seriesId": "llm-infra", "unit": "章"},
  {"id": "multimodal", "eyebrow": "MODEL LEARNING · 双语学习路线", "title": "多模态内容理解", "description": "从图像 patch、CLIP 和 masked reconstruction，到 VLM、视频采样与时间定位。", "href": "/learn/multimodal", "labels": ["CLIP", "MAE", "Video"], "theme": "sage", "seriesId": "multimodal", "unit": "章"},
  {"id": "hstu", "eyebrow": "MODEL LEARNING · 双语学习路线", "title": "HSTU：从行为到推荐", "description": "数据、逐张量推导、微型训练与规模化；连接多模态内容表示和序列推荐。", "href": "/learn/hstu", "labels": ["Actions", "HSTU", "Train"], "theme": "ink", "seriesId": "hstu", "unit": "章"},
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
    if (topic.standalone) return false;
    if (topic.pairKey) return post.pairKey === topic.pairKey;
    if (post.series !== topic.seriesId) return false;
    return !topic.excludePairKeys?.includes(post.pairKey);
  });
  const storyCount = new Set(topicPosts.map((post) => post.pairKey)).size;

  return {
    ...topic,
    storyCount: topic.standalone ? 1 : storyCount,
    editionCount: topic.standalone ? 2 : topicPosts.length,
  };
});
