export type Localized = { zh: string; en: string };
export type Difficulty = 'easy' | 'medium' | 'hard';
export type Case = { expression: string; expected?: unknown; raises?: string };
export type Question = {
  id: string; topic: string; difficulty: Difficulty; kind: 'choice' | 'code';
  title: Localized; prompt: Localized; explanation: Localized; followup: Localized;
  options?: Localized[]; answer?: number;
  starter?: string; solution?: string; tests?: Case[]; complexity?: Localized;
};
export const b = (zh: string, en: string): Localized => ({ zh, en });
export function choice(id: number, topic: string, difficulty: Difficulty, title: Localized, prompt: Localized, options: Localized[], explanation: Localized, followup: Localized): Question {
  // Source authors put the correct answer first. Rotate once at build/import time,
  // so the displayed answer position varies but remains stable on repeat visits.
  const rotation = id % options.length;
  return { id: String(id).padStart(3, '0'), topic, difficulty, kind: 'choice', title, prompt,
    options: [...options.slice(rotation), ...options.slice(0, rotation)],
    answer: (options.length - rotation) % options.length, explanation, followup };
}
export const topics = [
  { id: 'foundations', name: b('基础与 Token', 'Foundations & tokens'), source: 'https://huggingface.co/docs/transformers/tokenizer_summary' },
  { id: 'architecture', name: b('Attention 与架构', 'Attention & architecture'), source: 'https://arxiv.org/abs/1706.03762' },
  { id: 'training', name: b('预训练与微调', 'Training & fine-tuning'), source: 'https://arxiv.org/abs/2106.09685' },
  { id: 'alignment', name: b('对齐与推理', 'Alignment & reasoning'), source: 'https://arxiv.org/abs/2305.18290' },
  { id: 'serving', name: b('推理与 Serving', 'Inference & serving'), source: 'https://arxiv.org/abs/2309.06180' },
  { id: 'distributed', name: b('分布式与显存', 'Distributed & memory'), source: 'https://arxiv.org/abs/1910.02054' },
  { id: 'hardware', name: b('Kernel 与量化', 'Kernels & quantization'), source: 'https://arxiv.org/abs/2205.14135' },
  { id: 'rag', name: b('Embedding 与 RAG', 'Embeddings & RAG'), source: 'https://arxiv.org/abs/2005.11401' },
  { id: 'agents', name: b('Agent 与工具', 'Agents & tools'), source: 'https://arxiv.org/abs/2302.04761' },
  { id: 'multimodal', name: b('多模态与重建', 'Multimodal & reconstruction'), source: 'https://arxiv.org/abs/2111.06377' },
  { id: 'recommendation', name: b('HSTU 与推荐', 'HSTU & recommendation'), source: 'https://arxiv.org/abs/2402.17152' },
  { id: 'evaluation', name: b('评测与安全', 'Evaluation & safety'), source: 'https://arxiv.org/abs/2211.09110' },
];
