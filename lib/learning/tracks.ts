export type Locale = 'zh-CN' | 'en';
export const tracks = [
  {id:'llm-foundations', no:'01', title:{'zh-CN':'大模型：从第一个 token 开始',en:'LLMs, starting with one token'}, description:{'zh-CN':'从张量、注意力和训练目标，到真实模型输入输出、检索、MoE 与推理时计算。',en:'Tensors, attention and training objectives, then real model I/O, retrieval, MoE and inference-time compute.'}, tags:['Token → Tensor','Train → Align','Inspect → Evaluate'], lab:'attention', notebook:'llm-model-io'},
  {id:'llm-infra', no:'02', title:{'zh-CN':'LLM Infra：让模型高效运行',en:'LLM infrastructure, from memory to throughput'}, description:{'zh-CN':'vLLM、SGLang、TensorRT-LLM、Megatron-LM、DeepSpeed，以及 GPU/TPU、内核、分片和性能分析。',en:'vLLM, SGLang, TensorRT-LLM, Megatron-LM and DeepSpeed, with GPU/TPU architecture, kernels, sharding and profiling.'}, tags:['Serve','Shard','Profile'],lab:'infra',notebook:'infra-benchmark'},
  {id:'multimodal', no:'03', title:{'zh-CN':'多模态：从像素到内容理解',en:'Multimodal understanding, from pixels to meaning'}, description:{'zh-CN':'图像 patch、CLIP、masked reconstruction、视觉语言模型与视频时间轴：理解模型看见了什么。',en:'Image patches, CLIP, masked reconstruction, vision-language models and video timelines: understand what a model sees.'},tags:['Align','Reconstruct','Ground'],lab:'patches',notebook:'multimodal-inspection'},
  {id:'hstu', no:'04', title:{'zh-CN':'HSTU：从行为序列到生成式推荐',en:'HSTU, from actions to generative recommendation'}, description:{'zh-CN':'逐张量拆解 HSTU，训练一个微型序列推荐器，再理解 ragged execution、候选打分与离线评估。',en:'Trace HSTU tensor by tensor, train a tiny sequential recommender, then examine ragged execution, candidate scoring and evaluation.'},tags:['Sequence','Transduce','Rank'],lab:'hstu',notebook:'hstu-from-scratch'},
] as const;
export type Track = typeof tracks[number];
export function getTrack(id: string) { return tracks.find(t=>t.id===id); }
