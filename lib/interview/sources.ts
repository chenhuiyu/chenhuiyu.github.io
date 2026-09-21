import {b} from './schema';
// Reference material, never a claim that an employer asked our authored prompt.
export const sources = {
 cs336:{title:'Stanford CS336 · Language Modeling from Scratch',url:'https://cs336.stanford.edu/',kind:'course'},
 basics:{title:'Stanford CS336 · Assignment 1 repository',url:'https://github.com/stanford-cs336/assignment1-basics',kind:'course'},
 mlbook:{title:'Chip Huyen · Machine Learning Interviews Book',url:'https://github.com/chiphuyen/ml-interviews-book',kind:'community'},
 vllm:{title:'vLLM documentation',url:'https://docs.vllm.ai/en/latest/',kind:'official'},
 sglang:{title:'SGLang documentation',url:'https://docs.sglang.io/',kind:'official'},
 trt:{title:'NVIDIA TensorRT-LLM documentation',url:'https://nvidia.github.io/TensorRT-LLM/',kind:'official'},
 megatron:{title:'NVIDIA Megatron Core',url:'https://docs.nvidia.com/megatron-core/developer-guide/latest/',kind:'official'},
 zero:{title:'DeepSpeed · ZeRO tutorial',url:'https://www.deepspeed.ai/tutorials/zero/',kind:'official'},
 token:{title:'Hugging Face · Tokenization algorithms',url:'https://huggingface.co/docs/transformers/tokenizer_summary',kind:'official'},
 attention:{title:'Attention Is All You Need',url:'https://arxiv.org/abs/1706.03762',kind:'paper'},
 lora:{title:'LoRA',url:'https://arxiv.org/abs/2106.09685',kind:'paper'},
 qlora:{title:'QLoRA',url:'https://arxiv.org/abs/2305.14314',kind:'paper'},
 mamba:{title:'Mamba',url:'https://arxiv.org/abs/2312.00752',kind:'paper'},
 dpo:{title:'Direct Preference Optimization',url:'https://arxiv.org/abs/2305.18290',kind:'paper'},
 flash:{title:'FlashAttention',url:'https://arxiv.org/abs/2205.14135',kind:'paper'},
 rag:{title:'Retrieval-Augmented Generation',url:'https://arxiv.org/abs/2005.11401',kind:'paper'},
 toolformer:{title:'Toolformer',url:'https://arxiv.org/abs/2302.04761',kind:'paper'},
 mae:{title:'Masked Autoencoders Are Scalable Vision Learners',url:'https://arxiv.org/abs/2111.06377',kind:'paper'},
 clip:{title:'Learning Transferable Visual Models From Natural Language Supervision',url:'https://arxiv.org/abs/2103.00020',kind:'paper'},
 hstu:{title:'Actions Speak Louder than Words · HSTU',url:'https://arxiv.org/abs/2402.17152',kind:'paper'},
 helm:{title:'Holistic Evaluation of Language Models',url:'https://arxiv.org/abs/2211.09110',kind:'paper'},
 cuda:{title:'NVIDIA CUDA Programming Guide',url:'https://docs.nvidia.com/cuda/cuda-programming-guide/',kind:'official'},
 tpu:{title:'Google Cloud · TPU architecture',url:'https://cloud.google.com/tpu/docs/system-architecture-tpu-vm',kind:'official'},
 nsight:{title:'NVIDIA Nsight Systems User Guide',url:'https://docs.nvidia.com/nsight-systems/UserGuide/',kind:'official'},
 xprof:{title:'OpenXLA · XProf',url:'https://openxla.org/xprof',kind:'official'},
} as const;
export type SourceId = keyof typeof sources;
export const sourceKinds={course:b('课程与作业','Course & assignment'),official:b('官方文档','Official documentation'),paper:b('论文','Research paper'),community:b('公开面试学习资料','Public interview study material')};
export const provenanceNote=b('参考资料用于知识核对与延伸学习。本站题干、案例与评分要点为原创练习，不是原题转载或经核实的公司真题。','References support study and topic verification. Prompts, scenarios and rubrics are original exercises, not reproduced exams or verified employer questions.');
