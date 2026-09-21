import {b,choice,type Question,type Localized} from './schema';
import {foundations} from './curriculum-foundations';
import {architecture} from './curriculum-architecture';
import {training} from './curriculum-training';
import {alignment} from './curriculum-alignment';
import {serving} from './curriculum-serving';
import {distributed} from './curriculum-distributed';
import {hardware} from './curriculum-hardware';
import {retrieval} from './curriculum-rag';
import {agents} from './curriculum-agents';
import {multimodal} from './curriculum-multimodal';
import {recommendation} from './curriculum-recommendation';
import {evaluation} from './curriculum-evaluation';
export const curriculum=[...foundations,...architecture,...training,...alignment,...serving,...distributed,...hardware,...retrieval,...agents,...multimodal,...recommendation,...evaluation];
const join=(...parts:Localized[]):Localized=>b(parts.map(x=>x.zh).join('\n\n'),parts.map(x=>x.en).join('\n\n'));
// Five explicitly labelled study angles per authored topic. These are related
// drills, not five claims of independent employer provenance or new concepts.
export const expanded:Question[]=curriculum.flatMap((u,i)=>{
 const start=61+i*5;
 const specific=u.name.en==='QLoRA and quantized bases'?'qlora':u.name.en==='State-space models'?'mamba':u.source;
 const common={topic:u.topic,concept:u.name,sourceIds:[specific]};
 const followup=b(`把“${u.name.zh}”与一个相近方案比较：哪些前提变化会让你的选择反转？`,`Compare ${u.name.en} with a nearby alternative: which changed assumption would reverse your choice?`);
 const valid=i%2===0,claim=valid?u.core:u.trap;
 const judge=choice(start,u.topic,'easy',b(`${u.name.zh} · 论断判断`,`${u.name.en} · claim check`),b(`判断下列论断是否成立。\n\n${claim.zh}`,`Is the following claim valid?\n\n${claim.en}`),valid?[b('成立','Valid'),b('不成立','Invalid')]:[b('不成立','Invalid'),b('成立','Valid')],join(u.core,u.boundary),followup);
 const written=(offset:number,angle:string,title:Localized,prompt:Localized,points:Localized[],difficulty:'medium'|'hard'):Question=>({...common,id:String(start+offset).padStart(3,'0'),kind:'written',difficulty,angle,title,prompt,explanation:join(...points),rubric:points,followup});
 return [{...judge,...common,angle:'claim'},
 written(1,'mechanism',b(`${u.name.zh} · 机制口述`,`${u.name.en} · explain the mechanism`),b(`向初学者解释${u.name.zh}。先说明它要解决的问题，再描述关键运算或数据流，并说明一个不能省略的前提。不要只展开缩写。`,`Explain ${u.name.en} to a beginner: state the problem, describe the key operations or data flow, and identify one essential assumption. Do not merely expand the acronym.`),[u.core,u.boundary],'medium'),
 written(2,'debug',b(`${u.name.zh} · 纠正误区`,`${u.name.en} · challenge a misconception`),b(`代码评审中有人提出：\n\n“${u.trap.zh}”\n\n指出推理错误，给出正确说法，并提出一个能够区分两种说法的检查。`,`A code review includes this claim:\n\n“${u.trap.en}”\n\nIdentify the reasoning error, give the correct statement and propose a check that distinguishes them.`),[u.core,u.check],'medium'),
 written(3,'experiment',b(`${u.name.zh} · 最小验证实验`,`${u.name.en} · minimal experiment`),b(`你需要验证${u.name.zh}的实现或效果。设计最小可复现实验：明确输入或负载、对照、应观察的量，以及结果与预期不一致时先查什么。`,`You need to validate ${u.name.en}. Design a minimal reproducible experiment specifying inputs or workload, a control, observables and the first investigation if results differ from expectations.`),[u.check,u.core,u.boundary],'hard'),
 written(4,'tradeoff',b(`${u.name.zh} · 边界与取舍`,`${u.name.en} · limits and tradeoffs`),b(`评审一个依赖${u.name.zh}的方案。说明它成立的条件、至少一项可能被忽略的成本或失效边界，以及上线前必须补充的验证。`,`Review a proposal relying on ${u.name.en}. Explain when it applies, one overlooked cost or failure boundary, and the validation required before deployment.`),[u.boundary,u.check,u.core],'hard')];
});
