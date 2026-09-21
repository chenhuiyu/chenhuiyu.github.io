import type {Metadata} from 'next';
const root='https://chenhuiyu.github.io/interview';
export function interviewMetadata(en:boolean):Metadata {
 const title=en?'LLM Interview Practice — Huiyu Chen':'LLM 面试练习站 — Huiyu Chen';
 const description=en?'1,000 bilingual LLM practice problems: 232 choice, 32 executable Python and 736 self-assessed written drills, with references, explanations and local progress.':'1,000 道中英双语 LLM 练习：232 道选择题、32 道可运行 Python、736 道自评问答。184 个递进专题，含参考资料、解析和本地进度。';
 return {title,description,alternates:{canonical:root+(en?'/en':''),languages:{'zh-CN':root,en:root+'/en','x-default':root+'/en'}},openGraph:{title,description,url:root+(en?'/en':''),type:'website'},twitter:{card:'summary',title,description}};
}
export function interviewSchema(en:boolean){return {'@context':'https://schema.org','@type':'LearningResource',name:en?'LLM Interview Practice':'LLM 面试练习站',url:root+(en?'/en':''),inLanguage:en?'en':'zh-CN',learningResourceType:'Interactive practice problems',isAccessibleForFree:true,author:{'@type':'Person',name:'Huiyu Chen'}}}
