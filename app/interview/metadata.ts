import type {Metadata} from 'next';
const root='https://chenhuiyu.github.io/interview';
export function interviewMetadata(en:boolean):Metadata {
 const title=en?'LLM Interview Practice — Huiyu Chen':'LLM 面试练习站 — Huiyu Chen';
 const description=en?'60 bilingual LLM interview problems across 12 topics: interactive multiple choice, executable Python, test cases, solutions and local progress.':'60 道中英双语 LLM 面试题，覆盖 12 个方向。支持难度筛选、交互选择题、在线 Python 判题、详细解析和本地刷题进度。';
 return {title,description,alternates:{canonical:root+(en?'/en':''),languages:{'zh-CN':root,en:root+'/en','x-default':root+'/en'}},openGraph:{title,description,url:root+(en?'/en':''),type:'website'},twitter:{card:'summary',title,description}};
}
export function interviewSchema(en:boolean){return {'@context':'https://schema.org','@type':'LearningResource',name:en?'LLM Interview Practice':'LLM 面试练习站',url:root+(en?'/en':''),inLanguage:en?'en':'zh-CN',learningResourceType:'Interactive practice problems',isAccessibleForFree:true,author:{'@type':'Person',name:'Huiyu Chen'}}}
