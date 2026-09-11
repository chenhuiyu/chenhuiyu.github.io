import type {Metadata} from 'next';
const root='https://chenhuiyu.github.io/lab/transformer';
export function observatoryMetadata(en:boolean):Metadata{
 const title=en?'The Model Observatory — Huiyu Chen':'Transformer 解剖台 — Huiyu Chen';
 const description=en?'Inspect a trained tiny Transformer in 3D. Edit its input, trace real activations and attention, and ablate heads to compare next-token predictions.':'在浏览器里打开一个真实训练的微型 Transformer：修改输入、旋转三维激活矩阵、查看注意力，切断注意力头并比较预测变化。';
 const url=root+(en?'/en':'');
 return {title,description,alternates:{canonical:url,languages:{'zh-CN':root,en:root+'/en','x-default':root+'/en'}},openGraph:{title,description,url,type:'website',locale:en?'en_US':'zh_CN'},twitter:{card:'summary',title,description}};
}
export function observatorySchema(en:boolean){return {'@context':'https://schema.org','@type':'LearningResource',name:en?'The Model Observatory':'Transformer 解剖台',url:root+(en?'/en':''),inLanguage:en?'en':'zh-CN',learningResourceType:'Interactive experiment',educationalLevel:'Beginner to intermediate',author:{'@type':'Person',name:'Huiyu Chen'},isAccessibleForFree:true,about:'Causal Transformer inference, attention and head ablation'}}
