import type {Metadata} from 'next';
const root='https://chenhuiyu.github.io/lab/reconstruction';
export function reconstructionMetadata(en:boolean):Metadata{
 const title=en?'Reconstruction Lab — Huiyu Chen':'Reconstruction 实验室 — Huiyu Chen';
 const description=en?'Mask handwriting, reconstruct hidden pixels, test recognition and walk through latent space with a locally trained masked autoencoder.':'亲手遮挡笔画、重建隐藏像素、检验数字识别，并在潜空间中漫游。真实训练的 masked autoencoder 在浏览器本地运行。';
 return {title,description,alternates:{canonical:root+(en?'/en':''),languages:{'zh-CN':root,en:root+'/en','x-default':root+'/en'}},openGraph:{title,description,url:root+(en?'/en':''),type:'website',locale:en?'en_US':'zh_CN'},twitter:{card:'summary',title,description}};
}
export function reconstructionSchema(en:boolean){return {'@context':'https://schema.org','@type':'LearningResource',name:en?'Reconstruction Lab':'Reconstruction 实验室',url:root+(en?'/en':''),inLanguage:en?'en':'zh-CN',learningResourceType:'Interactive experiment',educationalLevel:'Beginner to intermediate',author:{'@type':'Person',name:'Huiyu Chen'},isAccessibleForFree:true,about:'Masked reconstruction, learned representations and visual content recognition'}}
