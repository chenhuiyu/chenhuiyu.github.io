import {b,type Localized} from './schema';
import type {SourceId} from './sources';
export type Unit={topic:string;source:SourceId;name:Localized;core:Localized;trap:Localized;check:Localized;boundary:Localized};
// Each authored row has five bilingual fields: concept, mechanism, false claim,
// falsifiable experiment, and limitation. No numeric parameter multiplication.
export function units(topic:string,source:SourceId,rows:string):Unit[]{return rows.trim().split('\n').filter(Boolean).map((line,index)=>{
 const f=line.split('||').map(s=>s.trim());
 if(f.length!==10||f.some(x=>!x))throw Error(`Invalid curriculum row ${topic}:${index+1}`);
 return {topic,source,name:b(f[0],f[1]),core:b(f[2],f[3]),trap:b(f[4],f[5]),check:b(f[6],f[7]),boundary:b(f[8],f[9])};
})}
