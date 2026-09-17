export type RecordEntry = { status:'solved'|'review'; attempts:number; viewed?:boolean; bookmarked?:boolean };
export type Progress = Record<string,RecordEntry>;
export const storageKey='huiyu-interview-v1';
export function readProgress(raw:string|null,validIds:string[]):Progress {
  if(!raw)return {};
  try {
    const parsed=JSON.parse(raw),result:Progress={};
    if(!parsed || typeof parsed!=='object')return result;
    for(const id of validIds){const v=parsed[id];if(v&&['solved','review'].includes(v.status)&&Number.isInteger(v.attempts)&&v.attempts>=0)result[id]={status:v.status,attempts:v.attempts,viewed:!!v.viewed,bookmarked:!!v.bookmarked};}
    return result;
  }catch{return {}}
}
