'use client';
import {useEffect,useState} from 'react';
import type {Question} from '@/lib/interview/schema';
export function WrittenAnswer({q,en,onGrade,onReveal,revealed}:{q:Question;en:boolean;onGrade:(ok:boolean)=>void;onReveal:()=>void;revealed:boolean}){
 const t=(zh:string,eng:string)=>en?eng:zh;
 const [answer,setAnswer]=useState(''),[checked,setChecked]=useState<number[]>([]),[notice,setNotice]=useState(''),[error,setError]=useState(false);
 useEffect(()=>{try{
  // Restoring an external browser draft after SSR hydration is intentional.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  setAnswer(localStorage.getItem(`huiyu-interview-written-${q.id}`)?.slice(0,20000)??'');
 }catch{setError(true)}},[q.id]);
 function edit(text:string){setAnswer(text);setChecked([]);setNotice('');try{localStorage.setItem(`huiyu-interview-written-${q.id}`,text)}catch{setError(true)}}
 function grade(ok:boolean){onGrade(ok);setNotice(ok?t('已记录：自评已掌握。','Recorded: self-assessed mastery.'):t('已记录：需要复习。','Recorded: needs review.'))}
 return <section className="ip-written" aria-label={t('书面作答与自评','Written answer and self-assessment')}>
  <label htmlFor={`written-${q.id}`}>{t('先写下你的回答','Write your answer first')}</label>
  <textarea id={`written-${q.id}`} value={answer} onChange={e=>edit(e.target.value)} maxLength={20000} placeholder={t('按“结论 → 原因 → 例子或验证 → 边界”组织回答…','Structure your answer: conclusion → reasoning → example or test → limits…')}/>
  <p className="ip-written-note">{error?t('本机存储不可用，请自行保留答案。','Local storage unavailable; keep your own copy.'):t('回答自动保存在本机。此题使用参考清单自评，不进行自动语义判分。','Answers autosave locally. Use the checklist to self-assess; no automatic semantic grading is performed.')}</p>
  {!revealed&&<button className="ip-primary" onClick={onReveal}>{t('对照参考要点','Compare with reference points')}</button>}
  {revealed&&<div className="ip-rubric"><h3>{t('核对我的回答','Check my answer')}</h3><p>{t('只有你的回答确实覆盖了该要点，才勾选。编辑回答会清空本次勾选。','Check only points your answer actually covers. Editing resets this checklist.')}</p>{q.rubric!.map((point,i)=><label key={i}><input type="checkbox" checked={checked.includes(i)} onChange={e=>setChecked(v=>e.target.checked?[...v,i]:v.filter(n=>n!==i))}/><span>{en?point.en:point.zh}</span></label>)}<div className="ip-answer-actions"><button className="ip-primary" disabled={answer.trim().length<10||checked.length!==q.rubric!.length} onClick={()=>grade(true)}>{t('记录自评：已掌握','Record: self-assessed mastery')}</button><button onClick={()=>grade(false)}>{t('记录：需要复习','Record: needs review')}</button></div><p>{t('至少写 10 个字符并完成要点核对，才能记录为自评已掌握。','Write at least 10 characters and check every point before recording mastery.')}</p></div>}
  {notice&&<p className="ip-feedback success" role="status">{notice}</p>}
 </section>
}
