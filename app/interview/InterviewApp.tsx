'use client';
/* Browser storage and URL state are restored after SSR hydration; storage failures
   also update visible status. These effects intentionally synchronize external state. */
/* eslint-disable react-hooks/set-state-in-effect */
import {useEffect,useMemo,useRef,useState} from 'react';
import {questions,topics,type Question,type Localized} from '@/lib/interview/bank';
import {readProgress,storageKey,type Progress,type RecordEntry} from '@/lib/interview/progress';

type Filters={query:string;topic:string;difficulty:string;kind:string;status:string};
type Result={cases?:{passed:boolean;actual:string}[];error?:string;output?:string;code:string;submitted:boolean};
const defaults:Filters={query:'',topic:'all',difficulty:'all',kind:'all',status:'all'};
const level={easy:{zh:'简单',en:'Easy'},medium:{zh:'中等',en:'Medium'},hard:{zh:'困难',en:'Hard'}};
function statusOf(p?:RecordEntry){return !p?.attempts?'new':p.status}
function matches(q:Question,f:Filters,p:Progress){return (f.topic==='all'||q.topic===f.topic)&&(f.difficulty==='all'||q.difficulty===f.difficulty)&&(f.kind==='all'||q.kind===f.kind)&&(f.status==='all'||(f.status==='bookmarked'?p[q.id]?.bookmarked:statusOf(p[q.id])===f.status))&&`${q.id} ${q.title.zh} ${q.title.en} ${q.prompt.zh} ${q.prompt.en}`.toLowerCase().includes(f.query.toLowerCase().trim())}
function Paragraphs({text}:{text:string}){return <>{text.split('\n\n').map((p,i)=><p key={i}>{p}</p>)}</>}

export default function InterviewApp({en=false}:{en?:boolean}){
 const tr=(s:Localized)=>en?s.en:s.zh,t=(zh:string,eng:string)=>en?eng:zh;
 const [id,setId]=useState('001'),[filters,setFilters]=useState<Filters>(defaults),[progress,setProgress]=useState<Progress>({}),[loaded,setLoaded]=useState(false),[storageError,setStorageError]=useState(false),[listOpen,setListOpen]=useState(false);
 useEffect(()=>{
  try{setProgress(readProgress(localStorage.getItem(storageKey),questions.map(q=>q.id)))}catch{setStorageError(true)}
  const restore=()=>{const q=new URLSearchParams(location.hash.slice(1)).get('q');if(questions.some(x=>x.id===q))setId(q!)};
  restore();setLoaded(true);window.addEventListener('hashchange',restore);return()=>window.removeEventListener('hashchange',restore);
 },[]);
 useEffect(()=>{if(loaded)try{localStorage.setItem(storageKey,JSON.stringify(progress))}catch{setStorageError(true)}},[loaded,progress]);
 const filtered=useMemo(()=>questions.filter(q=>matches(q,filters,progress)),[filters,progress]);
 const current=questions.find(q=>q.id===id)??questions[0],at=filtered.findIndex(q=>q.id===id);
 const solved=questions.filter(q=>statusOf(progress[q.id])==='solved').length,review=questions.filter(q=>statusOf(progress[q.id])==='review').length;
 function choose(next:string){setId(next);setListOpen(false);history.replaceState(null,'',`#q=${next}`)}
 function filter(patch:Partial<Filters>){const next={...filters,...patch};setFilters(next);const list=questions.filter(q=>matches(q,next,progress));if(list.length&&!list.some(q=>q.id===id))choose(list[0].id)}
 function update(qid:string,patch:Partial<RecordEntry>){setProgress(p=>({...p,[qid]:{...(p[qid]??{status:'review',attempts:0}),...patch}}))}
 function grade(passed:boolean){setProgress(p=>({...p,[id]:{...p[id],status:passed?'solved':'review',attempts:(p[id]?.attempts??0)+1}}))}
 return <main className="ip-app" lang={en?'en':'zh-CN'}>
  <header className="ip-header"><a className="ip-brand" href="/">Huiyu Chen <span>/</span> <strong>LLM Practice</strong></a><nav><a href={en?'/learn/en':'/learn'}>{t('学习路线','Learning paths')}</a><a href={(en?'/interview':'/interview/en')+`#q=${id}`}>{en?'中文':'EN'}</a><a href="/lab/transformer">Lab ↗</a></nav></header>
  <div className="ip-mobile-bar"><button onClick={()=>setListOpen(v=>!v)} aria-expanded={listOpen}>{t('题目列表与筛选','Problems & filters')} <span>{filtered.length}</span></button><span>{solved}/{questions.length} {t('已通过','solved')}</span></div>
  <div className="ip-layout">
   <aside className={`ip-sidebar ${listOpen?'is-open':''}`} aria-label={t('题库导航','Question bank')}>
    <div className="ip-sidebar-head"><div><p className="ip-eyebrow">THE INTERVIEW COLLECTION</p><h1>{t('把理解写成答案。','Turn understanding into answers.')}</h1></div><span className="ip-bank-size">{questions.length}<small>{t('道双语题','bilingual problems')}</small></span></div>
    <div className="ip-progress"><div><span>{t('已通过','Solved')} <b>{solved}</b></span><span>{t('待复习','Review')} <b>{review}</b></span><span>{questions.length-solved-review} {t('未做','new')}</span></div><div className="ip-progress-track"><i style={{width:`${100*solved/questions.length}%`}}/></div><p>{t('进度和代码草稿仅保存在当前浏览器。','Progress and code drafts stay in this browser only.')}</p>{storageError&&<p role="status">{t('浏览器存储不可用；本次仍可练习，但刷新后可能丢失进度。','Browser storage unavailable; practice still works, but a refresh may lose progress.')}</p>}</div>
    <div className="ip-filters"><label className="ip-search"><span className="ip-sr">{t('搜索题目','Search problems')}</span><input value={filters.query} onChange={e=>filter({query:e.target.value})} placeholder={t('搜索题目、概念、题号…','Search questions, concepts, IDs…')}/><span aria-hidden="true">⌕</span></label>
     <label>{t('方向','Topic')}<select value={filters.topic} onChange={e=>filter({topic:e.target.value})}><option value="all">{t('全部 12 个方向','All 12 topics')}</option>{topics.map(x=><option key={x.id} value={x.id}>{tr(x.name)}</option>)}</select></label>
     <div className="ip-filter-row"><label>{t('难度','Difficulty')}<select value={filters.difficulty} onChange={e=>filter({difficulty:e.target.value})}><option value="all">{t('全部难度','All levels')}</option>{Object.entries(level).map(([key,value])=><option value={key} key={key}>{tr(value)}</option>)}</select></label><label>{t('题型','Type')}<select value={filters.kind} onChange={e=>filter({kind:e.target.value})}><option value="all">{t('全部题型','All types')}</option><option value="choice">{t('选择题','Multiple choice')}</option><option value="code">{t('代码题','Python code')}</option></select></label></div>
     <label>{t('练习状态','Practice status')}<select value={filters.status} onChange={e=>filter({status:e.target.value})}>{[['all',t('全部状态','All states')],['new',t('未作答','Not attempted')],['review',t('待复习','Needs review')],['solved',t('已通过','Solved')],['bookmarked',t('收藏','Bookmarked')]].map(([v,l])=><option value={v} key={v}>{l}</option>)}</select></label>
    </div>
    <div className="ip-list-heading"><span>{filtered.length} {t('道题','problems')}</span><button disabled={filtered.length<2} onClick={()=>{const pool=filtered.filter(q=>q.id!==id);choose(pool[Math.floor(Math.random()*pool.length)].id)}}>{t('随机一题','Shuffle')} ↗</button></div>
    <div className="ip-question-list">{filtered.length?filtered.map(q=><button key={q.id} className={q.id===id?'active':''} onClick={()=>choose(q.id)} aria-current={q.id===id?'true':undefined}><span className={`ip-status-dot ${statusOf(progress[q.id])}`}>{statusOf(progress[q.id])==='solved'?'✓':statusOf(progress[q.id])==='review'?'↻':''}</span><div><span className="ip-list-title">{tr(q.title)}{progress[q.id]?.bookmarked&&<small> ★</small>}</span><span className="ip-list-meta">{q.id} <i/> {q.kind==='code'?'Python':t('选择题','Choice')} <i/> <b className={`ip-level-${q.difficulty}`}>{tr(level[q.difficulty])}</b></span></div><span className="ip-list-arrow">↗</span></button>):<div className="ip-empty"><p>{t('没有符合条件的题目。','No matching problems.')}</p><button onClick={()=>filter(defaults)}>{t('清除筛选','Clear filters')}</button></div>}</div>
    <a className="ip-scope" href={en?'/learn/llm-infra/en':'/learn/llm-infra'}>{t('GPU 工程实践 → LLM Infra 学习路线','GPU engineering practice → LLM Infra path')} ↗</a>
   </aside>
   <section className="ip-main" aria-label={t('练习区域','Practice workspace')}>
    <div className="ip-problem-nav"><span>{tr(topics.find(x=>x.id===current.topic)!.name)} <i>/</i> {current.id}</span><div><button onClick={()=>update(id,{bookmarked:!progress[id]?.bookmarked})} aria-pressed={!!progress[id]?.bookmarked} aria-label={t('收藏当前题','Bookmark this problem')}>{progress[id]?.bookmarked?'★':'☆'}</button><button disabled={at<=0} onClick={()=>choose(filtered[at-1].id)} aria-label={t('上一题','Previous problem')}>←</button><button disabled={!filtered.length||at===filtered.length-1} onClick={()=>choose(filtered[at+1]?.id??filtered[0].id)} aria-label={t('下一题','Next problem')}>→</button></div></div>
    <Workspace key={current.id} q={current} en={en} record={progress[id]} onGrade={grade} onViewed={()=>update(id,{viewed:true})}/>
   </section>
  </div>
 </main>
}

function Workspace({q,en,record,onGrade,onViewed}:{q:Question;en:boolean;record?:RecordEntry;onGrade:(ok:boolean)=>void;onViewed:()=>void}){
 const tr=(s:Localized)=>en?s.en:s.zh,t=(zh:string,eng:string)=>en?eng:zh;
 const [selected,setSelected]=useState<number|null>(null),[checked,setChecked]=useState(false),[showSolution,setShowSolution]=useState(false),[code,setCode]=useState(q.starter??''),[hydrated,setHydrated]=useState(false),[draftError,setDraftError]=useState(false),[result,setResult]=useState<Result|null>(null),[phase,setPhase]=useState<'idle'|'loading'|'running'>('idle'),[notice,setNotice]=useState('');
 const worker=useRef<Worker|null>(null),timer=useRef<ReturnType<typeof setTimeout>|null>(null),runId=useRef(0),codeRef=useRef(code),busy=useRef(false),editor=useRef<HTMLTextAreaElement>(null);
 const gradeRef=useRef(onGrade);useEffect(()=>{gradeRef.current=onGrade},[onGrade]);
 useEffect(()=>{if(q.kind==='code')try{const saved=localStorage.getItem(`huiyu-interview-code-${q.id}`);if(saved!==null){setCode(saved.slice(0,50000));codeRef.current=saved.slice(0,50000)}}catch{setDraftError(true)}setHydrated(true);return()=>{if(q.kind==='code')try{localStorage.setItem(`huiyu-interview-code-${q.id}`,codeRef.current)}catch{}worker.current?.terminate();if(timer.current)clearTimeout(timer.current)}},[q.id,q.kind]);
 useEffect(()=>{if(!hydrated||q.kind!=='code')return;const delay=setTimeout(()=>{try{localStorage.setItem(`huiyu-interview-code-${q.id}`,code)}catch{setDraftError(true)}},300);return()=>clearTimeout(delay)},[code,hydrated,q.id,q.kind]);
 function edit(value:string){setCode(value);codeRef.current=value}
 function end(){if(timer.current)clearTimeout(timer.current);busy.current=false;setPhase('idle')}
 function stop(message:string){worker.current?.terminate();worker.current=null;runId.current++;end();setNotice(message)}
 function arm(ms:number){if(timer.current)clearTimeout(timer.current);timer.current=setTimeout(()=>stop(t('执行超时，已停止。检查无限循环或先重试加载。','Timed out and stopped. Check for infinite loops or retry loading.')),ms)}
 function run(submitted:boolean){
  if(busy.current)return;busy.current=true;setNotice('');setResult(null);setPhase('loading');const snapshot=codeRef.current,id=++runId.current;arm(90000);
  try{
   if(!worker.current)worker.current=new Worker('/workers/interview-worker.mjs',{type:'module'});
   worker.current.onmessage=({data})=>{
    if(data.runId!==id||id!==runId.current)return;
    if(data.type==='loading'){setPhase('loading');return}
    if(data.type==='running'){setPhase('running');arm(10000);return}
    if(!['result','error'].includes(data.type))return;
    if(data.type==='error'){worker.current?.terminate();worker.current=null}
    end();const r:Result={...data,code:snapshot,submitted};setResult(r);
    if(data.type==='result'&&submitted&&snapshot===codeRef.current){const ok=!data.error&&data.cases?.length===q.tests!.length&&data.cases.every((c:{passed:boolean})=>c.passed);gradeRef.current(!!ok)}
   };
   worker.current.onerror=()=>{stop(t('Python 运行环境加载失败。请检查网络后重试。','Python runtime failed to load. Check connectivity and retry.'))};
   worker.current.postMessage({runId:id,code:snapshot,tests:submitted?q.tests:q.tests!.slice(0,2)});
  }catch{stop(t('当前浏览器无法启动 Python worker。','This browser could not start the Python worker.'))}
 }
 function reveal(){if(!showSolution)onViewed();setShowSolution(v=>!v)}
 const stale=!!result&&result.code!==code,passed=!!result&&!result.error&&!!result.cases?.length&&result.cases.every(c=>c.passed);
 return <div className={`ip-workspace ${q.kind==='code'?'with-code':''}`}>
  <div className="ip-reading">
   <div className="ip-badges"><span className={`ip-difficulty ip-level-${q.difficulty}`}>{tr(level[q.difficulty])}</span><span>{q.kind==='code'?t('代码 · Python 3','Code · Python 3'):t('单项选择','Single choice')}</span>{statusOf(record)==='solved'&&<span className="ip-solved-badge">✓ {t('已通过','Solved')}</span>}</div>
   <h2 className="ip-problem-title">{tr(q.title)}</h2><div className="ip-prompt"><Paragraphs text={tr(q.prompt)}/></div>
   {q.kind==='choice'?<form className="ip-choice-form" onSubmit={e=>{e.preventDefault();if(selected!==null&&!checked){setChecked(true);onGrade(selected===q.answer)}}}>
    <fieldset disabled={checked}><legend className="ip-sr">{t('选择一个答案','Select one answer')}</legend>{q.options!.map((option,i)=><label className={`ip-option ${selected===i?'selected':''} ${checked&&i===q.answer?'correct':''} ${checked&&selected===i&&i!==q.answer?'incorrect':''}`} key={i}><input type="radio" name={`answer-${q.id}`} checked={selected===i} onChange={()=>setSelected(i)}/><span className="ip-option-letter">{String.fromCharCode(65+i)}</span><span>{tr(option)}</span>{checked&&i===q.answer&&<b>✓</b>}</label>)}</fieldset>
    <div className="ip-answer-actions"><button className="ip-primary" type="submit" disabled={selected===null||checked}>{checked?t('已提交','Submitted'):t('提交答案','Check answer')} ↗</button>{checked&&<button type="button" onClick={()=>{setSelected(null);setChecked(false);setShowSolution(false)}}>{t('再做一次','Try again')}</button>}<button type="button" onClick={reveal}>{showSolution?t('收起解析','Hide solution'):t('查看解析','View solution')}</button></div>
    {checked&&<div className={`ip-feedback ${selected===q.answer?'success':'failure'}`} role="status"><strong>{selected===q.answer?t('回答正确。','Correct.'):t('还差一点，看看这一步。','Not quite. Let’s unpack it.')}</strong><span>{t('正确答案','Correct answer')} {String.fromCharCode(65+q.answer!)} · {tr(q.options![q.answer!])}</span></div>}
   </form>:<div className="ip-examples"><h3>{t('示例与接口','Examples & interface')}</h3>{q.tests!.slice(0,2).map((c,i)=><div className="ip-example" key={i}><span>{t('示例','Example')} {i+1}</span><pre>{c.expression}</pre><p>{t('预期','Expected')}: <code>{c.raises??JSON.stringify(c.expected)}</code></p></div>)}<div className="ip-code-contract"><span>{q.tests!.length} {t('个公开测试 · 浮点容差 1e-6','public tests · float tolerance 1e-6')}</span><p>{t('先运行两个示例，再提交全部测试。使用 Python 标准库；不提供 CUDA/PyTorch。这里是本地自测，不是防作弊考试系统。','Run two examples first, then submit all tests. Use Python’s standard library; CUDA/PyTorch are not provided. This is local practice, not an anti-cheating examination system.')}</p></div><button onClick={reveal}>{showSolution?t('收起参考解','Hide reference solution'):t('查看参考解与解析','Reference solution & explanation')}</button></div>}
   {(showSolution||checked)&&<section className="ip-solution"><div className="ip-section-title"><span>EXPLANATION</span><h3>{t('不仅知道选什么，还知道为什么。','Know why, not just what.')}</h3></div>{q.kind==='choice'&&!checked&&<p className="ip-answer-reveal">{t('正确答案','Correct answer')}: {String.fromCharCode(65+q.answer!)}</p>}<Paragraphs text={tr(q.explanation)}/>{q.solution&&<><h4>{t('参考实现','Reference implementation')}</h4><pre className="ip-reference"><code>{q.solution}</code></pre><h4>{t('复杂度','Complexity')}</h4><p>{tr(q.complexity!)}</p></>}<div className="ip-followup"><span>{t('面试官可能追问','INTERVIEW FOLLOW-UP')}</span><p>{tr(q.followup)}</p></div><a className="ip-source" href={topics.find(x=>x.id===q.topic)!.source} target="_blank" rel="noreferrer">{t('延伸阅读：原论文 / 官方文档','Further reading: primary paper / official documentation')} ↗</a></section>}
   <div className="ip-problem-foot"><span>{record?.attempts??0} {t('次提交','submissions')}{record?.viewed?` · ${t('已查看参考解','Reference viewed')}`:''}</span><span>{t('原创练习题 · 非公司真题','Original practice · not company interview leaks')}</span></div>
  </div>
  {q.kind==='code'&&<div className="ip-coding-pane"><div className="ip-editor-bar"><span><i/> Python 3 <small>· WebAssembly</small></span><button disabled={phase!=='idle'} onClick={()=>{if(window.confirm(t('恢复初始代码？会覆盖这道题的本地草稿。','Restore starter code? This overwrites this problem’s local draft.'))){edit(q.starter!);setResult(null)}}}>{t('重置','Reset')} ↺</button></div><div className="ip-editor-wrap"><div className="ip-line-numbers" aria-hidden="true">{code.split('\n').map((_,i)=><div key={i}>{i+1}</div>)}</div><textarea ref={editor} aria-label={t('Python 代码编辑器','Python code editor')} wrap="off" spellCheck={false} autoCapitalize="off" autoCorrect="off" value={code} maxLength={50000} onChange={e=>edit(e.target.value)} onScroll={e=>{const gutter=e.currentTarget.previousElementSibling;if(gutter)gutter.scrollTop=e.currentTarget.scrollTop}} onKeyDown={e=>{if(e.key==='Enter'&&(e.metaKey||e.ctrlKey)){e.preventDefault();run(true)}else if(e.key===']'&&(e.ctrlKey||e.metaKey)){e.preventDefault();const start=e.currentTarget.selectionStart,end=e.currentTarget.selectionEnd;edit(code.slice(0,start)+'    '+code.slice(end));requestAnimationFrame(()=>editor.current?.setSelectionRange(start+4,start+4))}}}/></div><div className="ip-editor-hint"><span>{draftError?t('草稿无法保存','Draft storage unavailable'):t('草稿自动保存在本机','Draft autosaved locally')}</span><span>Ctrl / ⌘ + ]: indent · Enter: run</span></div><div className="ip-run-bar">{phase==='idle'?<><button onClick={()=>run(false)}>{t('运行示例','Run examples')} ▷</button><button className="ip-primary" onClick={()=>run(true)}>{t('提交验证','Submit')} ↗</button></>:<><span role="status">{phase==='loading'?t('首次加载 Python…','Loading Python…'):t('正在执行测试…','Running tests…')}</span><button onClick={()=>stop(t('已停止，尚未判定本次提交。','Stopped; this submission was not graded.'))}>{t('停止','Stop')} ■</button></>}</div><section className="ip-results" aria-label={t('测试结果','Test results')} aria-live="polite"><div className="ip-results-title"><h3>{t('测试结果','Test results')}</h3><span>{result?.cases?.filter(c=>c.passed).length??0} / {result?.cases?.length??q.tests!.length}</span></div>{notice&&<p className="ip-run-notice" role="status">{notice}</p>}{!result&&!notice&&<p className="ip-results-empty">{t('运行你的代码，查看实际返回值与预期结果。Python 只在首次运行时加载。','Run your code to compare actual and expected values. Python loads only on the first run.')}</p>}{result&&<><div className={`ip-result-verdict ${stale?'stale':passed?'success':'failure'}`}>{stale?t('代码已修改，下方结果已过期。','Code changed; these results are stale.'):result.error?t('执行错误','Execution error'):passed?(result.submitted?t('通过全部测试','All tests passed'):t('示例通过 · 尚未提交','Examples passed · not submitted')):t('部分测试未通过','Some tests failed')}</div>{result.error&&<pre className="ip-traceback">{result.error}</pre>}{result.cases?.map((c,i)=><details className={`ip-case ${c.passed?'passed':'failed'}`} key={i} open={!c.passed}><summary><span>{c.passed?'✓':'×'} {t('测试','Case')} {i+1}</span><code>{q.tests![i].expression}</code></summary><p>{t('预期','Expected')}: <code>{q.tests![i].raises??JSON.stringify(q.tests![i].expected)}</code></p><p>{t('实际','Actual')}: <code>{c.actual}</code></p></details>)}{result.output&&<details className="ip-stdout" open><summary>stdout / stderr</summary><pre>{result.output}</pre></details>}</>}</section></div>}
 </div>
}
