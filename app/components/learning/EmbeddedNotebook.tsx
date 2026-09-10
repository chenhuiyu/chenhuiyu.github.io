'use client';
import {useEffect,useRef,useState} from 'react';
import {Marked} from 'marked';
import type {Locale} from '@/lib/learning/tracks';
type Plot={x:number[];y:number[];title:string;xlabel:string;ylabel:string};
type Output={output_type:string;text?:string[]|string;data?:Record<string,string[]|string>;ename?:string;evalue?:string};
type Cell={metadata?:{tags?:string[]};cell_type:string;source:string[]|string;outputs?:Output[];execution_count?:number|null};
type Notebook={cells:Cell[]};
type Result={output:string;plots?:Plot[];error?:boolean;count?:number;stale?:boolean};
const string=(value:string[]|string|undefined)=>Array.isArray(value)?value.join(''):value??'';
// Notebook markdown is repository-authored. Raw HTML is escaped, never executed.
const notebookMarkdown=new Marked({renderer:{html({text}){return text.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}}});
function NotebookText({source}:{source:string}) {return <div className="nb-markdown" dangerouslySetInnerHTML={{__html:notebookMarkdown.parse(source,{async:false}) as string}}/>}

function Chart({plot}:{plot:Plot}){
 if(!plot.x.length||plot.x.length!==plot.y.length||![...plot.x,...plot.y].every(Number.isFinite))return null;
 const minX=Math.min(...plot.x),maxX=Math.max(...plot.x),minY=Math.min(...plot.y),maxY=Math.max(...plot.y);
 const px=(x:number)=>64+(x-minX)/(maxX-minX||1)*540,py=(y:number)=>210-(y-minY)/(maxY-minY||1)*170;
 return <figure className="nb-plot"><figcaption>{plot.title}</figcaption><svg viewBox="0 0 650 265" role="img" aria-label={`${plot.title}; ${plot.xlabel}; ${plot.ylabel}`}><path d="M64 30V210H610" fill="none" stroke="currentColor"/><polyline points={plot.x.map((x,i)=>`${px(x)},${py(plot.y[i])}`).join(' ')} fill="none" stroke="#b35029" strokeWidth="2.5"/><text x="60" y="230">{minX.toPrecision(3)}</text><text x="560" y="230">{maxX.toPrecision(3)}</text><text x="2" y="44">{maxY.toPrecision(3)}</text><text x="2" y="208">{minY.toPrecision(3)}</text><text x="230" y="258">{plot.xlabel}</text><text x="65" y="20">{plot.ylabel}</text></svg><details><summary>Data / 数据</summary><pre>{plot.x.map((x,i)=>`${x}\t${plot.y[i]}`).join('\n')}</pre></details></figure>
}
export function EmbeddedNotebook({name,locale}:{name:string;locale:Locale}){
 const zh=locale==='zh-CN';
 const [nb,setNb]=useState<Notebook|null>(null),[original,setOriginal]=useState<Notebook|null>(null),[mode,setMode]=useState<'browser'|'original'>('browser'),[codes,setCodes]=useState<Record<number,string>>({}),[results,setResults]=useState<Record<number,Result>>({}),[busy,setBusy]=useState<number|null>(null),[status,setStatus]=useState(''),[loadError,setLoadError]=useState(''),[attempt,setAttempt]=useState(0);
 const modelWorker=useRef<Worker|null>(null);
 const worker=useRef<Worker|null>(null),timer=useRef<ReturnType<typeof setTimeout>|null>(null),pending=useRef<((ok:boolean)=>void)|null>(null),sequence=useRef(0),execution=useRef(0),epoch=useRef(0),locked=useRef(false);
 function shutdown(){epoch.current++;modelWorker.current?.terminate();modelWorker.current=null;worker.current?.terminate();worker.current=null;if(timer.current)clearTimeout(timer.current);pending.current?.(false);pending.current=null;locked.current=false;}
 useEffect(()=>{
  const controller=new AbortController();
  async function fetchNotebook(path:string){const r=await fetch(path,{signal:controller.signal});if(!r.ok)throw Error(`HTTP ${r.status}`);const n=await r.json();if(!Array.isArray(n.cells))throw Error('Invalid notebook');return n as Notebook;}
  fetchNotebook(`/notebooks/${name}-browser-${locale}.ipynb`).then(n=>{setNb(n);setCodes(Object.fromEntries(n.cells.map((c,i)=>[i,string(c.source)])))}).catch(e=>{if(!controller.signal.aborted)setLoadError(String(e))});
  return()=>{controller.abort();shutdown()};
 },[name,locale,attempt]);
 async function showOriginal(){setMode('original');if(original)return;setStatus(zh?'正在加载原始 Notebook…':'Loading original notebook…');try{const r=await fetch(`/notebooks/${name}-${locale}.ipynb`);if(!r.ok)throw Error(`HTTP ${r.status}`);setOriginal(await r.json())}catch(e){setStatus(String(e))}}
 function stop(){shutdown();setBusy(null);setStatus(zh?'内核已停止；变量已清空，请从第一格重新运行。':'Kernel stopped; variables cleared. Run again from the first cell.');setResults(prev=>Object.fromEntries(Object.entries(prev).map(([k,v])=>[k,{...v,stale:true}])))}
 async function execute(index:number):Promise<boolean>{
  setBusy(index);setStatus(zh?'正在执行…':'Running…');const id=++sequence.current;
  return new Promise(resolve=>{
   pending.current=resolve;
   const finish=(ok:boolean,r:Result)=>{if(timer.current)clearTimeout(timer.current);setResults(prev=>({...prev,[index]:{...r,count:++execution.current}}));pending.current=null;resolve(ok)};
   try{
    if(!worker.current)worker.current=new Worker('/workers/notebook-worker.mjs',{type:'module'});
    worker.current.onmessage=({data})=>{
      if(data.type==='model-request'){
        setStatus(zh?'下载真实模型并运行（约 23 MB 权重及运行库）…':'Downloading and running a real model (~23 MB weights plus runtime)…');
        try {
          modelWorker.current??=new Worker('/workers/model-inspector.mjs',{type:'module'});
          modelWorker.current.onmessage=({data:reply})=>{
            if(reply.runId!==data.requestId)return;
            if(reply.type==='progress')setStatus(`${reply.status}: ${reply.file} ${reply.progress===null?'':Math.round(reply.progress)+'%'}`);
            if(reply.type==='result'||reply.type==='error')worker.current?.postMessage({type:'model-response',requestId:data.requestId,result:reply,error:reply.type==='error'?reply.message:undefined});
          };
          modelWorker.current.onerror=e=>{worker.current?.postMessage({type:'model-response',requestId:data.requestId,error:e.message});modelWorker.current?.terminate();modelWorker.current=null};
          modelWorker.current.postMessage({texts:data.texts,runId:data.requestId});
        } catch(e){worker.current?.postMessage({type:'model-response',requestId:data.requestId,error:String(e)})}
        return;
      }
      if(data.runId!==id)return;if(data.type==='loading'){setStatus(zh?'首次加载 Python，随后各单元共用同一个内核…':'Loading Python; subsequent cells share this kernel…');return}finish(data.type==='result',{output:data.output,plots:data.plots,error:data.type==='error'})};
    worker.current.onerror=e=>{finish(false,{output:e.message,error:true});shutdown()};
    worker.current.postMessage({runId:id,code:codes[index]});
    timer.current=setTimeout(()=>{finish(false,{output:zh?'运行超时，内核已终止。普通单元限 90 秒，模型下载单元限 3 分钟。请检查网络或减小计算量后从头运行。':'Execution timed out. Kernel terminated; limit is 90 seconds, or 3 minutes for model downloads. Check connectivity or reduce the work and restart.',error:true});shutdown()},nb?.cells[index].metadata?.tags?.includes('requires-model')?180000:90000);
   }catch(e){finish(false,{output:String(e),error:true});shutdown()}
  });
 }
 async function run(indices:number[]){if(locked.current)return;locked.current=true;const current=epoch.current;let ok=true;for(const i of indices){if(current!==epoch.current){ok=false;break}ok=await execute(i);if(!ok)break}locked.current=false;setBusy(null);if(current===epoch.current)setStatus(ok?(zh?'执行完成。输出来自本次实际运行。':'Finished. Outputs are from this execution.'):(zh?'执行出错，已停止后续单元。请检查代码及执行顺序。':'Execution failed; remaining cells stopped. Check code and cell order.'))}
 function download(){if(!nb)return;const copy={...nb,cells:nb.cells.map((c,i)=>c.cell_type==='code'?{...c,source:codes[i].split(/(?<=\n)/),execution_count:null,outputs:[]}:c)};const url=URL.createObjectURL(new Blob([JSON.stringify(copy,null,2)],{type:'application/x-ipynb+json'}));const a=document.createElement('a');a.href=url;a.download=`${name}-edited-${locale}.ipynb`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
 return <section className="embedded-notebook" id="embedded-notebook"><p className="section-kicker">NOTEBOOK · LIVE PYTHON</p><h2>{zh?'在文章里运行 Notebook':'Run the notebook inside this article'}</h2><p>{zh?'逐格改代码、运行，或者一键运行全部。变量在单元之间保留；离开页面或重启内核会清空变量。编辑上游单元后，请重新运行后续单元。':'Edit and run cells, or run all. Variables persist between cells until you leave the page or restart the kernel. After editing an upstream cell, rerun the cells below it.'}</p><div className="lab-actions"><button aria-pressed={mode==='browser'} disabled={busy!==null} onClick={()=>setMode('browser')}>{zh?'浏览器可运行版':'Run in browser'}</button><button aria-pressed={mode==='original'} disabled={busy!==null} onClick={showOriginal}>{zh?'完整 PyTorch 实验与已有输出':'Full PyTorch experiment & saved outputs'}</button></div>
 {mode==='browser'?<><p className="nb-note">{zh?'此版用真实 Python 实现可检查的小型计算与训练，不需要 GPU。PyTorch/CUDA 不在此浏览器内核中；完整预训练模型 Notebook 见另一个标签，大模型基础的 Notebook 还提供单独加载真实 MiniLM 的可编辑单元。':'This edition runs inspectable small computations and training in real Python without a GPU. This kernel does not include PyTorch/CUDA. The other tab contains the full pretrained-model notebook; the foundations notebook also has a separate editable cell for real MiniLM inference.'}</p><div className="nb-toolbar lab-actions"><button disabled={!nb||busy!==null} onClick={()=>run(nb!.cells.flatMap((c,i)=>c.cell_type==='code'&&!c.metadata?.tags?.includes('requires-model')?[i]:[]))}>{zh?'运行全部基础单元':'Run all core cells'}</button><button disabled={busy===null} onClick={stop}>{zh?'停止':'Stop'}</button><button onClick={()=>{stop();execution.current=0;setResults({})}}>{zh?'重启内核':'Restart kernel'}</button><button disabled={!nb||busy!==null} onClick={download}>{zh?'下载我的代码':'Download my code'}</button></div><p role="status" aria-live="polite">{status||(zh?'内核尚未加载；点击运行开始。':'Kernel not loaded; click Run to start.')}</p>{loadError?<p role="alert">{loadError} <button onClick={()=>{setLoadError('');setAttempt(a=>a+1)}}>{zh?'重试':'Retry'}</button></p>:!nb?<p>{zh?'正在加载 Notebook…':'Loading notebook…'}</p>:nb.cells.map((c,i)=>c.cell_type==='markdown'?<NotebookText key={i} source={string(c.source)}/>:<div className="nb-cell" key={i}><div className="nb-cell-header"><span>In [{busy===i?'*':results[i]?.count??' '}]</span><button disabled={busy!==null} onClick={()=>run([i])}>{c.metadata?.tags?.includes('requires-model')?(zh?'下载模型并运行此格':'Download model & run cell'):(zh?'运行此格':'Run cell')}</button></div><label className="sr-only" htmlFor={`nb-code-${i}`}>{zh?'Python 单元':'Python cell'} {i}</label><textarea id={`nb-code-${i}`} className="python-editor" rows={Math.min(24,Math.max(4,(codes[i]??'').split('\n').length))} spellCheck={false} value={codes[i]??''} disabled={busy!==null} onChange={e=>{setCodes(prev=>({...prev,[i]:e.target.value}));setResults(prev=>Object.fromEntries(Object.entries(prev).map(([k,v])=>[k,Number(k)>=i?{...v,stale:true}:v])))}}/>{results[i]&&<div className={results[i].error?'nb-output nb-error':'nb-output'}>{results[i].stale&&<p>{zh?'代码或内核已变化；以下是上次输出，请重新运行。':'Code or kernel changed. Previous output below; rerun to update.'}</p>}<pre>{results[i].output||(zh?'运行完成，无文本输出。':'Finished without text output.')}</pre>{results[i].plots?.map((p,j)=><Chart key={j} plot={p}/>)}</div>}</div>)}</>:<><p className="nb-note">{zh?'以下为完整原始 Notebook 与保存的 CPU 执行结果，可在文章内阅读。它依赖 PyTorch 和模型权重，不能在本站的浏览器 Python 内核运行；点击 Colab 可执行完整版本。':'The full original notebook and saved CPU results are readable below. It requires PyTorch and model weights, which this browser Python kernel cannot run. Execute the full version in Colab.'}</p><div className="lab-actions"><a target="_blank" rel="noreferrer" href={`https://colab.research.google.com/github/chenhuiyu/chenhuiyu.github.io/blob/source/public/notebooks/${name}-${locale}.ipynb`}>{zh?'运行完整 PyTorch 版 ↗':'Run full PyTorch edition ↗'}</a><a download href={`/notebooks/${name}-${locale}.ipynb`}>{zh?'下载原版':'Download original'}</a></div>{!original?<p>{status|| (zh?'加载原始 Notebook…':'Loading original notebook…')}</p>:original.cells.map((c,i)=>c.cell_type==='markdown'?<NotebookText key={i} source={string(c.source)}/>:<details className="nb-original" key={i} open={i<4}><summary>{zh?'代码与保存输出':'Code and saved output'} · In [{c.execution_count??' '}]</summary><pre>{string(c.source)}</pre>{c.outputs?.map((o,j)=><div key={j}>{o.data?.['image/png']?<img alt={zh?'Notebook 保存的执行图表':'Saved notebook execution plot'} src={'data:image/png;base64,'+string(o.data['image/png'])}/>:<pre>{string(o.text)||string(o.data?.['text/plain'])||[o.ename,o.evalue].filter(Boolean).join(': ')}</pre>}</div>)}</details>)}</>}
 </section>
}
