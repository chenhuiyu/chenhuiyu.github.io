import test from 'node:test';
import assert from 'node:assert/strict';
import {Worker} from 'node:worker_threads';
import {readFileSync} from 'node:fs';
const workerURL=new URL('../public/workers/notebook-worker.mjs',import.meta.url).href;
function kernel(){
 const w=new Worker(`const {parentPort}=require('node:worker_threads');global.self={addEventListener:(_,fn)=>parentPort.on('message',data=>fn({data})),postMessage:data=>parentPort.postMessage(data)};import(${JSON.stringify(workerURL)}).then(()=>parentPort.postMessage({type:'ready'}));`,{eval:true});
 let id=0;
 const ready=new Promise((resolve,reject)=>{w.once('error',reject);const listener=data=>{if(data.type==='ready'){w.off('message',listener);resolve()}};w.on('message',listener)});
 return {w,async run(code){await ready;const runId=++id;return new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(Error('worker timeout')),90000);const listener=data=>{if(data.runId===runId&&['result','error'].includes(data.type)){clearTimeout(timeout);w.off('message',listener);resolve(data)}};w.on('message',listener);w.postMessage({code,runId})})}};
}
test('Actual Pyodide worker runs all browser notebooks and preserves cell state', {timeout:180000},async()=>{
 for(const name of ['llm-model-io','infra-benchmark','multimodal-inspection','hstu-from-scratch']){
  const en=JSON.parse(readFileSync(new URL(`../public/notebooks/${name}-browser-en.ipynb`,import.meta.url)));
  const zh=JSON.parse(readFileSync(new URL(`../public/notebooks/${name}-browser-zh-CN.ipynb`,import.meta.url)));
  const code=n=>n.cells.filter(c=>c.cell_type==='code').map(c=>c.source.join(''));
  assert.deepEqual(code(en),code(zh),'Bilingual cells must execute identical experiments');
  const k=kernel();let plots=0;
  try{
   for(const c of en.cells.filter(c=>c.cell_type==='code'&&!c.metadata?.tags?.includes('requires-model')).map(c=>c.source.join(''))){const r=await k.run(c);assert.equal(r.type,'result',`${name}: ${r.output}`);for(const p of r.plots??[]){assert.equal(p.x.length,p.y.length);assert.ok(p.y.every(Number.isFinite));plots++}}
   assert.ok(plots>0);
   if(name==='llm-model-io'||name==='hstu-from-scratch')assert.equal((await k.run('assert losses[-1] < losses[0]')).type,'result');
   const error=await k.run('print("before error")\nraise ValueError("expected failure")');assert.equal(error.type,'error');assert.match(error.output,/before error/);
   assert.equal((await k.run('assert 2+2 == 4\nprint("recovered")')).type,'result');
  }finally{await k.w.terminate()}
 }
});
test('Kernel restart clears variables', {timeout:120000},async()=>{
 const a=kernel();try{assert.equal((await a.run('sentinel=42')).type,'result')}finally{await a.w.terminate()}
 const b=kernel();try{assert.equal((await b.run('assert "sentinel" not in globals()')).type,'result')}finally{await b.w.terminate()}
});
test('Python model bridge forwards inputs, awaits results and surfaces failures', {timeout:120000},async()=>{
 const k=kernel();let requests=0;
 k.w.on('message',data=>{
  if(data.type!=='model-request')return;
  requests++;
  assert.deepEqual(data.texts,['first sentence','second sentence']);
  k.w.postMessage(requests===1?{type:'model-response',requestId:data.requestId,result:{dims:[2,384],cosine:0.5}}:{type:'model-response',requestId:data.requestId,error:'download failed'});
 });
 try{
  const result=await k.run('import json\nfrom browser_models import embed\nr=json.loads(await embed(["first sentence","second sentence"]))\nassert r["dims"] == [2,384]\nprint(r["cosine"])');
  assert.equal(result.type,'result',result.output);assert.match(result.output,/0.5/);
  const error=await k.run('await embed(["first sentence","second sentence"])');
  assert.equal(error.type,'error');assert.match(error.output,/download failed/);
  const invalid=await k.run('await embed(["only one"])');assert.equal(invalid.type,'error');assert.equal(requests,2);
 }finally{await k.w.terminate()}
});
