import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import ts from 'typescript';
import {Worker} from 'node:worker_threads';
const cache=new Map();
function load(name){
 if(cache.has(name))return cache.get(name);
 const filename=new URL(`../lib/interview/${name}.ts`,import.meta.url);
 const source=ts.transpileModule(readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText;
 const mod={exports:{}};cache.set(name,mod.exports);
 new Function('require','module','exports',source)(id=>id.startsWith('./')?load(id.slice(2)):createRequire(import.meta.url)(id),mod,mod.exports);
 return mod.exports;
}
const {questions,topics}=load('bank');
test('60 bilingual problems, exactly four choices and one runnable problem per topic',()=>{
 assert.equal(questions.length,60);assert.equal(new Set(questions.map(q=>q.id)).size,60);
 for(const topic of topics){const qs=questions.filter(q=>q.topic===topic.id);assert.equal(qs.length,5);assert.equal(qs.filter(q=>q.kind==='code').length,1)}
 for(const q of questions){for(const field of ['title','prompt','explanation','followup']){assert.ok(q[field].zh.length>5);assert.ok(q[field].en.length>5)}
 assert.ok(['easy','medium','hard'].includes(q.difficulty));
 if(q.kind==='choice'){assert.equal(q.options.length,4);assert.ok(q.answer>=0&&q.answer<4)}else{assert.ok(q.tests.length>=5);assert.ok(q.solution);assert.ok(q.starter)}}
});
test('Actual browser Python runner: all reference solutions pass; all starters fail; exceptions recover', {timeout:180000},async()=>{
 const url=new URL('../public/workers/interview-worker.mjs',import.meta.url).href;
 const w=new Worker(`const {parentPort}=require('node:worker_threads');global.self={addEventListener:(_,fn)=>parentPort.on('message',data=>fn({data})),postMessage:data=>parentPort.postMessage(data)};import(${JSON.stringify(url)}).then(()=>parentPort.postMessage({type:'ready'}));`,{eval:true});
 await new Promise((resolve,reject)=>{w.once('message',resolve);w.once('error',reject)});
 let id=0;
 const run=(code,tests)=>new Promise((resolve,reject)=>{const runId=++id;const timer=setTimeout(()=>reject(Error('timeout')),90000);const listener=data=>{if(data.runId===runId&&['result','error'].includes(data.type)){clearTimeout(timer);w.off('message',listener);resolve(data)}};w.on('message',listener);w.postMessage({runId,code,tests})});
 try{
  for(const q of questions.filter(q=>q.kind==='code')){const r=await run(q.solution,q.tests);assert.equal(r.type,'result',q.id);assert.ok(!r.error,`${q.id}: ${r.error}`);assert.equal(r.cases.length,q.tests.length);assert.ok(r.cases.every(c=>c.passed),`${q.id}: ${JSON.stringify(r.cases)}`);const bad=await run(q.starter,q.tests);assert.ok(bad.error||bad.cases.some(c=>!c.passed),q.id)}
  assert.match((await run('def invalid(:',[])).error,/SyntaxError/);
  assert.equal((await run('x=1',[{expression:'x',expected:1}])).cases[0].passed,true);
  assert.equal((await run('',[{expression:'x',raises:'NameError'}])).cases[0].passed,true);
  assert.equal((await run('',[{expression:'True',expected:1}])).cases[0].passed,false);
 }finally{await w.terminate()}
});
test('Corrupt local progress cannot inject unknown problems or invalid attempts',()=>{
 const {readProgress}=load('progress');assert.deepEqual(readProgress('broken',['001']),{});
 const p=readProgress(JSON.stringify({'001':{status:'solved',attempts:2},'999':{status:'solved',attempts:1}}),['001']);assert.equal(p['001'].attempts,2);assert.equal(p['999'],undefined);
});
