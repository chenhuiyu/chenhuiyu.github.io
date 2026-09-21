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
test('1,000 bilingual problems, stable IDs and declared coverage',()=>{
 assert.equal(questions.length,1000);assert.equal(new Set(questions.map(q=>q.id)).size,1000);
 assert.equal(new Set(questions.map(q=>q.title.en)).size,1000);
 assert.equal(new Set(questions.map(q=>q.prompt.en)).size,1000);
 assert.deepEqual(questions.map(q=>q.id),Array.from({length:1000},(_,i)=>String(i+1).padStart(3,'0')));
 assert.equal(questions.filter(q=>q.kind==='choice').length,232);assert.equal(questions.filter(q=>q.kind==='code').length,32);assert.equal(questions.filter(q=>q.kind==='written').length,736);
 const {curriculum}=load('curriculum');assert.equal(curriculum.length,184);
 assert.equal(new Set(curriculum.map(u=>u.name.en)).size,184);
 for(const topic of topics){const qs=questions.filter(q=>q.topic===topic.id);assert.ok(qs.length>=60);assert.ok(qs.some(q=>q.kind==='code'))}
 for(const q of questions){for(const field of ['title','prompt','explanation','followup']){assert.ok(q[field].zh.length>(field==='title'?1:5),`${q.id} ${field} zh`);assert.ok(q[field].en.length>5,`${q.id} ${field} en`)}
 assert.ok(['easy','medium','hard'].includes(q.difficulty));
 if(q.kind==='choice'){assert.ok([2,4].includes(q.options.length));assert.ok(q.answer>=0&&q.answer<q.options.length)}else if(q.kind==='code'){assert.ok(q.tests.length>=5);assert.ok(q.solution);assert.ok(q.starter)}else{assert.ok(q.rubric.length>=2);assert.ok(q.rubric.every(p=>p.zh&&p.en));assert.ok(q.sourceIds.length)}
 for(const id of q.sourceIds??[])assert.ok(load('sources').sources[id],`Unknown source ${id}`)}
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
