import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {forward,tokenize} from '../lib/observatory/model.ts';
const model=JSON.parse(await readFile(new URL('../public/models/recall/weights.json',import.meta.url),'utf8'));
const fixtures=JSON.parse(await readFile(new URL('./fixtures/recall-reference.json',import.meta.url),'utf8'));
function close(actual,expected,tolerance=2e-4){const a=actual.flat(Infinity),b=expected.flat(Infinity);assert.equal(a.length,b.length);let max=0;for(let i=0;i<a.length;i++){assert.ok(Number.isFinite(a[i]));max=Math.max(max,Math.abs(a[i]-b[i]))}assert.ok(max<tolerance,`max absolute error ${max}`)}
test('browser forward agrees with independently executed PyTorch: activations, attention, probabilities and all 8 head interventions',()=>{
 for(const f of fixtures){assert.deepEqual(tokenize(f.prompt,model),f.ids);const result=forward(model,f.ids);close(result.states,f.states);close(result.attention,f.attention);close(result.probabilities,f.probabilities);for(let i=0;i<8;i++)close(forward(model,f.ids,[Math.floor(i/4),i%4]).probabilities,f.ablations[i]);}
});
test('causal mask prevents future tokens from altering earlier activations',()=>{
 const original=tokenize(fixtures[0].prompt,model),altered=[...original];altered[7]=model.meta.vocab.indexOf('yak');const a=forward(model,original),b=forward(model,altered);
 for(let l=0;l<a.states.length;l++)close(a.states[l].slice(0,7),b.states[l].slice(0,7),1e-10);
 for(const layer of a.attention)for(const head of layer)for(let t=0;t<head.length;t++){assert.ok(Math.abs(head[t].reduce((x,y)=>x+y,0)-1)<1e-12);for(let s=t+1;s<head.length;s++)assert.equal(head[t][s],0)}
});
test('trained model responds to changed bindings and passes held-out recall evaluation',()=>{
 for(const accuracy of Object.values(model.meta.validation))assert.ok(accuracy>.95,`recall accuracy ${accuracy}`);
 for(const [i,answer] of [[0,'fox'],[1,'owl'],[2,'elk']]){const probabilities=forward(model,fixtures[i].ids).probabilities;assert.equal(model.meta.vocab[probabilities.indexOf(Math.max(...probabilities))],answer)}
});
test('malformed input is rejected and supported punctuation is tokenized consistently',()=>{
 assert.throws(()=>tokenize('hello world',model),/vocabulary/);assert.throws(()=>tokenize('',model),/empty/);assert.throws(()=>tokenize('red '.repeat(32),model),/length/);assert.deepEqual(tokenize('red=fox; blue=owl; red',model),tokenize(fixtures[0].prompt,model));
});
