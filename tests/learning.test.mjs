import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {attention,softmax,kvBytes,hstuWeights,maskedMse,ringAllReduceBytes} from '../lib/learning/math.ts';
test('causal attention forbids future access and preserves row normalization',()=>{
 for(const t of [.1,1,3])attention(t,true).forEach((row,i)=>{assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-12);row.forEach((v,j)=>{if(j>i)assert.equal(v,0)})});
 assert.deepEqual(attention(.1,true)[0],[1,0,0,0]);
 assert.ok(attention(1,false)[0][1]>0);
 const a=softmax([1,2,3]),b=softmax([101,102,103]);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-12));
});
test('memory estimates count K and V and use GiB correctly',()=>{
 assert.equal(kvBytes(4,4096,32,8,128,2),2*2**30);
 assert.equal(kvBytes(4,4096,32,1,128,2),2**28);
 assert.equal(ringAllReduceBytes(1024,1),0);
 assert.equal(ringAllReduceBytes(1024,8),1792);
});
test('HSTU teaching aggregation is signed, causal, and not row softmax',()=>{
 const rows=hstuWeights(.5);assert.ok(rows.some(row=>row.some(v=>v<0)));
 rows.forEach((row,i)=>row.forEach((v,j)=>{if(j>i)assert.equal(v,0)}));
 assert.notEqual(rows[0].reduce((a,b)=>a+b,0),1);
});
test('masked reconstruction ignores visible errors',()=>{
 assert.equal(maskedMse([1,2],[0,999],[true,false]),1);
 assert.equal(maskedMse([1],[0],[false]),0);
});
test('every curriculum chapter has complete bilingual text and local diagrams',async()=>{
 const curriculum=JSON.parse(await readFile(new URL('../content/learning-curriculum.json',import.meta.url),'utf8'));
 const posts=JSON.parse(await readFile(new URL('../content/authored-posts.json',import.meta.url),'utf8'));
 assert.equal(curriculum.length,25);
 for(const lesson of curriculum){
  const pair=posts.filter(p=>p.pairKey===lesson.key);assert.equal(pair.length,2,lesson.key);
  assert.deepEqual(new Set(pair.map(p=>p.language)),new Set(['en','zh-CN']));
  for(const p of pair){assert.equal(p.series,lesson.track);assert.equal(p.seriesOrder,lesson.order);assert.ok(p.toc.length>=4);assert.ok(!p.content.includes('katex-error'),p.slug);assert.ok(p.alternateSlug);assert.ok(p.content.includes('<details>'));
   for(const match of p.content.matchAll(/<img[^>]+src="([^"]+)"/g))await readFile(new URL('../public'+match[1],import.meta.url));
  }
 }
});
