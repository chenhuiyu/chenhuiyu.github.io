import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {encode,decode,classify,project,reconstruct,pixelMask,maskOrder} from '../lib/reconstruction/model.ts';
const model=JSON.parse(await readFile(new URL('../public/models/reconstruction/model.json',import.meta.url),'utf8'));
const fixtures=JSON.parse(await readFile(new URL('./fixtures/reconstruction-reference.json',import.meta.url),'utf8'));
function close(a,b,tol=2e-5){assert.equal(a.length,b.length);assert.ok(a.every(Number.isFinite));const error=Math.max(...a.map((v,i)=>Math.abs(v-b[i])));assert.ok(error<tol,`max error ${error}`)}
test('JavaScript agrees with independent PyTorch / sklearn reference across clean, masked and all-hidden inputs',()=>{
 for(const f of fixtures){const z=encode(model,f.pixels,f.mask);close(z,f.latent);close(decode(model,z),f.reconstruction);close(classify(model,z),f.probabilities);close(project(model,z),f.projection)}
});
test('hidden target pixels never leak into encoder, reconstruction or probe',()=>{
 const f=fixtures[1],pm=pixelMask(f.mask),modified=f.pixels.map((v,i)=>pm[i]?1-v:v);const a=reconstruct(model,f.pixels,f.mask),b=reconstruct(model,modified,f.mask);close(a.latent,b.latent,1e-12);close(a.prediction,b.prediction,1e-12);close(a.probabilities,b.probabilities,1e-12);
 const full=Array(16).fill(1);close(encode(model,Array(64).fill(0),full),encode(model,Array(64).fill(1),full),1e-12);
});
test('mask expansion, preserved pixels, and masked-only metrics handle boundary cases',()=>{
 const mask=Array(16).fill(0);mask[5]=1;assert.deepEqual(pixelMask(mask).flatMap((v,i)=>v?[i]:[]),[18,19,26,27]);
 const f=fixtures[0],result=reconstruct(model,f.pixels,mask);for(let i=0;i<64;i++)if(!pixelMask(mask)[i])assert.equal(result.composite[i],f.pixels[i]);
 const expected=[18,19,26,27].reduce((s,i)=>s+(result.prediction[i]-f.pixels[i])**2,0)/4;assert.ok(Math.abs(result.mse-expected)<1e-12);assert.equal(reconstruct(model,f.pixels,Array(16).fill(0)).mse,null);
 assert.deepEqual([...maskOrder(73)].sort((a,b)=>a-b),Array.from({length:16},(_,i)=>i));assert.deepEqual(maskOrder(73),maskOrder(73));
});
test('latent interpolation passes through the decoder, not a pixel crossfade',()=>{
 const z0=fixtures[0].latent,z1=fixtures[9].latent,a=decode(model,z0),b=decode(model,z1),middle=decode(model,z0.map((v,i)=>(v+z1[i])/2));assert.ok(Math.max(...middle.map((v,i)=>Math.abs(v-(a[i]+b[i])/2)))>.01);
});
test('recorded held-out reconstruction beats mean filling and probe exceeds chance',()=>{
 const m=model.meta.metrics;assert.equal(m.testCount,360);assert.equal(m.trainCount,1437);assert.ok(m.halfMaskedMSE<m.meanFillMSE);assert.ok(m.halfMaskedProbeAccuracy>.7);assert.equal(new Set(model.gallery.map(x=>x.id)).size,20);
});
