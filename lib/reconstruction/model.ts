export type ReconstructionModel={
 meta:{name:string;parameters:number;probeParameters:number;metrics:{testCount:number;trainCount:number;cleanProbeAccuracy:number;halfMaskedProbeAccuracy:number;halfMaskedMSE:number;meanFillMSE:number}};
 weights:Record<string,number[]|number[][]>;
 meanImage:number[];
 pca:{mean:number[];components:number[][];variance:number[]};
 cloud:{label:number;xy:number[]}[];
 gallery:{id:number;label:number;pixels:number[]}[];
};
export function pixelMask(mask:number[]){return Array.from({length:64},(_,i)=>mask[Math.floor(i/16)*4+Math.floor(i%8/2)])}
function linear(m:ReconstructionModel,x:number[],name:string){return (m.weights[name+'.weight'] as number[][]).map((row,i)=>row.reduce((s,v,j)=>s+v*x[j],(m.weights[name+'.bias'] as number[])[i]))}
export function encode(m:ReconstructionModel,pixels:number[],mask:number[]){
 const pm=pixelMask(mask),input=[...pixels.map((v,i)=>pm[i]?0:v),...mask.map(v=>1-v)];
 return linear(m,linear(m,input,'e1').map(v=>Math.max(0,v)),'e2').map(Math.tanh);
}
export function decode(m:ReconstructionModel,z:number[]){return linear(m,linear(m,z,'d1').map(v=>Math.max(0,v)),'d2').map(v=>1/(1+Math.exp(-v)))}
export function classify(m:ReconstructionModel,z:number[]){const scores=linear(m,z,'probe'),max=Math.max(...scores),exp=scores.map(v=>Math.exp(v-max)),sum=exp.reduce((s,v)=>s+v,0);return exp.map(v=>v/sum)}
export function project(m:ReconstructionModel,z:number[]){return m.pca.components.map(row=>row.reduce((s,v,i)=>s+v*(z[i]-m.pca.mean[i]),0))}
export function reconstruct(m:ReconstructionModel,pixels:number[],mask:number[]){
 const latent=encode(m,pixels,mask),prediction=decode(m,latent),pm=pixelMask(mask),count=pm.reduce((a,b)=>a+b,0),errors=prediction.map((v,i)=>(v-pixels[i])**2);
 return {latent,prediction,composite:prediction.map((v,i)=>pm[i]?v:pixels[i]),probabilities:classify(m,latent),xy:project(m,latent),errors,mse:count?errors.reduce((s,v,i)=>s+v*pm[i],0)/count:null};
}
/** Seeded ordering makes ratio changes nested: increasing hides additional patches. */
export function maskOrder(seed:number){let state=seed>>>0;const rng=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296};return Array.from({length:16},(_,i)=>({i,r:rng()})).sort((a,b)=>a.r-b.r).map(v=>v.i)}
