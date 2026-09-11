/** Exact inference for the accompanying trained pre-norm causal Transformer.
 * Row-major PyTorch weights, population-variance LayerNorm, tanh GELU.
 * All visualizations consume this trace; there is no simulated attention.
 */
export type Matrix = number[][];
export type ModelData = {
  meta: {name:string; dimension:number; heads:number; layers:number; context:number; vocab:string[]; parameters:number; validation:Record<string,number>; steps:number; seed:number};
  weights: Record<string, number[] | Matrix>;
};
export type Trace = {ids:number[]; states:Matrix[]; attention:Matrix[][]; probabilities:number[]};
export function softmax(a:number[]):number[]{const m=Math.max(...a),e=a.map(v=>Math.exp(v-m)),s=e.reduce((x,y)=>x+y,0);return e.map(v=>v/s)}
export function tokenize(text:string,model:ModelData):number[]{
  const words=text.trim().replace(/([=;])/g,' $1 ').split(/\s+/).filter(Boolean);
  if(!words.length)throw new Error('empty');
  const ids=words.map(w=>model.meta.vocab.indexOf(w.toLowerCase()));
  if(ids.some(x=>x<1))throw new Error('vocabulary');
  if(ids.length+1>model.meta.context)throw new Error('length');
  return [0,...ids];
}
export function forward(model:ModelData,ids:number[],ablate?:[number,number]):Trace{
  const w=model.weights,d=model.meta.dimension,heads=model.meta.heads,dh=d/heads;
  const mat=(key:string)=>w[key] as Matrix,vec=(key:string)=>w[key] as number[];
  const linear=(x:number[],key:string)=>mat(key+'.weight').map((row,i)=>row.reduce((s,v,j)=>s+v*x[j],(w[key+'.bias'] as number[]|undefined)?.[i]??0));
  const norm=(x:number[],key:string)=>{const mean=x.reduce((a,b)=>a+b,0)/d,v=x.reduce((s,z)=>s+(z-mean)**2,0)/d;return x.map((z,i)=>(z-mean)/Math.sqrt(v+1e-5)*vec(key+'.weight')[i]+vec(key+'.bias')[i])};
  let x=ids.map((id,t)=>mat('token.weight')[id].map((v,i)=>v+mat('position.weight')[t][i]));
  const states=[x],attention:Matrix[][]=[];
  for(let layer=0;layer<model.meta.layers;layer++){
    const p=`blocks.${layer}`,qkv=x.map(v=>linear(norm(v,p+'.ln1'),p+'.qkv'));
    const att:Matrix[]=Array.from({length:heads},()=>[]);
    const joined=x.map((_,t)=>{
      const out:number[]=[];
      for(let h=0;h<heads;h++){
        const offset=h*dh;
        const scores=ids.map((_,s)=>s>t?-Infinity:Array.from({length:dh},(_,i)=>qkv[t][offset+i]*qkv[s][d+offset+i]).reduce((a,b)=>a+b,0)/Math.sqrt(dh));
        const a=softmax(scores);att[h].push(a);
        for(let i=0;i<dh;i++)out.push(ablate?.[0]===layer&&ablate[1]===h?0:a.reduce((s,v,j)=>s+v*qkv[j][2*d+offset+i],0));
      }return out;
    });
    x=x.map((v,t)=>{const delta=linear(joined[t],p+'.proj');return v.map((z,i)=>z+delta[i])});
    x=x.map(v=>{const hidden=linear(norm(v,p+'.ln2'),p+'.fc1').map(z=>.5*z*(1+Math.tanh(Math.sqrt(2/Math.PI)*(z+.044715*z**3))));const delta=linear(hidden,p+'.fc2');return v.map((z,i)=>z+delta[i])});
    states.push(x);attention.push(att);
  }
  return {ids,states,attention,probabilities:softmax(linear(norm(x[x.length-1],'ln'),'output'))};
}
