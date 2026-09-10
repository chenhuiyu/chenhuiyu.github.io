export function softmax(xs:number[], temperature=1){const top=Math.max(...xs);const e=xs.map(x=>Math.exp((x-top)/temperature));const sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum)}
export const toyVectors=[[1,0],[0,1],[1,1],[-1,0.5]];
export function attention(temperature=1,causal=true){return toyVectors.map((q,i)=>softmax(toyVectors.map((k,j)=>causal&&j>i?-Infinity:(q[0]*k[0]+q[1]*k[1])/Math.sqrt(2)),temperature))}
export function kvBytes(batch:number,tokens:number,layers:number,heads:number,dim:number,bytes:number){return 2*batch*tokens*layers*heads*dim*bytes}
export function hstuWeights(bias:number){return toyVectors.map((q,i)=>toyVectors.map((k,j)=>{if(j>i)return 0;const s=q[0]*k[0]+q[1]*k[1]-bias*(i-j);return s/(1+Math.exp(-s))/toyVectors.length}))}
export function maskedMse(target:number[],pred:number[],mask:boolean[]){let n=0,sum=0;target.forEach((v,i)=>{if(mask[i]){sum+=(v-pred[i])**2;n++}});return n?sum/n:0}
export function ringAllReduceBytes(bytes:number,ranks:number){return 2*(ranks-1)/ranks*bytes}
