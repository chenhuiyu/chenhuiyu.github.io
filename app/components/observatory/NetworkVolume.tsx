'use client';
import {useRef,useState} from 'react';
import type {Trace} from '@/lib/observatory/model';

type Props={trace:Trace;vocab:string[];layer:number;head:number;token:number;ablated:boolean;en:boolean};
export default function NetworkVolume({trace,vocab,layer,head,token,ablated,en}:Props){
  const [angle,setAngle]=useState({yaw:-.66,pitch:.22});
  const drag=useRef<{x:number;y:number;yaw:number;pitch:number}|null>(null);
  const [zoom,setZoom]=useState(1);
  const n=trace.ids.length,dx=270/n,dy=8;
  const project=(x:number,y:number,z:number)=>{const rx=x*Math.cos(angle.yaw)+z*Math.sin(angle.yaw),rz=-x*Math.sin(angle.yaw)+z*Math.cos(angle.yaw);return [420+rx*zoom,268+(y*Math.cos(angle.pitch)-rz*Math.sin(angle.pitch))*zoom]};
  const point=(l:number,t:number,c:number)=>project((l-1)*230,(c-16)*dy,(t-n/2)*dx);
  const poly=(l:number,t:number,c:number)=>[point(l,t,c),point(l,t+1,c),point(l,t+1,c+1),point(l,t,c+1)].map(p=>p.join(',')).join(' ');
  const magnitude=Math.max(...trace.states.flat(2).map(Math.abs),.001);
  const color=(v:number)=>v>=0?`rgba(123,239,199,${.1+.86*Math.sqrt(v/magnitude)})`:`rgba(241,130,111,${.1+.86*Math.sqrt(-v/magnitude)})`;
  return <div className="ov-volume">
    <div className="ov-volume-tools"><span>{en?'RESIDUAL STREAM / 3D':'残差流 / 3D'}</span><div><button onClick={()=>setZoom(z=>Math.max(.7,z-.1))} aria-label={en?'Zoom out':'缩小'}>−</button><button onClick={()=>setZoom(z=>Math.min(1.3,z+.1))} aria-label={en?'Zoom in':'放大'}>+</button><button onClick={()=>{setAngle({yaw:-.66,pitch:.22});setZoom(1)}}>{en?'Reset view':'重置视角'}</button></div></div>
    <svg viewBox="0 0 840 540" role="img" aria-label={en?'Three residual activation matrices connected by the selected head’s attention weights. Drag to rotate.':'三个残差激活矩阵，由所选注意力头的权重连接。拖拽可旋转。'} style={{touchAction:'pan-y'}}
      onPointerDown={e=>{if(e.pointerType==='touch')return;drag.current={x:e.clientX,y:e.clientY,...angle};e.currentTarget.setPointerCapture(e.pointerId)}}
      onPointerMove={e=>{if(!drag.current)return;const a=drag.current;setAngle({yaw:Math.max(-1.25,Math.min(-.15,a.yaw+(e.clientX-a.x)*.004)),pitch:Math.max(-.5,Math.min(.6,a.pitch+(e.clientY-a.y)*.003))})}}
      onPointerUp={()=>{drag.current=null}} onPointerCancel={()=>{drag.current=null}}>
      <title>{en?'True activations, projected in 3D':'真实激活的三维投影'}</title>
      <defs><pattern id="ov-grid" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".7" fill="#34413b"/></pattern></defs>
      <rect width="840" height="540" fill="url(#ov-grid)" opacity=".45"/>
      {[0,1].map(l=>trace.ids.map((_,t)=>{const a=point(l,t+.5,16),b=point(l+1,t+.5,16);return <line key={`${l}-${t}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#5e776d" strokeOpacity=".12"/>}))}
      {trace.states.map((state,l)=><g key={l}>
        <polygon points={[point(l,0,0),point(l,n,0),point(l,n,32),point(l,0,32)].map(p=>p.join(',')).join(' ')} fill="#0d1713" stroke={l===layer+1?'#8be7c3':'#476055'} strokeWidth={l===layer+1?1.5:.7}/>
        {state.map((row,t)=>row.map((v,c)=><polygon key={`${t}-${c}`} points={poly(l,t,c)} fill={color(v)} stroke="#101b16" strokeWidth=".5"><title>{`L${l} · ${vocab[trace.ids[t]]}[${t}] · d${c}: ${v.toFixed(5)}`}</title></polygon>))}
        <polygon points={[point(l,token,0),point(l,token+1,0),point(l,token+1,32),point(l,token,32)].map(p=>p.join(',')).join(' ')} fill="none" stroke="#eeeedc" strokeWidth="1.2"/>
        <text x={point(l,n/2,39)[0]} y={point(l,n/2,39)[1]} textAnchor="middle" fill={l===layer+1?'#b6f6dc':'#bac4be'} fontSize="15">{l===0?'EMBEDDING':`BLOCK 0${l}`}</text>
        <text x={point(l,n/2,42)[0]} y={point(l,n/2,42)[1]} textAnchor="middle" fill="#82988c" fontSize="12">{n} × 32</text>
      </g>)}
      {trace.attention[layer][head][token].map((a,t)=>{
        if(a<.005)return null;const from=point(layer,t+.5,16),to=point(layer+1,token+.5,16);const bend=(from[0]+to[0])/2;
        return <path key={t} d={`M ${from} C ${bend} ${from[1]-58},${bend} ${to[1]-58},${to}`} fill="none" stroke={ablated?'#f18b76':'#adffda'} strokeWidth={.6+4*Math.sqrt(a)} opacity={ablated?.18:.22+.65*a} className={ablated?'':'ov-flow'}><title>{`${vocab[trace.ids[t]]}[${t}] → ${vocab[trace.ids[token]]}[${token}]: ${(100*a).toFixed(2)}%`}</title></path>
      })}
      <text x="28" y="504" fill="#9daf9f" fontSize="12">{en?'CHANNELS × TOKEN POSITIONS':'通道 × TOKEN 位置'}</text>
      <text x="810" y="504" textAnchor="end" fill="#9daf9f" fontSize="12">{en?'PATH WIDTH = ATTENTION':'连线宽度 = 注意力权重'}</text>
    </svg>
    <div className="ov-volume-bottom"><span><i className="ov-negative"/> −{magnitude.toFixed(1)} <i className="ov-positive"/> +{magnitude.toFixed(1)}</span><label>{en?'Rotate':'旋转'}<input type="range" min="-125" max="-15" value={Math.round(angle.yaw*100)} onChange={e=>setAngle(a=>({...a,yaw:Number(e.target.value)/100}))}/></label><span>{en?'Drag to orbit · hover a cell for its value':'拖动旋转 · 悬停查看数值'}</span></div>
  </div>
}
