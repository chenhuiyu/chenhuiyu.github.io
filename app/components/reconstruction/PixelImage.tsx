'use client';
import {useId} from 'react';
import {pixelMask} from '@/lib/reconstruction/model';
export default function PixelImage({pixels,mask,heat=false,label}:{pixels:number[];mask?:number[];heat?:boolean;label:string}){
 const id=useId().replace(/:/g,''),hidden=mask?pixelMask(mask):null;
 return <svg viewBox="0 0 8 8" role="img" aria-label={label} shapeRendering="crispEdges"><title>{label}</title><defs><pattern id={id} width=".25" height=".25" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width=".25" height=".25" fill="#243429"/><line x1="0" y1="0" x2="0" y2=".25" stroke="#7b8658" strokeWidth=".055"/></pattern></defs>{pixels.map((v,i)=><rect key={i} x={i%8} y={Math.floor(i/8)} width="1" height="1" fill={hidden?.[i]?`url(#${id})`:heat?`rgb(${Math.round(30+225*v)},${Math.round(36+135*v)},${Math.round(28+34*v)})`:`rgb(${Math.round(14+220*v)},${Math.round(23+220*v)},${Math.round(18+208*v)})`}><title>{`[${Math.floor(i/8)},${i%8}] ${v.toFixed(4)}`}</title></rect>)}</svg>
}
