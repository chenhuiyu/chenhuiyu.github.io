import {notFound} from 'next/navigation';
import {TrackPage,learningMetadata} from '../LearningPage';
import {tracks,getTrack} from '@/lib/learning/tracks';
export function generateStaticParams(){return tracks.map(t=>({track:t.id}))}
export async function generateMetadata({params}:{params:Promise<{track:string}>}){const {track}=await params;return learningMetadata('zh-CN',track)}
export default async function Page({params}:{params:Promise<{track:string}>}){const {track}=await params;if(!getTrack(track))notFound();return <TrackPage id={track} locale="zh-CN"/>}
