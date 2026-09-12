import ReconstructionLab from '@/app/components/reconstruction/ReconstructionLab';
import {reconstructionMetadata,reconstructionSchema} from './metadata';
import './reconstruction.css';
export const metadata=reconstructionMetadata(false);
export default function Page(){return <><script type="application/ld+json" dangerouslySetInnerHTML={{__html:JSON.stringify(reconstructionSchema(false))}}/><ReconstructionLab/></>}
