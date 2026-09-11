import Observatory from '@/app/components/observatory/Observatory';
import {observatoryMetadata,observatorySchema} from '../metadata';
import '../observatory.css';
export const metadata=observatoryMetadata(true);
export default function Page(){return <><script type="application/ld+json" dangerouslySetInnerHTML={{__html:JSON.stringify(observatorySchema(true))}}/><Observatory en/></>}
