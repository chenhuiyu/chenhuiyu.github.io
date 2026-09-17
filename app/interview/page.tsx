import InterviewApp from './InterviewApp';
import {interviewMetadata,interviewSchema} from './metadata';
import './interview.css';
export const metadata=interviewMetadata(false);
export default function Page(){return <><script type="application/ld+json" dangerouslySetInnerHTML={{__html:JSON.stringify(interviewSchema(false))}}/><InterviewApp/></>}
