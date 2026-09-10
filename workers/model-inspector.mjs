// Explicitly loaded on reader action; inference runs locally in this worker.
let pipe;
self.onmessage=async ({data})=>{
 const {texts,runId}=data;
 if(!Array.isArray(texts)||texts.length!==2||texts.some(x=>typeof x!=='string'||x.length>1000))return;
 try{
  const {pipeline,env,mean_pooling}=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
  env.allowLocalModels=false;
  env.backends.onnx.wasm.numThreads=1;
  pipe??=await pipeline('feature-extraction','Xenova/all-MiniLM-L6-v2',{device:'wasm',dtype:'q8',progress_callback:p=>self.postMessage({runId,type:'progress',file:p.file??'',progress:p.progress??null,status:p.status})});
  const encoded=pipe.tokenizer(texts,{padding:true,truncation:true,max_length:128});
  const raw=await pipe.model(encoded);
  const out=mean_pooling(raw.last_hidden_state,encoded.attention_mask).normalize(2,-1);
  const embeddings=out.tolist();
  const cosine=embeddings[0].reduce((s,v,i)=>s+v*embeddings[1][i],0);
  const ids=encoded.input_ids.tolist().map(row=>row.map(Number));
  const masks=encoded.attention_mask.tolist().map(row=>row.map(Number));
  self.postMessage({runId,type:'result',cosine,embeddings,ids,masks,tokens:ids.map(row=>pipe.tokenizer.model.convert_ids_to_tokens(row)),dims:out.dims});
 }catch(e){pipe=undefined;self.postMessage({runId,type:'error',message:String(e)})}
};
