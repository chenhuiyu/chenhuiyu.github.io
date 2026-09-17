// A local practice runner, not an adversarial/remote examination sandbox.
let runtime;
let busy = false;
const harness = `
import json, math, traceback
def _judge(payload):
    job = json.loads(payload)
    namespace = {'__name__': '__submission__'}
    def equal(a, b):
        if isinstance(b, bool) or b is None:
            return type(a) is type(b) and a == b
        if isinstance(b, (int, float)):
            return isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6)
        if isinstance(b, list):
            return isinstance(a,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        if isinstance(b, dict):
            return isinstance(a,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in b)
        return type(a) is type(b) and a == b
    try:
        exec(compile(job['code'], '<submission>', 'exec'), namespace)
    except BaseException:
        return json.dumps({'error':traceback.format_exc(limit=4)[-4000:], 'cases':[]})
    results=[]
    for case in job['tests']:
        try:
            actual=eval(case['expression'], namespace)
            passed='raises' not in case and equal(actual,case.get('expected'))
            results.append({'passed':passed,'actual':repr(actual)[:500]})
        except BaseException as exc:
            results.append({'passed':type(exc).__name__==case.get('raises'), 'actual':type(exc).__name__+': '+str(exc)[:400]})
    return json.dumps({'cases':results})
_judge(_interview_payload)
`;
self.addEventListener('message',async ({data})=>{
  if(busy || typeof data?.code!=='string' || !Array.isArray(data.tests) || !Number.isInteger(data.runId)) return;
  busy=true;
  const {runId}=data;
  let output='';
  const append=text=>{if(output.length<20000)output+=(text+'\n').slice(0,20000-output.length)};
  try {
    if(data.code.length>50000) throw Error('Submission exceeds 50,000 characters.');
    if(!runtime){
      self.postMessage({type:'loading',runId});
      const indexURL=new URL('../vendor/pyodide/',import.meta.url).href;
      runtime=import(new URL('pyodide.mjs',indexURL).href).then(({loadPyodide})=>loadPyodide({indexURL:indexURL.startsWith('file:')?new URL(indexURL).pathname:indexURL}));
    }
    const p=await runtime;
    p.setStdout({batched:append});p.setStderr({batched:append});
    p.globals.set('_interview_payload',JSON.stringify({code:data.code,tests:data.tests}));
    self.postMessage({type:'running',runId});
    const result=JSON.parse(await p.runPythonAsync(harness));
    self.postMessage({type:'result',runId,...result,output});
  } catch(error){self.postMessage({type:'error',runId,error:String(error),output});}
  finally{busy=false;}
});
