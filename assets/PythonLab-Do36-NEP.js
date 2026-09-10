import{r as e}from"./rolldown-runtime-S-ySWqyJ.js";import{i as t,r as n}from"./framework-CXnKph_e.js";var r=e(t(),1),i=n(),a={"llm-foundations":`import math
# Change the logits and temperature, then predict the output.
logits = [2.0, 1.0, -1.0]
temperature = 0.8
assert temperature > 0
z = [math.exp((x-max(logits))/temperature) for x in logits]
p = [x/sum(z) for x in z]
print("probabilities:", [round(x, 4) for x in p])
print("sum:", sum(p))
print("target=0, NLL:", -math.log(p[0]))`,"llm-infra":`# FP16 weights/gradients + FP32 master weights + Adam moments.
parameters = 7_000_000_000
ranks = 8
assert ranks > 0
for stage in range(4):
    weights = 2*parameters / (ranks if stage >= 3 else 1)
    grads = 2*parameters / (ranks if stage >= 2 else 1)
    optimizer = 12*parameters / (ranks if stage >= 1 else 1)
    print("ZeRO",stage,"GiB per rank:",round((weights+grads+optimizer)/2**30, 2))
print("Excludes activations, gathered layers, communication buffers.")`,multimodal:`import math
# A two-pair contrastive objective. Rows=image, columns=text.
similarity = [[0.8, 0.2], [0.3, 0.7]]
temperature = 0.1
assert temperature > 0
losses = []
for i, row in enumerate(similarity):
    z = [math.exp((x-max(row))/temperature) for x in row]
    p = [x/sum(z) for x in z]
    losses.append(-math.log(p[i]))
    print("image",i,"probabilities:",p)
print("Image-to-text mean loss:",sum(losses)/len(losses))
print("CLIP also trains the reverse text-to-image direction.")`,hstu:`import math
# Educational HSTU-like aggregation; these are not probabilities.
q = [1.0, 0.5]
keys = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]
values = [[1.0, 2.0], [3.0, 1.0], [0.0, 4.0]]
scores = [sum(a*b for a,b in zip(q,k)) for k in keys]
weights = [s/(1+math.exp(-s))/len(keys) for s in scores]
output = [sum(w*v[d] for w,v in zip(weights,values)) for d in range(2)]
print("weights:",weights,"sum:",sum(weights))
print("aggregated V:",output)
print("Full block: normalize, gate by U, project, add residual.")`};function o({track:e,locale:t}){let n=t===`zh-CN`,o=a[e]??a[`llm-foundations`],[s,c]=(0,r.useState)(o),[l,u]=(0,r.useState)(``),[d,f]=(0,r.useState)(!1),p=(0,r.useRef)(null),m=(0,r.useRef)(null);(0,r.useEffect)(()=>()=>{p.current?.terminate(),m.current&&clearTimeout(m.current)},[]);function h(){p.current?.terminate(),p.current=null,m.current&&clearTimeout(m.current),f(!1)}function g(){h(),f(!0),u(n?`首次运行正在加载 Python…`:`Loading Python for the first run…`);try{let e=new Worker(`/workers/generative-lab-worker.mjs`,{type:`module`});p.current=e,e.onmessage=({data:e})=>{(e.type===`result`||e.type===`error`)&&(u(e.output||(n?`运行完成，无输出。`:`Finished without output.`)),h())},e.onerror=e=>{u(e.message),h()},e.postMessage({code:s,runId:1}),m.current=setTimeout(()=>{h(),u(n?`运行超时，已停止。可修改代码后重试。`:`Execution timed out. Edit the code and retry.`)},6e4)}catch(e){h(),u(String(e))}}return(0,i.jsxs)(`section`,{className:`concept-lab`,id:`editable-python`,children:[(0,i.jsx)(`p`,{className:`section-kicker`,children:`PYTHON · WEBASSEMBLY · NO GPU`}),(0,i.jsx)(`h2`,{children:n?`改代码，再验证你的预测`:`Edit the code and test your prediction`}),(0,i.jsx)(`p`,{children:n?`真实 Python 在独立 Worker 中运行，只用标准库；60 秒自动终止。完整 PyTorch 和模型权重实验请打开 Notebook。`:`Real Python runs in a separate worker, using the standard library, with a 60-second limit. Use the notebook for PyTorch and pretrained model experiments.`}),(0,i.jsxs)(`label`,{children:[n?`Python 代码`:`Python code`,(0,i.jsx)(`textarea`,{className:`python-editor`,spellCheck:!1,value:s,onChange:e=>c(e.target.value),disabled:d})]}),(0,i.jsxs)(`div`,{className:`lab-actions`,children:[(0,i.jsx)(`button`,{onClick:g,disabled:d,children:n?`运行 Python`:`Run Python`}),(0,i.jsx)(`button`,{onClick:()=>{h(),u(n?`已停止`:`Stopped`)},disabled:!d,children:n?`停止`:`Stop`}),(0,i.jsx)(`button`,{onClick:()=>{c(o),u(``)},disabled:d,children:n?`恢复示例`:`Reset code`})]}),(0,i.jsx)(`pre`,{role:`status`,"aria-live":`polite`,children:l||(n?`运行后显示实际输出。`:`Actual output appears after running.`)})]})}export{o as PythonLab};