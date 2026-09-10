#!/usr/bin/env python3
"""Generate paired, reviewable notebooks. --execute runs one edition and shares identical code outputs."""
from pathlib import Path
import argparse, copy, json, os
import nbformat as nbf
import io, contextlib, base64, traceback
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'public/notebooks'
SETUP='''import sys, subprocess, importlib.util
needed = [p for p in ['torch','transformers','numpy','matplotlib','PIL'] if importlib.util.find_spec(p) is None]
if needed:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch>=2.6,<3', 'transformers==4.57.1', 'numpy', 'matplotlib', 'pillow'])
import torch, numpy as np, matplotlib.pyplot as plt, transformers
print({'python':sys.version.split()[0], 'torch':torch.__version__, 'transformers':transformers.__version__, 'cuda':torch.cuda.is_available()})
torch.manual_seed(7); np.random.seed(7)
torch.set_num_threads(min(4, torch.get_num_threads()))
'''
BOOKS={}
def book(name,zt,et,zintro,eintro,cells):BOOKS[name]=(zt,et,zintro,eintro,cells)
def section(zh,en,code):return (zh,en,code)
book('llm-model-io','真实语言模型输入输出实验','Real language-model I/O laboratory',
'CPU 可运行；首次从 Hugging Face 下载 DistilGPT2 权重（数百 MB），建议至少 2 GB 可用内存。需要网络。它是英文预训练小模型，不是聊天助手。所有 attention 图来自实际前向；随机教学示例会另作说明。依赖 transformers 4.57.1；若环境不同，请先运行 `%pip install transformers==4.57.1` 后重启内核。',
'Runs on CPU; the first run downloads DistilGPT2 weights (hundreds of MB) from Hugging Face. Allow at least 2 GB free memory and network access. This is a small English pretrained model, not a chat assistant. Attention plots come from actual forward passes. Target transformers 4.57.1; for another environment run `%pip install transformers==4.57.1` and restart the kernel.',[
section('## 1 · 模型配置与可复现输入\n先改 prompt，预测 token 数。模型 revision 可改为固定 commit；下面会打印已解析的 revision。','## 1 · Configuration and reproducible inputs\nEdit the prompt and predict its token count. MODEL_REVISION can be a pinned commit; the resolved revision is printed.', '''from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_ID = 'distilbert/distilgpt2'
MODEL_REVISION = 'main'
PROMPT = 'The cat is sleeping on the'
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, attn_implementation='eager').eval()
inputs = tokenizer(PROMPT, return_tensors='pt', truncation=True, max_length=128)
print('resolved revision:', model.config._commit_hash)
print({k:tuple(v.shape) for k,v in inputs.items()})
ids = inputs['input_ids'][0].tolist()
tokens = tokenizer.convert_ids_to_tokens(ids)
print(list(zip(ids,tokens)))
print('layers / heads / hidden:', model.config.n_layer, model.config.n_head, model.config.n_embd)
'''),
section('## 2 · 一次前向究竟返回什么\n区分 embedding 输出、每层 hidden states 与词表 logits。','## 2 · What a forward pass returns\nDistinguish input embeddings, layer states and vocabulary logits.', '''with torch.no_grad():
    out = model(**inputs, output_hidden_states=True, output_attentions=True, use_cache=True)
print('logits:',tuple(out.logits.shape))
print('hidden states:',[tuple(h.shape) for h in out.hidden_states])
print('attention:',[tuple(a.shape) for a in out.attentions])
assert out.logits.shape[:2] == inputs['input_ids'].shape
assert len(out.hidden_states) == model.config.n_layer+1
p = out.logits[0,-1].softmax(-1)
values, indices = p.topk(8)
for index,value in zip(indices.tolist(), values.tolist()):
    print(repr(tokenizer.decode([index])),round(value,6))
'''),
section('## 3 · 可视化真实 attention\n改变 LAYER / HEAD 重新运行。横轴是被读取位置，纵轴是 query；不是因果解释。','## 3 · Visualize actual attention\nChange LAYER / HEAD. Columns are read positions and rows are queries. This is not a causal explanation.', '''LAYER, HEAD = 0, 0
weights = out.attentions[LAYER][0,HEAD].detach().cpu()
assert torch.allclose(weights.sum(-1),torch.ones(weights.shape[0]),atol=1e-5)
assert weights.triu(1).abs().max().item() < 1e-6
fig,ax=plt.subplots(figsize=(8,6))
im=ax.imshow(weights.numpy(),cmap='Blues',vmin=0)
ax.set_xticks(range(len(tokens)),tokens,rotation=65,ha='right')
ax.set_yticks(range(len(tokens)),tokens)
ax.set(xlabel='Key position',ylabel='Query position',title=f'Actual attention: layer {LAYER}, head {HEAD}')
fig.colorbar(im,ax=ax);plt.tight_layout();plt.show()
'''),
section('## 4 · KV cache 等价性检验\n单条无 padding 序列。先计算前缀，再只输入最后一个 token。不要把整个 prompt 再附到 cache 后面。','## 4 · Check KV-cache equivalence\nOne unpadded sequence: cache its prefix, then feed only the last token. Do not append the entire prompt again.', '''assert inputs['input_ids'].shape[1] >= 2
with torch.no_grad():
    prefix = model(input_ids=inputs['input_ids'][:,:-1],use_cache=True)
    last = model(input_ids=inputs['input_ids'][:,-1:],past_key_values=prefix.past_key_values,use_cache=True)
max_error=(last.logits[:,-1]-out.logits[:,-1]).abs().max().item()
print('full vs cached max error:',max_error)
assert torch.allclose(last.logits[:,-1],out.logits[:,-1],atol=1e-4,rtol=1e-4)
'''),
section('## 5 · 温度与生成\n比较概率熵，再比较 greedy 和采样。不同版本与设备可能产生数值差异；固定 seed 不保证跨硬件逐位一致。','## 5 · Temperature and generation\nCompare entropy, then greedy and sampled output. A fixed seed does not ensure bitwise equality across devices and versions.', '''for temperature in [.3,1.,2.]:
    prob=(out.logits[0,-1]/temperature).softmax(-1)
    entropy=-(prob*prob.clamp_min(1e-20).log()).sum()
    print('temperature / entropy:',temperature,entropy.item())
with torch.no_grad():
    greedy=model.generate(**inputs,max_new_tokens=24,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    torch.manual_seed(7)
    sampled=model.generate(**inputs,max_new_tokens=24,do_sample=True,temperature=.8,top_p=.9,pad_token_id=tokenizer.eos_token_id)
print('greedy:',tokenizer.decode(greedy[0],skip_special_tokens=True))
print('sample:',tokenizer.decode(sampled[0],skip_special_tokens=True))
''')])
book('infra-benchmark','Infra：理论预算与真实计时','Infrastructure: budgets and actual timings',
'CPU 可运行全部默认实验；CUDA 可用时自动改用 GPU。CPU 时间不代表 GPU/TPU 性能。Notebook 验证 TP 数学分解、ZeRO 状态预算和实际 matmul 时间；不是五大框架的端到端 benchmark。框架启动与 profiling 见配套章节。',
'All default experiments run on CPU, selecting CUDA when available. CPU timings do not represent GPU/TPU performance. This notebook verifies TP algebra, ZeRO state accounting and actual matmul timing, not end-to-end benchmarks of five frameworks. See the chapters for serving/profiling workflows.',[
section('## 1 · 显存账本\n改并发和长度。GiB 用 2³⁰，GB 用 10⁹；两者不要混用。','## 1 · Memory accounting\nChange concurrency and length. GiB uses 2³⁰; GB uses 10⁹.', '''B,T,L,Hkv,Dh,bytes_per_element=4,4096,32,8,128,2
kv=2*B*T*L*Hkv*Dh*bytes_per_element
print('KV GiB:',kv/2**30)
assert kv/2**30 == 2
N,P=7_000_000_000,8
states=[16*N,4*N+12*N/P,2*N+14*N/P,16*N/P]
for stage,bytes_used in enumerate(states):
    print(f'ZeRO {stage}: {bytes_used/2**30:.3f} GiB per rank (persistent state only)')
plt.bar(['DP','ZeRO-1','ZeRO-2','ZeRO-3'],np.array(states)/2**30)
plt.ylabel('GiB / rank');plt.title('Assumed 16-byte Adam states; excludes activations/gathers');plt.show()
'''),
section('## 2 · 模拟张量并行，验证数值\n这是单进程模拟，没有 NCCL 通信，不能测 scaling。','## 2 · Simulate tensor partitioning\nSingle-process algebra only: no NCCL and no scaling measurement.', '''x,w=torch.randn(16,64),torch.randn(64,128)
ref=x@w
column=torch.cat([x@part for part in w.chunk(4,dim=1)],dim=1)
row=sum(a@b for a,b in zip(x.chunk(4,dim=1),w.chunk(4,dim=0)))
assert torch.allclose(column,ref,atol=1e-4,rtol=1e-4)
assert torch.allclose(row,ref,atol=1e-4,rtol=1e-4)
print('Column and row partition equivalence passed.')
S=64*1024**2
print('Ideal ring all-reduce MiB/rank:',2*(P-1)/P*S/2**20)
'''),
section('## 3 · 真实 matmul 时间\n先 warm-up，GPU 用 events，CPU 用 perf_counter。输出包含所有 repeats 的分布；没有外推成服务吞吐。','## 3 · Actual matmul timings\nWarm up, use GPU events or CPU perf_counter, and report repeats. Do not extrapolate this into service throughput.', '''import time, platform
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
DTYPE=torch.float16 if DEVICE=='cuda' else torch.float32
print('device:',torch.cuda.get_device_name(0) if DEVICE=='cuda' else platform.processor() or 'CPU')
print('dtype:',DTYPE)
results=[]
for m in [1,16,128,512]:
    a=torch.randn(m,1024,device=DEVICE,dtype=DTYPE)
    b=torch.randn(1024,1024,device=DEVICE,dtype=DTYPE)
    for _ in range(8): c=a@b
    times=[]
    for _ in range(20):
        if DEVICE=='cuda':
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); c=a@b; end.record(); end.synchronize()
            elapsed=start.elapsed_time(end)
        else:
            start=time.perf_counter(); c=a@b; elapsed=(time.perf_counter()-start)*1000
        times.append(elapsed)
    row={'m':m,'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95))}
    results.append(row);print(row)
    assert torch.isfinite(c).all()
plt.plot([r['m'] for r in results],[r['p50_ms'] for r in results],marker='o',label='p50')
plt.plot([r['m'] for r in results],[r['p95_ms'] for r in results],marker='o',label='p95')
plt.xlabel('Rows M');plt.ylabel('Measured milliseconds');plt.title(f'{DEVICE}: (M,1024) @ (1024,1024)');plt.legend();plt.show()
'''),
section('## 4 · 保存可复查的结果\n这个 JSON 包含环境与实测值，不含未执行的 GPU 数字。下一步可固定一个 shape，用 Nsight/XProf 分析对应实现。','## 4 · Save reviewable results\nThe JSON contains the environment and actual timings, without invented GPU numbers. Next, fix one shape and profile its implementation.', '''import json
report={'device':DEVICE,'dtype':str(DTYPE),'torch':torch.__version__,'threads':torch.get_num_threads(),'results':results}
with open('infra-benchmark-results.json','w') as f: json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
''')])
book('multimodal-inspection','多模态：CLIP 与 MAE 的真实模型实验','Multimodal: actual CLIP and MAE experiments',
'CPU 可运行；首次下载两套模型权重，合计超过 1 GB，建议至少 4 GB 空闲内存。默认使用明确标注的合成色块图，无外部图片版权问题；改 IMAGE_PATH 使用自己的图片。依赖 transformers 4.57.1。合成图上的分数不代表现实图像质量。',
'Runs on CPU; first use downloads two weight sets totaling over 1 GB. Allow at least 4 GB free memory. The default is an explicitly synthetic color-grid image; set IMAGE_PATH to your own image. Target transformers 4.57.1. Scores on synthetic pixels do not measure real-image quality.',[
section('## 1 · 准备输入并检查 patch 往返\n先检查通道与空间顺序，再加载模型。','## 1 · Prepare an input and verify patch round trips\nCheck channels and spatial order before loading a model.', '''from PIL import Image
IMAGE_PATH = ''
if IMAGE_PATH:
    image=Image.open(IMAGE_PATH).convert('RGB')
else:
    yy,xx=np.mgrid[:224,:224]
    pixels=np.stack([(xx//28%2)*255,(yy//28%2)*255,((xx+yy)//56%2)*255],axis=-1).astype('uint8')
    image=Image.fromarray(pixels)
plt.imshow(image);plt.title('Input image (synthetic grid unless IMAGE_PATH is set)');plt.axis('off');plt.show()
x=torch.tensor(np.asarray(image.resize((224,224))).copy()).permute(2,0,1)[None].float()/255
patches=x.reshape(1,3,14,16,14,16).permute(0,2,4,1,3,5).reshape(1,196,768)
restored=patches.reshape(1,14,14,3,16,16).permute(0,3,1,4,2,5).reshape_as(x)
assert torch.equal(x,restored)
print('patches:',patches.shape)
'''),
section('## 2 · 加载真实 CLIP\n修改候选描述，观察条件 softmax 如何随候选集合变化。','## 2 · Load real CLIP\nChange candidate descriptions and inspect how conditional softmax depends on the candidate set.', '''from transformers import CLIPModel, AutoProcessor
clip_id='openai/clip-vit-base-patch32'
clip=CLIPModel.from_pretrained(clip_id).eval()
processor=AutoProcessor.from_pretrained(clip_id)
labels=['a colorful geometric pattern','a photograph of a cat','a mountain landscape']
clip_inputs=processor(text=labels,images=image,return_tensors='pt',padding=True)
with torch.no_grad(): clip_out=clip(**clip_inputs)
print({k:tuple(v.shape) for k,v in clip_inputs.items()})
print('image embedding:',clip_out.image_embeds.shape,'text embeddings:',clip_out.text_embeds.shape)
print('resolved revision:',clip.config._commit_hash)
prob=clip_out.logits_per_image.softmax(-1)[0]
for label,p in zip(labels,prob.tolist()):print(label,round(p,5))
plt.barh(labels,prob.numpy());plt.xlim(0,1);plt.xlabel('Probability relative to these candidate labels');plt.tight_layout();plt.show()
del clip,clip_out,clip_inputs
import gc;gc.collect()
'''),
section('## 3 · 加载真实 MAE，改变 mask ratio\n固定随机 seed 保持 mask 可复现。读取 norm_pix_loss 决定是否需恢复 patch 目标尺度。','## 3 · Load real MAE and change masking\nFix the random seed. Inspect norm_pix_loss before interpreting reconstruction scale.', '''from transformers import ViTMAEForPreTraining, AutoImageProcessor
mae_id='facebook/vit-mae-base'
mae=ViTMAEForPreTraining.from_pretrained(mae_id).eval()
image_processor=AutoImageProcessor.from_pretrained(mae_id)
MASK_RATIO=.75
assert 0 < MASK_RATIO < 1
mae.config.mask_ratio=MASK_RATIO
mae_inputs=image_processor(images=image,return_tensors='pt')
torch.manual_seed(7)
with torch.no_grad(): mae_out=mae(**mae_inputs)
print('resolved revision:',mae.config._commit_hash)
print('pixels:',mae_inputs['pixel_values'].shape,'logits:',mae_out.logits.shape)
print('masked patches:',mae_out.mask.sum().item(),'loss:',mae_out.loss.item())
assert torch.isfinite(mae_out.loss)
'''),
section('## 4 · 显示真实重建\n可见区域沿用输入，隐藏区域使用预测。若使用 normalized pixel targets，恢复目标统计仅用于诊断显示，不是部署时知道隐藏真值的方案。','## 4 · Display the actual reconstruction\nKeep visible input and fill hidden regions from predictions. If normalized pixel targets are enabled, target statistics are used only for diagnostic display, not as a deployment method with hidden ground truth.', '''pixel_values=mae_inputs['pixel_values']
pred=mae_out.logits
if mae.config.norm_pix_loss:
    target=mae.patchify(pixel_values)
    mean=target.mean(-1,keepdim=True)
    var=target.var(-1,keepdim=True)
    pred=pred*(var+1e-6).sqrt()+mean
reconstruction=mae.unpatchify(pred)
mask=mae.unpatchify(mae_out.mask.unsqueeze(-1).repeat(1,1,pred.shape[-1]))
mean=torch.tensor(image_processor.image_mean).view(1,3,1,1)
std=torch.tensor(image_processor.image_std).view(1,3,1,1)
def display_pixels(t):return ((t*std+mean).clamp(0,1)[0].permute(1,2,0)).numpy()
original=display_pixels(pixel_values)
masked=display_pixels(pixel_values*(1-mask))
filled=display_pixels(pixel_values*(1-mask)+reconstruction*mask)
fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,img,title in zip(axes,[original,masked,filled],['Original','Masked','Actual MAE reconstruction']):
    ax.imshow(img);ax.set_title(title);ax.axis('off')
plt.tight_layout();plt.show()
''')])
book('hstu-from-scratch','从零训练 HSTU-inspired 微型推荐器','Train a tiny HSTU-inspired recommender',
'全部实验在 CPU 上运行，无模型下载。使用合成数据、单头、简化相对位置偏置；不是官方 HSTU 复现，不含工业特征、Stochastic Length 或 M-FALCON。数据与模型在下方完整定义。图中的 loss 和 top-k 来自本次真实训练。',
'All experiments run on CPU with no model download. Uses synthetic data, one head and simplified relative-position bias. This is not an official HSTU reproduction and omits industrial features, Stochastic Length and M-FALCON. Data and model definitions are complete below. Loss and top-k come from actual training.',[
section('## 1 · 可控序列数据\n两组商品循环，0 保留给 padding。保留不同起点作简单测试；这不是现实分布的泛化证明。','## 1 · Controlled sequence data\nTwo item cycles; zero is padding. Hold out different starts for a simple test, not a real-world generalization claim.', '''import torch.nn as nn
import torch.nn.functional as F
cycles=[[1,2,3,4],[5,6,7,8]]
def sequence(cycle,start,length=9):return [cycle[(start+j)%len(cycle)] for j in range(length)]
train=torch.tensor([sequence(c,start) for c in cycles for start in [0,1]])
valid=torch.tensor([sequence(c,start) for c in cycles for start in [2,3]])
inputs,targets=train[:,:-1],train[:,1:]
print('train:',train.tolist(),'held-out:',valid.tolist())
'''),
section('## 2 · 定义完整教学模型\n先 norm，再融合投影为 U/V/Q/K；用 causal 与有效 key mask，归一化聚合后由 U 门控。','## 2 · Define the complete teaching model\nNormalize, project into U/V/Q/K, apply causal and valid-key masks, then normalize the aggregate and gate with U.', '''class TinyHSTU(nn.Module):
    def __init__(self,vocab=9,dim=24):
        super().__init__()
        self.embedding=nn.Embedding(vocab,dim,padding_idx=0)
        self.norm=nn.LayerNorm(dim)
        self.uvqk=nn.Linear(dim,4*dim)
        self.aggregate_norm=nn.LayerNorm(dim)
        self.output=nn.Linear(dim,dim)
        self.head=nn.Linear(dim,vocab)
        self.time_strength=nn.Parameter(torch.tensor(.1))
    def forward(self,ids):
        x=self.embedding(ids);b,t,d=x.shape
        u,v,q,k=F.silu(self.uvqk(self.norm(x))).chunk(4,-1)
        positions=torch.arange(t,device=ids.device)
        age=(positions[:,None]-positions[None,:]).clamp_min(0).float()
        score=q@k.transpose(-2,-1)-self.time_strength.abs()*age
        allowed=(positions[None,:]<=positions[:,None])[None]&ids.ne(0)[:,None,:]
        a=(F.silu(score)/t).masked_fill(~allowed,0)
        h=x+self.output(u*self.aggregate_norm(a@v))
        return self.head(h)
model=TinyHSTU()
print(model)
print('parameters:',sum(p.numel() for p in model.parameters()))
print('logits:',model(inputs).shape)
'''),
section('## 3 · 真正训练并画 loss\n修改学习率或步数，观察是否收敛。grad_norm 只是诊断，不是质量指标。','## 3 · Train and plot actual loss\nChange learning rate or steps. Gradient norm is diagnostic, not a quality metric.', '''optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
losses=[]
for step in range(150):
    model.train()
    logits=model(inputs)
    loss=F.cross_entropy(logits.reshape(-1,9),targets.reshape(-1),ignore_index=0)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    grad_norm=nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step();losses.append(loss.item())
print('initial / final loss:',losses[0],losses[-1],'last grad norm:',float(grad_norm))
assert np.isfinite(losses).all() and losses[-1]<losses[0]
plt.plot(losses);plt.xlabel('Training step');plt.ylabel('Cross-entropy');plt.title('Actual synthetic-data training');plt.show()
'''),
section('## 4 · 因果与 padding 测试\n改变未来 token；再改变 padding embedding。合法的较早输出不能变化。','## 4 · Causality and padding tests\nChange future tokens, then padding embeddings. Earlier legal outputs must remain unchanged.', '''model.eval()
with torch.no_grad():
    base=model(inputs)
    changed=inputs.clone();changed[:,-1]=8
    future_changed=model(changed)
    assert torch.allclose(base[:,:-1],future_changed[:,:-1],atol=1e-6)
    padded=torch.tensor([[1,2,0,0]])
    original=model(padded)[:,:2]
    old=model.embedding.weight[0].clone()
    model.embedding.weight[0].fill_(100.)
    modified=model(padded)[:,:2]
    model.embedding.weight[0].copy_(old)
    assert torch.allclose(original,modified,atol=1e-6)
print('Causality and padding checks passed.')
'''),
section('## 5 · 推荐输出与 holdout 指标\n屏蔽 ID 0。保持重复消费候选，因为此任务本身是循环行为。','## 5 · Recommendations and held-out metrics\nExclude ID zero. Keep repeated items because this task deliberately models cycles.', '''with torch.no_grad():
    scores=model(valid[:,:-1])[:,-1].clone();scores[:,0]=-torch.inf
    top=scores.topk(3,-1).indices
    truth=valid[:,-1]
    hit=(top==truth[:,None])
    hr3=hit.any(-1).float().mean().item()
    discounts=1/torch.log2(torch.arange(2,5).float())
    ndcg3=(hit.float()*discounts).sum(-1).mean().item()
print('HR@3:',hr3,'NDCG@3:',ndcg3)
for seq,pred,target in zip(valid[:,:-1].tolist(),top.tolist(),truth.tolist()):
    print('history:',seq,'top3:',pred,'target:',target)
assert not (top==0).any()
''')])
def create(name,locale):
    zt,et,zi,ei,cells=BOOKS[name];zh=locale=='zh-CN'
    nb=nbf.v4.new_notebook();nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'learning_lab':{'execution_status':'not executed','source':'scripts/build-learning-notebooks.py'}}
    nb.cells=[nbf.v4.new_markdown_cell('# '+(zt if zh else et)+'\n\n'+(zi if zh else ei)+'\n\n'+('执行状态：尚未运行。使用「全部运行」生成实际结果。' if zh else 'Execution status: not yet run. Run all cells to obtain actual results.')),nbf.v4.new_code_cell(SETUP)]
    for z,e,code in cells:nb.cells += [nbf.v4.new_markdown_cell(z if zh else e),nbf.v4.new_code_cell(code)]
    nb.cells.append(nbf.v4.new_markdown_cell(('## 下一步\n改变一个参数，先预测结果，再重跑。记录环境、seed、输入与失败情况；不要把教学实验外推成生产结论。' if zh else '## Next experiment\nChange one parameter, predict the result and rerun. Record the environment, seed, inputs and failures. Do not extrapolate teaching results into production conclusions.')))
    return nb
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--only',choices=list(BOOKS));args=ap.parse_args();DEST.mkdir(exist_ok=True)
    for name in ([args.only] if args.only else BOOKS):
        en,zh=create(name,'en'),create(name,'zh-CN')
        if args.execute:
            print('Executing',name,flush=True)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            namespace={'__name__':'__main__'}
            count=0
            for cell in en.cells:
                if cell.cell_type!='code':continue
                count+=1;cell.execution_count=count;cell.outputs=[]
                def show(*args,**kwargs):
                    for num in plt.get_fignums():
                        buf=io.BytesIO();plt.figure(num).savefig(buf,format='png',dpi=110,bbox_inches='tight')
                        cell.outputs.append(nbf.v4.new_output('display_data',data={'image/png':base64.b64encode(buf.getvalue()).decode(),'text/plain':'Actual matplotlib output'}))
                    plt.close('all')
                plt.show=show
                stdout,stderr=io.StringIO(),io.StringIO()
                with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                    exec(compile(cell.source,f'{name}-cell-{count}','exec'),namespace)
                if stdout.getvalue():cell.outputs.append(nbf.v4.new_output('stream',name='stdout',text=stdout.getvalue()))
                if stderr.getvalue():cell.outputs.append(nbf.v4.new_output('stream',name='stderr',text=stderr.getvalue()))
            for a,b in zip(en.cells,zh.cells):
                if a.cell_type=='code':b.outputs=copy.deepcopy(a.outputs);b.execution_count=a.execution_count
            for nb,locale in [(en,'en'),(zh,'zh-CN')]:
                nb.metadata.learning_lab.execution_status='executed on CPU in authoring environment; CUDA path not tested'
                nb.cells[0].source=nb.cells[0].source.replace('Execution status: not yet run. Run all cells to obtain actual results.','Execution status: executed on CPU during authoring. Outputs below are real; CUDA/TPU paths were not tested. Rerun to obtain results for your environment.').replace('执行状态：尚未运行。使用「全部运行」生成实际结果。','执行状态：编写时已在 CPU 上运行，下方为真实输出；CUDA/TPU 路径未经测试。请重新运行以获取你的环境结果。')
        for nb,locale in [(en,'en'),(zh,'zh-CN')]:
            nbf.validate(nb);nbf.write(nb,DEST/f'{name}-{locale}.ipynb')
        print('Wrote',name,flush=True)
