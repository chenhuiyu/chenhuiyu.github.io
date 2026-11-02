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
INFRA_SETUP='''import sys, subprocess, importlib.util
requirements = {'torch': 'torch>=2.6,<3', 'numpy': 'numpy', 'matplotlib': 'matplotlib'}
missing = [package for module, package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
import torch, numpy as np, matplotlib.pyplot as plt
print({'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.cuda.is_available()})
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
'无需模型权重。前半验证 TP、ZeRO 和 matmul（CUDA 可用时 matmul 会选择 GPU）；第 5–9 节固定 CPU FP32，逐步实现两层 Decoder 的 KV cache，检查 FP32/FP64 正确性、错误对照和续算计时。GPU/TPU 与框架性能不在该 KV 实验的结论范围内。',
'No model weights required. The first half verifies TP, ZeRO and matmul (matmul selects CUDA when available). Sections 5–9 fix CPU FP32, implement a two-layer decoder KV cache, check FP32/FP64 equivalence and negative controls, then time continuation. The KV experiment makes no GPU/TPU or serving-framework performance claims.',[
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
'''),
section('## 5 · 从零实现两层 Decoder\n固定 seed、CPU FP32、两层 pre-norm、4 个 MHA heads、learned positions，无 dropout。模型未训练，只检验计算一致性。每层独立缓存投影后的 K/V；动态拼接保持旧 cache 不变。',
'## 5 · Build a two-layer decoder\nFixed seed, CPU FP32, two pre-norm blocks, four MHA heads, learned positions and no dropout. Untrained weights test computational equivalence, not language quality. Each layer caches its own projected K/V; concatenation leaves the old cache unchanged.', '''import math
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)
torch.set_num_threads(1)

class CacheBlock(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        assert width % heads == 0
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.out = nn.Linear(width, 3*width), nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x, past=None, offset_mask=True):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = [t.reshape(b, n, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v)]
        past_len = 0 if past is None else past[0].shape[-2]
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
        # Local query i corresponds to absolute position past_len + i.
        query_pos = torch.arange(n) + (past_len if offset_mask else 0)
        key_pos = torch.arange(k.shape[-2])
        allowed = key_pos[None, :] <= query_pos[:, None]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = scores.masked_fill(~allowed, -torch.inf).softmax(dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.out(context)
        x = x + self.ff(self.ln2(x))
        return x, (k, v)

class CacheDecoder(nn.Module):
    def __init__(self, vocab=97, width=64, heads=4, layers=2, max_positions=512):
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(max_positions, width)
        self.blocks = nn.ModuleList([CacheBlock(width, heads) for _ in range(layers)])
        self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, vocab, bias=False)

    def forward(self, ids, cache=None, use_cache=True, reset_positions=False, offset_mask=True):
        past_len = 0 if cache is None else cache[0][0].shape[-2]
        if cache is not None:
            assert len(cache) == len(self.blocks)
            assert all(k.shape[-2] == past_len == v.shape[-2] for k, v in cache)
        start = 0 if reset_positions else past_len
        positions = torch.arange(start, start + ids.shape[1])
        x = self.token(ids) + self.position(positions)[None, :, :]
        updated = []
        for layer, block in enumerate(self.blocks):
            x, kv = block(x, None if cache is None else cache[layer], offset_mask=offset_mask)
            if use_cache:
                updated.append(kv)
        return self.head(self.norm(x)), tuple(updated)

decoder = CacheDecoder().cpu().float().eval()
fixed_ids = torch.randint(0, 97, (2, 288))
print('parameters:', sum(p.numel() for p in decoder.parameters()))
with torch.inference_mode():
    prefill_logits, prefix_cache = decoder(fixed_ids[:, :5])
print('prefill logits:', tuple(prefill_logits.shape))
print('per-layer K/V:', [(tuple(k.shape), tuple(v.shape)) for k, v in prefix_cache])
'''),
section('## 6 · 逐位置比较 logits 与每层 K/V\n对 batch=1/2、chunk=1/2/5 和 FP32/FP64 分别检查全部 24 个位置。断言检查全部词表 logits，而不是只比较 argmax；同时检查因果性、cache 长度和旧 cache 未被原地修改。',
'## 6 · Compare every logit and layer cache\nCheck all 24 positions for batch sizes 1/2, chunks 1/2/5 and FP32/FP64. Assert every vocabulary logit, not only argmax; also check causality, cache length and that old caches are not mutated.', '''from copy import deepcopy
correctness = []
with torch.inference_mode():
    for dtype, atol in [(torch.float32, 1e-5), (torch.float64, 1e-10)]:
        checked = deepcopy(decoder).to(dtype=dtype)
        for batch in [1, 2]:
            tokens = fixed_ids[:batch, :24]
            full, full_cache = checked(tokens)
            changed = tokens.clone()
            changed[:, -1] = (changed[:, -1] + 1) % 97
            alternate, _ = checked(changed, use_cache=False)
            torch.testing.assert_close(full[:, :-1], alternate[:, :-1], atol=atol, rtol=atol)
            for chunk in [1, 2, 5]:
                cache, outputs, errors = None, [], []
                for start in range(0, tokens.shape[1], chunk):
                    end = min(start + chunk, tokens.shape[1])
                    old = None if cache is None else tuple((k.clone(), v.clone()) for k, v in cache)
                    logits, new_cache = checked(tokens[:, start:end], cache)
                    reference, reference_cache = checked(tokens[:, :end])
                    torch.testing.assert_close(logits, reference[:, start:end], atol=atol, rtol=atol)
                    for layer, (k, v) in enumerate(new_cache):
                        assert k.shape == v.shape == (batch, 4, end, 16)
                        for value, target in zip((k, v), reference_cache[layer]):
                            torch.testing.assert_close(value, target, atol=atol, rtol=atol)
                        if old is not None:
                            assert torch.equal(cache[layer][0], old[layer][0])
                            assert torch.equal(cache[layer][1], old[layer][1])
                    outputs.append(logits)
                    errors.extend((logits-reference[:, start:end]).abs().amax(dim=(0, 2)).tolist())
                    cache = new_cache
                torch.testing.assert_close(torch.cat(outputs, dim=1), full, atol=atol, rtol=atol)
                row = {'dtype': str(dtype), 'batch': batch, 'chunk': chunk, 'max_abs_logit_error': max(errors)}
                correctness.append(row)
                print(row)
                if dtype == torch.float32 and batch == 2 and chunk == 1:
                    position_errors = errors
plt.figure(figsize=(8, 3.6))
plt.plot(range(1, 25), position_errors, marker='o')
plt.xlabel('Consumed prefix length'); plt.ylabel('Max absolute logit error')
plt.title('CPU FP32: full recompute vs token-by-token cache (B=2)')
plt.grid(alpha=.25); plt.tight_layout(); plt.show()
'''),
section('## 7 · 故意写错位置与 mask\ncache 长度为 4，当前 chunk 为 2：query 的绝对位置是 4、5。将位置重置为 0，或使用左上角三角 mask，都会改变输出；反例必须明显偏离正确参考。',
'## 7 · Deliberately break positions and masks\nWith four cached tokens and two new tokens, query positions are 4 and 5. Resetting positions to zero or using an upper-left triangle changes outputs; negative controls must visibly disagree with the reference.', '''with torch.inference_mode():
    tokens = fixed_ids[:1, :6]
    reference, _ = decoder(tokens, use_cache=False)
    _, cache = decoder(tokens[:, :4])
    negative_controls = {}
    for name, options in [('reset_positions', {'reset_positions': True}), ('unshifted_mask', {'offset_mask': False})]:
        wrong, _ = decoder(tokens[:, 4:], cache, **options)
        error = (wrong-reference[:, 4:]).abs().max().item()
        assert error > 1e-3
        negative_controls[name] = error
        print(name, 'max absolute logit error:', error)
past_len, new_len = 4, 2
keys = torch.arange(past_len + new_len)
correct_mask = keys[None, :] <= (past_len + torch.arange(new_len))[:, None]
wrong_mask = keys[None, :] <= torch.arange(new_len)[:, None]
fig, axes = plt.subplots(1, 2, figsize=(9, 2.7))
for ax, mask, title in zip(axes, [correct_mask, wrong_mask], ['Correct: key <= past + query', 'Broken: key <= local query']):
    ax.imshow(mask.numpy(), cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set(xticks=range(6), yticks=range(2), yticklabels=[4, 5], xlabel='Key absolute position', ylabel='Query absolute position', title=title)
    for i in range(2):
        for j in range(6):
            ax.text(j, i, int(mask[i, j]), ha='center', va='center', color='white' if mask[i, j] else 'black')
plt.tight_layout(); plt.show()
'''),
section('## 8 · 固定 token 流，分开 prefill 与续算计时\nCPU FP32、单线程；每种路径预热 3 次、重复 15 次，交替测量完整重算和缓存。Decode 窗口都消费相同的 16 个预定 token，输出每步全部 logits；prefix prefill 单独计时，cache 创建后复用且不原地修改。动态 cat、分配和 Python 开销包含在内。此窗口不是客户端 TTFT/ITL。',
'## 8 · Time prefill and continuation separately\nCPU FP32, one thread, three warm-ups and 15 repeats per path. Alternate recompute/cached measurement order. Both continuation windows consume the same 16 predetermined tokens and output all logits at each step. Time prefix prefill separately; reuse an immutable prefix cache. Include dynamic cat, allocation and Python overhead. This window is not client TTFT/ITL.', '''import time, platform, os
NEW_TOKENS, REPEATS, WARMUPS = 16, 15, 3
kv_timings = []

def timed_ms(fn):
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1e6

with torch.inference_mode():
    for prompt in [16, 64, 128, 256]:
        stream = fixed_ids[:1, :prompt + NEW_TOKENS]
        _, seed_cache = decoder(stream[:, :prompt])

        def prefill():
            return decoder(stream[:, :prompt])

        def full_continuation():
            rows = []
            for j in range(NEW_TOKENS):
                logits, _ = decoder(stream[:, :prompt+j+1], use_cache=False)
                rows.append(logits[:, -1:])
            return torch.cat(rows, dim=1)

        def cached_continuation():
            cache, rows = seed_cache, []
            for j in range(NEW_TOKENS):
                logits, cache = decoder(stream[:, prompt+j:prompt+j+1], cache)
                rows.append(logits)
            return torch.cat(rows, dim=1), cache

        cached_logits, final_cache = cached_continuation()
        torch.testing.assert_close(cached_logits, full_continuation(), atol=1e-5, rtol=1e-5)
        for _ in range(WARMUPS):
            prefill(); full_continuation(); cached_continuation()
        samples = {'prefill': [], 'full': [], 'cached': []}
        functions = {'prefill': prefill, 'full': full_continuation, 'cached': cached_continuation}
        for repeat in range(REPEATS):
            # Rotate all three paths so no path is always measured first.
            order = ['prefill', 'full', 'cached']
            order = order[repeat % 3:] + order[:repeat % 3]
            for path in order:
                samples[path].append(timed_ms(functions[path]))
        actual_bytes = sum(t.numel()*t.element_size() for pair in final_cache for t in pair)
        expected_bytes = 2 * 1 * (prompt+NEW_TOKENS) * 2 * 4 * 16 * 4
        assert actual_bytes == expected_bytes
        row = {'prompt_tokens': prompt, 'appended_tokens': NEW_TOKENS, 'cache_payload_bytes': actual_bytes,
               'samples_ms': samples, 'full_projected_positions_per_layer': sum(prompt+j+1 for j in range(NEW_TOKENS)),
               'cached_projected_positions_per_layer': NEW_TOKENS}
        for path in samples:
            row[path+'_p50_ms'] = float(np.median(samples[path]))
            row[path+'_p95_ms'] = float(np.percentile(samples[path], 95))
        kv_timings.append(row)
        print({k: v for k, v in row.items() if k != 'samples_ms'})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for path, label in [('full', 'Recompute full prefix'), ('cached', 'Reuse layer KV')]:
    axes[0].plot([r['prompt_tokens'] for r in kv_timings], [r[path+'_p50_ms'] for r in kv_timings], marker='o', label=label)
axes[0].set(xlabel='Prompt tokens', ylabel='16-token continuation p50 (ms)', title='CPU FP32, one thread; prefill excluded')
axes[0].legend(); axes[0].grid(alpha=.25)
axes[1].plot([r['prompt_tokens']+NEW_TOKENS for r in kv_timings], [r['cache_payload_bytes']/1024 for r in kv_timings], marker='o')
axes[1].set(xlabel='Cached tokens after continuation', ylabel='Logical tensor payload (KiB)', title='K + V, two layers; allocator peak excluded')
axes[1].grid(alpha=.25); plt.tight_layout(); plt.show()
'''),
section('## 9 · 保存环境、原始样本与正确性结果\n结果文件包含全部 15 次样本，p95 是小样本经验分位数。逻辑 tensor 字节数不等于进程峰值或设备峰值。模型未训练、无 padding/GQA/RoPE/采样；这里只验证 MHA 与 learned positions 的缓存机制。',
'## 9 · Save environment, raw samples and correctness\nKeep all 15 samples; p95 is a small-sample empirical quantile. Logical tensor bytes are not process or device peak memory. The untrained model uses MHA and learned positions, without padding, GQA, RoPE or sampling.', '''from datetime import datetime, timezone
from pathlib import Path
cpu_name = platform.processor()
if Path('/proc/cpuinfo').is_file():
    cpu_name = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), cpu_name)
kv_cache_report = {
    'scope': 'Untrained two-layer decoder, synthetic fixed-token continuation; CPU microbenchmark only',
    'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
                    'device': 'cpu', 'dtype': 'float32', 'threads': torch.get_num_threads(),
                    'platform': platform.platform(), 'cpu': cpu_name, 'logical_cpus': os.cpu_count(),
                    'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': 42},
    'model': {'vocab': 97, 'layers': 2, 'width': 64, 'heads': 4, 'head_dim': 16, 'max_positions': 512,
              'parameters': sum(p.numel() for p in decoder.parameters()), 'trained': False, 'position_encoding': 'learned absolute'},
    'protocol': {'warmups': WARMUPS, 'repeats': REPEATS, 'timer': 'perf_counter_ns',
                 'continuation_prefill_included': False, 'cache': 'immutable dynamic torch.cat',
                 'gradients': 'torch.inference_mode', 'path_order': 'rotated each repeat',
                 'inputs': 'predetermined tokens, no sampling or EOS', 'memory': 'logical final K/V payload only'},
    'correctness': correctness, 'position_errors_fp32_batch2_chunk1': position_errors,
    'negative_controls': negative_controls, 'timings': kv_timings,
}
report['kv_cache'] = kv_cache_report
with open('infra-benchmark-results.json', 'w') as f:
    json.dump(report, f, indent=2)
print(json.dumps(kv_cache_report['environment'], indent=2))
print('Saved raw samples and checks to infra-benchmark-results.json')
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
    nb.cells=[nbf.v4.new_markdown_cell('# '+(zt if zh else et)+'\n\n'+(zi if zh else ei)+'\n\n'+('执行状态：尚未运行。使用「全部运行」生成实际结果。' if zh else 'Execution status: not yet run. Run all cells to obtain actual results.')),nbf.v4.new_code_cell(INFRA_SETUP if name == 'infra-benchmark' else SETUP)]
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
            if name == 'infra-benchmark':
                # Derive article figures from the executed notebook, never by hand.
                figures = [o for c in en.cells for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]
                for output, filename in zip(figures[-3:], ['kv-cache-correctness.png', 'kv-cache-mask.png', 'kv-cache-cpu-timing.png']):
                    (ROOT/'public/learning'/filename).write_bytes(base64.b64decode(output.data['image/png']))
                (ROOT/'public/learning/kv-cache-results.json').write_text(json.dumps(namespace['kv_cache_report'], indent=2)+'\n')
            for a,b in zip(en.cells,zh.cells):
                if a.cell_type=='code':b.outputs=copy.deepcopy(a.outputs);b.execution_count=a.execution_count
            for nb,locale in [(en,'en'),(zh,'zh-CN')]:
                nb.metadata.learning_lab.execution_status='executed on CPU in authoring environment; CUDA path not tested'
                nb.cells[0].source=nb.cells[0].source.replace('Execution status: not yet run. Run all cells to obtain actual results.','Execution status: executed on CPU during authoring. Outputs below are real; CUDA/TPU paths were not tested. Rerun to obtain results for your environment.').replace('执行状态：尚未运行。使用「全部运行」生成实际结果。','执行状态：编写时已在 CPU 上运行，下方为真实输出；CUDA/TPU 路径未经测试。请重新运行以获取你的环境结果。')
        for nb,locale in [(en,'en'),(zh,'zh-CN')]:
            nbf.validate(nb);nbf.write(nb,DEST/f'{name}-{locale}.ipynb')
        print('Wrote',name,flush=True)
