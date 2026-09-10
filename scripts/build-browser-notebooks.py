"""Generate paired, dependency-free notebooks executed in the embedded Pyodide kernel."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
books={}
def book(key,zh,en,cells): books[key]=(zh,en,cells)
book('llm-model-io','从字符到注意力，再亲手训练一个语言模型','From characters to attention: train a small language model',[
('1 · 建立词表和张量','1 · Build a vocabulary and tensors','字符切分只是教学 tokenizer，不是 BPE。修改 text，观察词表大小和序列长度。后续单元使用这里的变量。','Character splitting is a teaching tokenizer, not BPE. Edit text and inspect vocabulary size and sequence length. Later cells reuse these variables.', '''import math, random
random.seed(7)
text = "hello model. hello world. hello model. "
vocab = sorted(set(text))
to_id = {c: i for i, c in enumerate(vocab)}
ids = [to_id[c] for c in text]
V, D = len(vocab), 4
embedding = [[random.uniform(-1,1) for _ in range(D)] for _ in vocab]
x = [embedding[i] for i in ids[:8]]
print("vocab:", to_id)
print("input IDs:", ids[:8])
print("embedding shape:", (len(x), D))
print("first vector:", x[0])'''),
('2 · 因果注意力','2 · Causal attention','为简化观察，Q、K、V 都直接用 embedding；真实层各有可训练投影。第一行只能看自己。','For inspection Q, K and V all use embeddings; real layers use separate learned projections. The first row can only read itself.', '''def softmax(z):
    e = [math.exp(v-max(z)) for v in z]
    return [v/sum(e) for v in e]
A, hidden = [], []
for i, q in enumerate(x):
    scores = [sum(a*b for a,b in zip(q,k))/D**0.5 for k in x[:i+1]]
    a = softmax(scores) + [0.0]*(len(x)-i-1)
    A.append(a)
    hidden.append([sum(a[j]*x[j][d] for j in range(len(x))) for d in range(D)])
print("attention shape:", (len(A),len(A)))
print("first row:", A[0])
print("hidden shape:", (len(hidden), D))
assert A[0] == [1.0]+[0.0]*(len(x)-1)
display_plot(list(range(len(A))), A[-1], "Last query attention", "key position", "weight")'''),
('3 · 真实梯度训练','3 · Real gradient training','现在训练独立的 bigram 模型：仅凭当前字符预测下一个字符。它不使用上一格 attention，也不是 Transformer；这样可以完整看清交叉熵梯度。','Train a separate bigram model that predicts the next character from the current one. It does not use the previous attention cell and is not a Transformer; the complete cross-entropy gradient is visible.', '''W = [[0.0]*V for _ in range(V)]
pairs = list(zip(ids[:-1], ids[1:]))
learning_rate, epochs = 4.0, 120
losses = []
for epoch in range(epochs):
    grad = [[0.0]*V for _ in range(V)]
    loss = 0.0
    for source, target in pairs:
        p = softmax(W[source])
        loss -= math.log(max(p[target],1e-12))/len(pairs)
        for j in range(V):
            grad[source][j] += (p[j]-(j==target))/len(pairs)
    for i in range(V):
        for j in range(V):
            W[i][j] -= learning_rate*grad[i][j]
    losses.append(loss)
print("parameters:", V*V)
print("initial / final training NLL:", losses[0], losses[-1])
print("This measures training fit, not held-out generalization.")
display_plot(list(range(epochs)), losses, "Training cross-entropy", "epoch", "NLL")'''),
('4 · 输入输出与采样','4 · Inputs, outputs and sampling','修改 temperature 对比概率；模型只记住相邻字符的统计关系，不会理解世界。采样种子固定，便于比较。','Change temperature and compare probabilities. The model learns neighboring-character statistics, not world knowledge. The sampling seed is fixed for comparison.', '''temperature = 0.7
assert temperature > 0
random.seed(11)
current = to_id['h']
generated = [vocab[current]]
print("logits shape:", (1, V))
p = softmax([v/temperature for v in W[current]])
print("top next characters:", sorted(zip(vocab,p),key=lambda z:-z[1])[:5])
for _ in range(60):
    p = softmax([v/temperature for v in W[current]])
    current = random.choices(range(V), weights=p)[0]
    generated.append(vocab[current])
print("".join(generated))''')])
book('infra-benchmark','推理资源账本与真实本机计时','Serving resource accounting and actual local timing',[
('1 · KV 显存账本','1 · Account for KV memory','K、V 各一份；head 数指 KV heads。这里是估算，不是 GPU 显存采样。','Count both K and V; heads means KV heads. This is an estimate, not a measurement of GPU memory.', '''layers, kv_heads, head_dim = 32, 8, 128
bytes_per_element, concurrency = 2, 8
def kv_gib(tokens):
    return 2*layers*kv_heads*head_dim*bytes_per_element*concurrency*tokens/2**30
lengths = [512,1024,2048,4096,8192]
for n in lengths:
    print(n, "tokens =>", kv_gib(n), "GiB")
display_plot(lengths, [kv_gib(n) for n in lengths], "KV estimate", "tokens / request", "GiB")'''),
('2 · TP 分片真的保持结果吗','2 · Does tensor sharding preserve results?','这个例子展示沿输入维度分片，局部乘积必须求和。输出维度分片则拼接结果，两者不同。','This splits the input dimension, requiring a sum of partial products. Output-dimension sharding concatenates results instead.', '''x = [2,3,5,7]
W = [[1,2],[3,4],[5,6],[7,8]]
full = [sum(x[i]*W[i][j] for i in range(4)) for j in range(2)]
parts = [[sum(x[i]*W[i][j] for i in shard) for j in range(2)] for shard in ([0,1],[2,3])]
merged = [sum(p[j] for p in parts) for j in range(2)]
print("local contributions:",parts)
print("all-reduce sum:",merged, "reference:",full)
assert merged == full'''),
('3 · ZeRO 训练状态','3 · ZeRO training states','16 bytes/参数 是此处混合精度 Adam 的约定：2 权重、2 梯度、4 master weight、8 moments；不含 activation。','16 bytes/parameter is this mixed-precision Adam convention: 2 weights, 2 gradients, 4 master weights, 8 moments. Activations are excluded.', '''P, ranks = 7_000_000_000, 8
for stage in range(4):
    weights = 2*P/(ranks if stage>=3 else 1)
    gradients = 2*P/(ranks if stage>=2 else 1)
    optimizer = 12*P/(ranks if stage>=1 else 1)
    print("ZeRO",stage,"persistent GiB/rank:",(weights+gradients+optimizer)/2**30)
print("Peak also includes activations, temporary gathered parameters and buffers.")'''),
('4 · 真实浏览器 CPU 计时','4 · Actual browser CPU timing','测的是 Python 循环，不是 CUDA、BLAS 或 TPU。固定输入，先预热再重复；修改 n 观察增长。不要拿此结果对比推理框架。','This measures Python loops, not CUDA, BLAS or TPU. Fix inputs, warm up, then repeat. Change n to observe growth; do not compare serving frameworks with these numbers.', '''import time, statistics
n = 24
matrix = [[(i+j)%7/7 for j in range(n)] for i in range(n)]
def matmul():
    return [[sum(matrix[i][k]*matrix[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
matmul()
times = []
for _ in range(12):
    start = time.perf_counter()
    out = matmul()
    times.append((time.perf_counter()-start)*1000)
print("shape:",(n,n),"checksum:",sum(map(sum,out)))
print("median ms:",statistics.median(times),"max ms:",max(times))
display_plot(list(range(1,13)),times,"Measured Python matmul","repeat","ms")''')])
book('multimodal-inspection','图像分块、对比学习与可训练重建基线','Image patches, contrastive learning and a trainable reconstruction baseline',[
('1 · 像素与可逆分块','1 · Pixels and reversible patches','用合成灰度图把维度压到能手算。修改 patch，检查整除关系。','Use synthetic grayscale pixels small enough to inspect. Change patch size and check divisibility.', '''import math, random
side, patch = 8, 2
assert side % patch == 0
pixels = [[(x+y)/(2*(side-1)) for x in range(side)] for y in range(side)]
patches = [[pixels[y+dy][x+dx] for dy in range(patch) for dx in range(patch)] for y in range(0,side,patch) for x in range(0,side,patch)]
print("image shape:",(side,side),"patch tensor:",(len(patches),patch*patch))
rebuilt = [[0.0]*side for _ in range(side)]
for i,p in enumerate(patches):
    y,x = (i//(side//patch))*patch,(i%(side//patch))*patch
    for j,value in enumerate(p):
        rebuilt[y+j//patch][x+j%patch] = value
assert rebuilt == pixels
print("patchify -> unpatchify: exact round trip")'''),
('2 · 对比学习的二维玩具例子','2 · A two-dimensional contrastive example','手工向量演示归一化与图文双向 loss，不是 CLIP 的预训练输出。把配对次序交换，观察 loss 增大。','Handmade vectors demonstrate normalization and bidirectional loss; these are not pretrained CLIP outputs. Swap the pair order and observe the loss.', '''images, texts = [[1,0],[0,1]], [[0.9,0.1],[0.2,0.8]]
def unit(v):
    return [x/sum(z*z for z in v)**0.5 for x in v]
images,texts = [unit(v) for v in images],[unit(v) for v in texts]
temperature = 0.1
S = [[sum(x*y for x,y in zip(a,b))/temperature for b in texts] for a in images]
def ce(row,target):
    m=max(row)
    return math.log(sum(math.exp(x-m) for x in row))+m-row[target]
loss = (sum(ce(S[i],i) for i in range(2))+sum(ce([S[j][i] for j in range(2)],i) for i in range(2)))/4
print("scaled similarity:",S)
print("symmetric contrastive loss:",loss)'''),
('3 · 隐藏像素，训练重建器','3 · Hide pixels and train a reconstructor','这里训练 z=a*x+b*y+c，只看可见像素，评估隐藏像素。它是可解释的重建基线，不是 MAE；MAE 的完整权重实验保留在 PyTorch Notebook。','Train z=a*x+b*y+c on visible pixels and evaluate hidden pixels. This is an interpretable reconstruction baseline, not MAE; the full pretrained MAE experiment is in the PyTorch notebook.', '''random.seed(19)
mask_ratio = 0.75
assert 0 < mask_ratio < 1
coords = [(x/(side-1),y/(side-1),pixels[y][x]) for y in range(side) for x in range(side)]
masked = set(random.sample(range(side*side),int(side*side*mask_ratio)))
train = [c for i,c in enumerate(coords) if i not in masked]
w = [0.0,0.0,0.0]
losses=[]
for epoch in range(250):
    grad=[0.0]*3
    for x,y,z in train:
        err = w[0]*x+w[1]*y+w[2]-z
        for j,f in enumerate([x,y,1]):
            grad[j] += 2*err*f/len(train)
    w = [a-0.2*g for a,g in zip(w,grad)]
    mse = sum((w[0]*coords[i][0]+w[1]*coords[i][1]+w[2]-coords[i][2])**2 for i in masked)/len(masked)
    losses.append(mse)
print("visible / masked:",len(train),len(masked))
print("learned coefficients:",w)
print("masked MSE:",losses[-1])
display_plot(list(range(250)),losses,"Held-out pixel reconstruction","epoch","masked MSE")'''),
('4 · 分布变化会怎样','4 · What happens under a distribution change?','把平滑渐变换成棋盘格，线性函数不能表达纹理；这展示低 loss 的结论依赖数据分布。','Replace the smooth gradient with a checkerboard. A linear function cannot represent the texture; a low-loss conclusion depends on the data distribution.', '''checker = [(x+y)%2 for y in range(side) for x in range(side)]
prediction = [w[0]*x+w[1]*y+w[2] for x,y,z in coords]
print("gradient-image MSE:",sum((p-c[2])**2 for p,c in zip(prediction,coords))/len(coords))
print("checkerboard MSE:",sum((p-z)**2 for p,z in zip(prediction,checker))/len(coords))
display_plot(list(range(side)),prediction[:side],"Predicted first image row","column","intensity")''')])
book('hstu-from-scratch','检查 HSTU 聚合，再训练推荐打分头','Inspect HSTU-style aggregation and train a recommendation head',[
('1 · 时序数据与标签','1 · Temporal data and targets','使用三个物品的循环合成序列，物品 ID 从 0 开始且没有 padding。该任务容易，不代表真实用户泛化。','Use synthetic cycles of three items, with IDs starting at zero and no padding. This easy task does not establish generalization to real users.', '''import math, random
random.seed(23)
K, D = 3, 3
E = [[float(i==j) for j in range(D)] for i in range(K)]
examples = [([(start+j)%K for j in range(length)],(start+length)%K) for start in range(K) for length in range(1,7)]
print("example history -> target:",examples[:4])
print("embedding shape:",(K,D))'''),
('2 · 带符号的 SiLU 聚合','2 · Signed SiLU aggregation','使用固定 one-hot embedding、时间距离偏置和残差；省略标准 HSTU 的完整投影、归一化与门控，只用于检查聚合。','Use fixed one-hot embeddings, a distance bias and a residual. Full HSTU projections, normalization and gating are omitted so aggregation is easy to inspect.', '''def silu(x): return x/(1+math.exp(-x))
def features(history):
    q=E[history[-1]]
    scores=[sum(a*b for a,b in zip(q,E[item]))-0.3*(len(history)-1-j) for j,item in enumerate(history)]
    weights=[silu(s)/len(history) for s in scores]
    out=[q[d]+sum(a*E[item][d] for a,item in zip(weights,history)) for d in range(D)]
    return out,weights
h,weights=features([0,1,2,0])
print("weights:",weights,"sum:",sum(weights))
print("user vector:",h)
print("Negative weights are allowed: these are not probabilities.")
display_plot(list(range(len(weights))),weights,"Signed aggregation","history position","weight")'''),
('3 · 训练打分头','3 · Train a scoring head','聚合器固定，仅训练 D×K 个打分参数。梯度是 h*(p-y)。完整端到端 HSTU-inspired 训练在 PyTorch Notebook 中。','Keep the aggregator fixed and train only D×K scoring parameters. The gradient is h*(p-y). Full end-to-end HSTU-inspired training is in the PyTorch notebook.', '''W=[[random.uniform(-0.1,0.1) for _ in range(K)] for _ in range(D)]
def softmax(z):
    e=[math.exp(x-max(z)) for x in z]
    return [x/sum(e) for x in e]
def predict(history):
    h,_=features(history)
    return softmax([sum(h[d]*W[d][j] for d in range(D)) for j in range(K)])
losses=[]
for epoch in range(180):
    grad=[[0.0]*K for _ in range(D)]
    loss=0.0
    for history,target in examples:
        h,_=features(history)
        p=predict(history)
        loss-=math.log(max(p[target],1e-12))/len(examples)
        for d in range(D):
            for j in range(K):
                grad[d][j]+=h[d]*(p[j]-(j==target))/len(examples)
    for d in range(D):
        for j in range(K): W[d][j]-=0.8*grad[d][j]
    losses.append(loss)
print("trained parameters:",D*K)
print("training loss:",losses[0],"->",losses[-1])
display_plot(list(range(180)),losses,"Head training loss","epoch","NLL")'''),
('4 · 换历史，查看排序','4 · Edit the history and inspect ranking','修改 history，检查输出顺序。长序列测试仍来自相同循环机制，不是独立生产数据。','Edit history and inspect the ranking. Longer-sequence tests still follow the same synthetic cycle; they are not independent production data.', '''history=[0,1,2,0]
assert history and all(0<=item<K for item in history)
p=predict(history)
print("history:",history)
print("item probabilities:",p)
print("ranking:",sorted(range(K),key=lambda j:-p[j]))
tests=[([(start+j)%K for j in range(8)],(start+8)%K) for start in range(K)]
hits=sum(max(range(K),key=lambda j:predict(h)[j])==target for h,target in tests)
print("Synthetic longer-sequence HR@1:",hits/len(tests))''')])
for key,(zh,en,sections) in books.items():
 for locale,title in [('zh-CN',zh),('en',en)]:
  cells=[{'cell_type':'markdown','metadata':{},'source':[f'# {title}\n\n', '浏览器 Python 实验：按顺序运行，变量在单元之间共享。图表来自当前代码计算。\n' if locale=='zh-CN' else 'Browser Python lab: run in order; cells share variables. Charts come from the code you execute.\n']}]
  for i,(zhh,enh,zhp,enp,code) in enumerate(sections):
   if i == 0:
    code = "# The blog provides live charts; standalone Python prints chart data.\nif 'display_plot' not in globals():\n    def display_plot(x, y, title='', xlabel='', ylabel=''):\n        print(title, list(zip(x,y)))\n\n" + code
   cells.append({'cell_type':'markdown','metadata':{},'source':[f'## {zhh if locale=="zh-CN" else enh}\n\n',zhp if locale=='zh-CN' else enp]})
   cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)})
  if key == 'llm-model-io':
   cells.append({'cell_type':'markdown','metadata':{},'source':[
    '## 5 · 可选：在这个 Notebook 加载真实 MiniLM\n\n首次单独运行此格会下载约 23 MB 权重及推理库，需要访问 Hugging Face 和 jsDelivr。真实推理在本机执行；这是英文 embedding 模型，不是上面训练的 bigram。此格不包含在“运行全部”中，需要博客提供的 browser_models 接口。修改两句话，重新运行，观察真实 token、mask 与 384 维输出。' if locale=='zh-CN' else
    '## 5 · Optional: load real MiniLM inside this notebook\n\nRunning this cell separately downloads about 23 MB of weights plus runtime from Hugging Face and jsDelivr. Inference runs locally. This is an English embedding model, separate from the bigram trained above. Run all excludes this cell; it requires the blog-provided browser_models bridge. Edit the two sentences and inspect real tokens, masks and 384-dimensional outputs.'
   ]})
   cells.append({'cell_type':'code','metadata':{'tags':['requires-model']},'execution_count':None,'outputs':[],'source':"""import json
from browser_models import embed
texts = ["A cat is sleeping on the sofa.", "A kitten is resting on a couch."]
result = json.loads(await embed(texts))
print("actual input IDs:", result['ids'])
print("actual tokens:", result['tokens'])
print("attention masks:", result['masks'])
print("output shape:", result['dims'])
print("cosine similarity:", result['cosine'])
vector = result['embeddings'][0]
display_plot(list(range(32)), vector[:32], "Real MiniLM embedding (first 32 dimensions)", "dimension", "value")
""".splitlines(True)})
  nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python (browser)','language':'python','name':'python3'},'language_info':{'name':'python'},'learning_runtime':'pyodide-stdlib'},'cells':cells}
  # Stable cell ids make this a valid nbformat 4.5 notebook.
  for i,c in enumerate(cells): c['id']=f'cell-{i}'
  (ROOT/'public/notebooks'/f'{key}-browser-{locale}.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
print('Generated 8 browser notebooks, 16 standard-library cells + 1 optional pretrained-model cell.')
