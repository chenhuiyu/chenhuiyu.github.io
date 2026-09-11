"""Reproduce the local, synthetic associative-recall Transformer. CPU: pip install torch.
Training data is generated here; no external dataset or pretrained weights.
Run from repository root: python scripts/train-recall-transformer.py
"""
import json, random, math, pathlib, time
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)
random.seed(42)
torch.set_num_threads(2)
VOCAB = ['<bos>', '=', ';', 'red', 'blue', 'green', 'gold', 'pink', 'cyan', 'fox', 'owl', 'cat', 'elk', 'bee', 'yak']
D, H, L, CONTEXT = 32, 4, 2, 32
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3*D), nn.Linear(D,D)
        self.fc1, self.fc2 = nn.Linear(D,4*D), nn.Linear(4*D,D)
    def forward(self,x, ablate=-1):
        b,t,c=x.shape
        q,k,v=self.qkv(self.ln1(x)).chunk(3,dim=-1)
        q,k,v=[a.view(b,t,H,D//H).transpose(1,2) for a in (q,k,v)]
        att=(q@k.transpose(-2,-1)/math.sqrt(D//H)).masked_fill(torch.ones(t,t,dtype=torch.bool).triu(1),float('-inf')).softmax(-1)
        heads=att@v
        if ablate>=0:
            heads=heads.clone(); heads[:,ablate]=0
        x=x+self.proj(heads.transpose(1,2).reshape(b,t,D))
        return x+self.fc2(F.gelu(self.fc1(self.ln2(x)),approximate='tanh')),att
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token,self.position=nn.Embedding(len(VOCAB),D),nn.Embedding(CONTEXT,D)
        self.blocks=nn.ModuleList([Block() for _ in range(L)])
        self.ln=nn.LayerNorm(D)
        self.output=nn.Linear(D,len(VOCAB),bias=False)
    def forward(self,ids,ablate=None,trace=False):
        x=self.token(ids)+self.position(torch.arange(ids.shape[1]))
        states=[x];atts=[]
        for i,block in enumerate(self.blocks):
            x,att=block(x,ablate[1] if ablate and ablate[0]==i else -1)
            states.append(x);atts.append(att)
        logits=self.output(self.ln(x))
        return (logits,states,atts) if trace else logits

def batch(size,n):
    rows=[]; answers=[]
    for _ in range(size):
        keys=random.sample(range(3,9),n); values=random.sample(range(9,15),n)
        query=random.randrange(n); row=[0]
        for k,v in zip(keys,values):row.extend([k,1,v,2])
        row.extend([keys[query]]);rows.append(row);answers.append(values[query])
    return torch.tensor(rows),torch.tensor(answers)

model=Model()
def initialize(module):
    if isinstance(module,(nn.Linear,nn.Embedding)):
        nn.init.normal_(module.weight,mean=0.,std=.02)
        if isinstance(module,nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)
model.apply(initialize)
key_probe=nn.Linear(D,6) # Training-only probe; not part of inference.
optimizer=torch.optim.AdamW(list(model.parameters())+list(key_probe.parameters()),lr=.001,weight_decay=.01)
start=time.time()
for step in range(4000):
    ids,y=batch(64,random.choice([2,3,4]))
    logits,states,att=model(ids,trace=True)
    # Transparent auxiliary supervision makes this a controlled teaching model:
    # layer 1 / head 1 reads each value's key; layer 2 / head 1 reads the answer.
    value_positions=torch.arange(3,ids.shape[1]-1,4)
    answer_positions=(ids[:,value_positions]==y[:,None]).long().argmax(-1)*4+3
    binding_loss=-att[0][:,0,value_positions,value_positions-2].clamp_min(1e-8).log().mean()
    recall_loss=-att[1][torch.arange(ids.shape[0]),0,-1,answer_positions].clamp_min(1e-8).log().mean()
    key_positions=torch.cat([value_positions,torch.tensor([ids.shape[1]-1])])
    key_targets=torch.cat([ids[:,value_positions-2]-3,ids[:,-1:]-3],dim=1)
    key_loss=F.cross_entropy(key_probe(states[1][:,key_positions]).reshape(-1,6),key_targets.reshape(-1))
    loss=key_loss+.5*(binding_loss+recall_loss)+F.cross_entropy(logits[:,-1],y)+.15*F.cross_entropy(logits[:,:-1].reshape(-1,len(VOCAB)),ids[:,1:].reshape(-1))
    optimizer.zero_grad();loss.backward();optimizer.step()
    if step%1000==0: print(step,round(loss.item(),4),flush=True)
model.eval()
with torch.no_grad():
    scores={}
    for n in [2,3,4]:
        ids,y=batch(1024,n);scores[str(n)]=round((model(ids)[:,-1].argmax(-1)==y).float().mean().item(),5)
    state={k:v.tolist() for k,v in model.state_dict().items()}
    meta={'name':'Recall-32','seed':42,'steps':4000,'batchSize':64,'dimension':D,'heads':H,'layers':L,'context':CONTEXT,'vocab':VOCAB,'parameters':sum(p.numel() for p in model.parameters()),'validation':scores,'auxiliarySupervision':'Layer 1 head 1 is supervised to attend from each animal to its colour; layer 2 head 1 is supervised to attend from the query to the correct animal. A training-only colour probe aligns first-block animal and query representations. Objective: colour-probe cross-entropy + answer cross-entropy + 0.15 language-model cross-entropy + 0.5 times each attention loss. This is a guided teaching model, not an emergent-circuit claim.','training':'Synthetic key-value recall; 2–4 unique colour/animal bindings. Fresh random validation: 1024 examples per length.','torchVersion':torch.__version__}
    root=pathlib.Path('public/models/recall');root.mkdir(parents=True,exist_ok=True)
    (root/'weights.json').write_text(json.dumps({'meta':meta,'weights':state},separators=(',',':')))
    fixtures=[]
    for sentence in ['red = fox ; blue = owl ; red','red = owl ; blue = fox ; red','gold = bee ; cyan = elk ; pink = yak ; cyan']:
        ids=torch.tensor([[0]+[VOCAB.index(t) for t in sentence.split()]])
        logits,states,atts=model(ids,trace=True)
        fixtures.append({'prompt':sentence,'ids':ids[0].tolist(),'probabilities':logits[0,-1].softmax(-1).tolist(),'states':[s[0].tolist() for s in states],'attention':[a[0].tolist() for a in atts], 'ablations':[model(ids,(l,h))[0,-1].softmax(-1).tolist() for l in range(L) for h in range(H)]})
    pathlib.Path('tests/fixtures').mkdir(exist_ok=True)
    pathlib.Path('tests/fixtures/recall-reference.json').write_text(json.dumps(fixtures,separators=(',',':')))
    (root/'model-card.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2));print('seconds',time.time()-start)
