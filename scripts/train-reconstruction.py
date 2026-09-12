"""CPU reproduction: python scripts/train-reconstruction.py.
Requires torch, scikit-learn, numpy. No image/class labels enter AE training.
Data: sklearn's bundled UCI Optical Recognition of Handwritten Digits (8x8).
"""
import json, pathlib, random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

torch.manual_seed(73);random.seed(73);np.random.seed(73);torch.set_num_threads(2)
data=load_digits();train,test=train_test_split(np.arange(len(data.data)),test_size=.2,stratify=data.target,random_state=73)
x=torch.tensor(data.data/16,dtype=torch.float32);y=torch.tensor(data.target)
class AE(nn.Module):
 def __init__(self):
  super().__init__();self.e1=nn.Linear(80,96);self.e2=nn.Linear(96,24);self.d1=nn.Linear(24,96);self.d2=nn.Linear(96,64)
 def encode(self,x,mask):
  visible=(1-mask).reshape(-1,4,4).repeat_interleave(2,1).repeat_interleave(2,2).reshape(-1,64)
  return torch.tanh(self.e2(F.relu(self.e1(torch.cat([x*visible,1-mask],-1)))))
 def decode(self,z):return torch.sigmoid(self.d2(F.relu(self.d1(z))))
 def forward(self,x,mask):return self.decode(self.encode(x,mask))
def masks(n):
 counts=torch.randint(0,14,(n,1));order=torch.rand(n,16).argsort(1).argsort(1)
 return (order<counts).float()
def pixel_mask(mask):return mask.reshape(-1,4,4).repeat_interleave(2,1).repeat_interleave(2,2).reshape(-1,64)
model=AE();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
for step in range(3500):
 ids=torch.tensor(np.random.choice(train,128));batch=x[ids];mask=masks(128);pm=pixel_mask(mask);pred=model(batch,mask);err=(pred-batch)**2
 missing=(err*pm).sum(1)/pm.sum(1).clamp_min(1);visible=(err*(1-pm)).sum(1)/(1-pm).sum(1).clamp_min(1)
 loss=(missing+.1*visible).mean();opt.zero_grad();loss.backward();opt.step()
 if step%1000==0:print('reconstruction',step,round(loss.item(),4),flush=True)
model.eval()
for param in model.parameters():param.requires_grad_(False)
probe=nn.Linear(24,10);opt=torch.optim.AdamW(probe.parameters(),lr=.01,weight_decay=.001)
for step in range(1600):
 ids=torch.tensor(np.random.choice(train,128));mask=masks(128)
 if step%2==0:mask.zero_()
 with torch.no_grad():z=model.encode(x[ids],mask)
 loss=F.cross_entropy(probe(z),y[ids]);opt.zero_grad();loss.backward();opt.step()
with torch.no_grad():
 clean=torch.zeros(len(test),16);half=(torch.rand(len(test),16).argsort(1).argsort(1)<8).float();pm=pixel_mask(half);pred=model(x[test],half)
 metrics={'testCount':len(test),'trainCount':len(train),'cleanProbeAccuracy':(probe(model.encode(x[test],clean)).argmax(-1)==y[test]).float().mean().item(),'halfMaskedProbeAccuracy':(probe(model.encode(x[test],half)).argmax(-1)==y[test]).float().mean().item(),'halfMaskedMSE':(((pred-x[test])**2*pm).sum()/pm.sum()).item(),'meanFillMSE':(((x[train].mean(0)[None]-x[test])**2*pm).sum()/pm.sum()).item()}
 ztrain=model.encode(x[train],torch.zeros(len(train),16)).numpy();pca=PCA(2).fit(ztrain)
 gallery=[]
 for label in range(10):
  for idx in [i for i in test if data.target[i]==label][:2]:gallery.append({'id':int(idx),'label':label,'pixels':x[idx].tolist()})
 cloud=[{'label':int(data.target[i]),'xy':pca.transform(model.encode(x[i:i+1],torch.zeros(1,16)).numpy())[0].tolist()} for i in test[:180]]
 meta={'name':'Reconstruct-24','seed':73,'pixels':64,'patchSize':2,'patches':16,'latent':24,'steps':3500,'probeSteps':1600,'parameters':sum(p.numel() for p in model.parameters()),'probeParameters':sum(p.numel() for p in probe.parameters()),'metrics':metrics,'architecture':'MLP 80→96 ReLU→24 tanh→96 ReLU→64 sigmoid; binary visibility concatenated to masked pixels. This is a masked autoencoder, NOT the ViT MAE architecture.','training':'AE sees no class labels. Mean masked-pixel MSE + 0.1 mean visible-pixel MSE, random 0–13 of 16 patches masked. Frozen encoder; linear classifier subsequently trained using digit labels, on alternating clean and masked training inputs. Test split untouched by optimization.','dataset':'UCI Optical Recognition of Handwritten Digits, sklearn load_digits; 1,797 images, grayscale 8×8. Seeded stratified 80/20 image split, not writer-disjoint.','torchVersion':torch.__version__}
 weights={k:v.tolist() for k,v in model.state_dict().items()};weights.update({'probe.'+k:v.tolist() for k,v in probe.state_dict().items()})
 artifact={'meta':meta,'weights':weights,'meanImage':x[train].mean(0).tolist(),'pca':{'mean':pca.mean_.tolist(),'components':pca.components_.tolist(),'variance':pca.explained_variance_ratio_.tolist()},'cloud':cloud,'gallery':gallery}
 root=pathlib.Path('public/models/reconstruction');root.mkdir(parents=True,exist_ok=True);(root/'model.json').write_text(json.dumps(artifact,separators=(',',':')));(root/'model-card.json').write_text(json.dumps(meta,indent=2))
 fixtures=[]
 for sample in gallery[::4]:
  for mask in [torch.zeros(1,16),half[:1],torch.ones(1,16)]:
   inp=torch.tensor([sample['pixels']]);z=model.encode(inp,mask);reconstruction=model.decode(z)
   fixtures.append({'pixels':sample['pixels'],'mask':mask[0].tolist(),'latent':z[0].tolist(),'reconstruction':reconstruction[0].tolist(),'probabilities':probe(z).softmax(-1)[0].tolist(),'projection':pca.transform(z.numpy())[0].tolist()})
 pathlib.Path('tests/fixtures/reconstruction-reference.json').write_text(json.dumps(fixtures,separators=(',',':')))
 print(json.dumps(meta,indent=2),flush=True)
