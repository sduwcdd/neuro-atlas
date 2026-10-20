"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python wav2vec2_core.py --steps 20 --device auto
Requires Python 3.11 and PyTorch; DQN/PPO also need stable-baselines3 and gymnasium.
"""
import argparse, json
import torch
from torch import nn
from torch.nn import functional as F

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=20)
parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
args=parser.parse_args()
if args.steps<1: parser.error('--steps must be positive')
if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
torch.manual_seed(7)
torch.set_num_threads(2)

# Masked contrastive core only. No raw-audio front end, quantizer, diversity loss or CTC fine-tuning.
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.project=nn.Linear(8,16);self.mask=nn.Parameter(torch.zeros(16))
        self.position=nn.Parameter(torch.randn(1,8,16)*.02)
        self.context=nn.TransformerEncoder(nn.TransformerEncoderLayer(16,2,32,dropout=0,batch_first=True),1)
        self.predict=nn.Linear(16,16)
    def forward(self,x):
        z=self.project(x);selected=torch.zeros(x.shape[:2],dtype=torch.bool,device=x.device);selected[:,3]=True
        corrupted=torch.where(selected[:,:,None],self.mask,z)
        c=self.predict(self.context(corrupted+self.position))
        # Candidate 0 is the stop-gradient target at this same position.
        target=z.detach();positions=torch.tensor([3,1,5],device=x.device)
        candidates=F.normalize(target[:,positions,:],dim=-1);query=F.normalize(c[:,3,:],dim=-1)
        return (query[:,None,:]*candidates).sum(-1)/.2
model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=.005)
for _ in range(args.steps):
    phase=torch.rand(16,1,1,device=device)*6.28
    x=torch.sin(phase+torch.arange(8,device=device)[None,:,None]*.3+torch.arange(8,device=device)[None,None,:]*.1)
    opt.zero_grad();logits=model(x);loss=F.cross_entropy(logits,torch.zeros(16,dtype=torch.long,device=device));loss.backward();opt.step()

if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
print(json.dumps({'example':'wav2vec2_core','device':str(device),'parameter_updates':args.steps,'final_loss':loss.item(),'scope':'Teaching core on synthetic data; finite loss is not evidence of benchmark quality'}))
