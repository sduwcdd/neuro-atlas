"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python bert.py --steps 20 --device auto
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

# Simplified MLM only: all selected tokens become MASK; no NSP, no 80/10/10 recipe.
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(13,16);self.pos=nn.Parameter(torch.randn(1,8,16)*.02)
        self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(16,2,32,dropout=0,batch_first=True),1)
        self.head=nn.Linear(16,12)
    def forward(self,x):return self.head(self.enc(self.emb(x)+self.pos))
model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=.01)
for _ in range(args.steps):
    start=torch.randint(0,12,(16,1),device=device);original=(start+torch.arange(8,device=device))%12
    selected=torch.rand(16,8,device=device)<.25;selected[:,3]=True
    corrupted=original.masked_fill(selected,12)
    logits=model(corrupted)
    opt.zero_grad();loss=F.cross_entropy(logits[selected],original[selected]);loss.backward();opt.step()

if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
print(json.dumps({'example':'bert','device':str(device),'parameter_updates':args.steps,'final_loss':loss.item(),'scope':'Teaching core on synthetic data; finite loss is not evidence of benchmark quality'}))
