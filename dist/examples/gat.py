"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python gat.py --steps 20 --device auto
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

# Dense single-head additive GAT, self-loops, small synthetic graph.
A=torch.tensor([[1,1,0,0,0,0],[1,1,1,0,0,0],[0,1,1,1,0,0],[0,0,1,1,1,0],[0,0,0,1,1,1],[0,0,0,0,1,1]],device=device,dtype=torch.bool)
x=torch.randn(6,4,device=device);y=torch.tensor([0,0,0,1,1,1],device=device);train_idx=torch.tensor([0,1,3,4],device=device)
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(4,8,bias=False);self.score=nn.Parameter(torch.randn(16)*.1);self.head=nn.Linear(8,2)
    def forward(self,x):
        h=self.proj(x);pairs=torch.cat([h[:,None,:].expand(-1,6,-1),h[None,:,:].expand(6,-1,-1)],-1)
        scores=F.leaky_relu(pairs@self.score,.2).masked_fill(~A,float('-inf'))
        attention=scores.softmax(-1)
        return self.head(F.elu(attention@h))
model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=.02)
for _ in range(args.steps):
    opt.zero_grad();loss=F.cross_entropy(model(x)[train_idx],y[train_idx]);loss.backward();opt.step()

if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
print(json.dumps({'example':'gat','device':str(device),'parameter_updates':args.steps,'final_loss':loss.item(),'scope':'Teaching core on synthetic data; finite loss is not evidence of benchmark quality'}))
