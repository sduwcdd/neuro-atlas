"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python gcn.py --steps 20 --device auto
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

# Two-layer GCN, synthetic six-node graph, labels only at train_idx.
A=torch.tensor([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]],dtype=torch.float32,device=device)
A=A+torch.eye(6,device=device);inv=A.sum(1).rsqrt();normalized=inv[:,None]*A*inv[None,:]
x=torch.randn(6,4,device=device);y=torch.tensor([0,0,0,1,1,1],device=device);train_idx=torch.tensor([0,1,3,4],device=device)
class Model(nn.Module):
    def __init__(self):super().__init__();self.first=nn.Linear(4,8,bias=False);self.second=nn.Linear(8,2,bias=False)
    def forward(self,x):return normalized@self.second(F.relu(normalized@self.first(x)))
model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=.02)
for _ in range(args.steps):
    opt.zero_grad();loss=F.cross_entropy(model(x)[train_idx],y[train_idx]);loss.backward();opt.step()

if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
print(json.dumps({'example':'gcn','device':str(device),'parameter_updates':args.steps,'final_loss':loss.item(),'scope':'Teaching core on synthetic data; finite loss is not evidence of benchmark quality'}))
