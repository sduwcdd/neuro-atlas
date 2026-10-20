"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python transformer.py --steps 20 --device auto
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

# Causal decoder-style core; original Transformer also had an encoder and cross-attention.
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(12,16);self.position=nn.Parameter(torch.randn(1,8,16)*.02)
        self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(16,2,32,dropout=0,batch_first=True),1)
        self.head=nn.Linear(16,12)
    def forward(self,x):
        mask=torch.triu(torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool),diagonal=1)
        return self.head(self.enc(self.emb(x)+self.position[:,:x.shape[1]],mask=mask))
start=torch.randint(0,12,(16,1),device=device)
sequence=(start+torch.arange(9,device=device))%12
x,y=sequence[:,:-1],sequence[:,1:]
model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=.01)
for _ in range(args.steps):
    opt.zero_grad();loss=F.cross_entropy(model(x).reshape(-1,12),y.reshape(-1));loss.backward();opt.step()

if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
print(json.dumps({'example':'transformer','device':str(device),'parameter_updates':args.steps,'final_loss':loss.item(),'scope':'Teaching core on synthetic data; finite loss is not evidence of benchmark quality'}))
