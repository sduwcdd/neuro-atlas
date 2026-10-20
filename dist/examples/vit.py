"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python vit.py --steps 20 --device auto
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

# Tiny patch encoder with global average pooling, not pretrained ViT.
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.patch=nn.Conv2d(1,16,4,stride=4)
        self.position=nn.Parameter(torch.randn(1,4,16)*.02)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(16,2,32,dropout=0,batch_first=True),1)
        self.head=nn.Linear(16,2)
    def forward(self,x):
        tokens=self.patch(x).flatten(2).transpose(1,2)+self.position
        return self.head(self.encoder(tokens).mean(1))
x=torch.randn(16,1,8,8,device=device);y=(x.mean((1,2,3))>0).long()
model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=.01)
for _ in range(args.steps):
    opt.zero_grad();loss=F.cross_entropy(model(x),y);loss.backward();opt.step()

if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
print(json.dumps({'example':'vit','device':str(device),'parameter_updates':args.steps,'final_loss':loss.item(),'scope':'Teaching core on synthetic data; finite loss is not evidence of benchmark quality'}))
