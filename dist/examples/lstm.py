"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python lstm.py --steps 20 --device auto
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

# Modern LSTM with forget gate; synthetic windows, no dataset benchmark.
class Model(nn.Module):
    def __init__(self):super().__init__();self.lstm=nn.LSTM(1,16,batch_first=True);self.head=nn.Linear(16,1)
    def forward(self,x):h,_=self.lstm(x);return self.head(h[:,-1])
model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=.01)
for _ in range(args.steps):
    phase=torch.rand(32,1,device=device)*6.28
    sequence=torch.sin(phase+torch.arange(17,device=device)*.2)
    x,y=sequence[:,:16,None],sequence[:,16,None]
    opt.zero_grad();loss=F.mse_loss(model(x),y);loss.backward();opt.step()

if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
print(json.dumps({'example':'lstm','device':str(device),'parameter_updates':args.steps,'final_loss':loss.item(),'scope':'Teaching core on synthetic data; finite loss is not evidence of benchmark quality'}))
