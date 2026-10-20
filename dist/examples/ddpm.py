"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python ddpm.py --steps 20 --device auto
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

# Noise-prediction training only on 2D synthetic data; no reverse sampler.
timesteps=100
beta=torch.linspace(.0001,.02,timesteps,device=device);alpha_bar=torch.cumprod(1-beta,0)
model=nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,2)).to(device)
opt=torch.optim.Adam(model.parameters(),lr=.003)
for _ in range(args.steps):
    x0=torch.randn(64,2,device=device)*.3+torch.tensor([1.,-1.],device=device)
    t=torch.randint(timesteps,(64,),device=device);noise=torch.randn_like(x0);ab=alpha_bar[t,None]
    noisy=ab.sqrt()*x0+(1-ab).sqrt()*noise
    predicted=model(torch.cat([noisy,t[:,None].float()/timesteps],1))
    opt.zero_grad();loss=F.mse_loss(predicted,noise);loss.backward();opt.step()

if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
print(json.dumps({'example':'ddpm','device':str(device),'parameter_updates':args.steps,'final_loss':loss.item(),'scope':'Teaching core on synthetic data; finite loss is not evidence of benchmark quality'}))
