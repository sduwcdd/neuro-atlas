"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python gan.py --steps 20 --device auto
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

# 1D Gaussian GAN with a non-saturating generator objective.
generator=nn.Sequential(nn.Linear(2,16),nn.ReLU(),nn.Linear(16,1)).to(device)
discriminator=nn.Sequential(nn.Linear(1,16),nn.LeakyReLU(.2),nn.Linear(16,1)).to(device)
g_opt=torch.optim.Adam(generator.parameters(),lr=.005);d_opt=torch.optim.Adam(discriminator.parameters(),lr=.005)
for _ in range(args.steps):
    real=2+.5*torch.randn(64,1,device=device);fake=generator(torch.randn(64,2,device=device))
    d_opt.zero_grad()
    d_loss=F.binary_cross_entropy_with_logits(discriminator(real),torch.ones(64,1,device=device))+F.binary_cross_entropy_with_logits(discriminator(fake.detach()),torch.zeros(64,1,device=device))
    d_loss.backward();d_opt.step()
    for p in discriminator.parameters():p.requires_grad_(False)
    g_opt.zero_grad();fake=generator(torch.randn(64,2,device=device))
    loss=F.binary_cross_entropy_with_logits(discriminator(fake),torch.ones(64,1,device=device));loss.backward();g_opt.step()
    for p in discriminator.parameters():p.requires_grad_(True)

if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
print(json.dumps({'example':'gan','device':str(device),'parameter_updates':args.steps,'final_loss':loss.item(),'scope':'Teaching core on synthetic data; finite loss is not evidence of benchmark quality'}))
