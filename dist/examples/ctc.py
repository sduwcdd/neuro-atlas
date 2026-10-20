"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python ctc.py --steps 20 --device auto
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

# Synthetic frame features and short unaligned targets; no speech recognition system.
class Model(nn.Module):
    def __init__(self):super().__init__();self.rnn=nn.GRU(6,12,batch_first=True,bidirectional=True);self.head=nn.Linear(24,4)
    def forward(self,x):h,_=self.rnn(x);return self.head(h).log_softmax(-1).transpose(0,1)
model=Model().to(device);opt=torch.optim.Adam(model.parameters(),lr=.01)
x=torch.randn(4,8,6,device=device);targets=torch.tensor([[1,2],[2,3],[1,3],[3,1]],device=device,dtype=torch.long)
# Lengths are real sequence lengths; blank=0 must not appear in targets.
input_lengths=torch.full((4,),8,dtype=torch.long);target_lengths=torch.full((4,),2,dtype=torch.long)
criterion=nn.CTCLoss(blank=0,zero_infinity=False)
for _ in range(args.steps):
    opt.zero_grad();loss=criterion(model(x),targets,input_lengths,target_lengths);loss.backward();opt.step()

if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
print(json.dumps({'example':'ctc','device':str(device),'parameter_updates':args.steps,'final_loss':loss.item(),'scope':'Teaching core on synthetic data; finite loss is not evidence of benchmark quality'}))
