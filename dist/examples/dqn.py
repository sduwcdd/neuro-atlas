"""Teaching core only; synthetic data / small environment, not a paper reproduction.
Run: python dqn.py --steps 20 --device auto
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

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
env=gym.make('CartPole-v1')
model=DQN('MlpPolicy',env,device=str(device),learning_starts=8,buffer_size=2048,batch_size=8,train_freq=1,gradient_steps=1,target_update_interval=16,seed=7,verbose=0)
model.learn(total_timesteps=args.steps)
evaluation_env=gym.make('CartPole-v1')
mean_return,std_return=evaluate_policy(model,evaluation_env,n_eval_episodes=2,deterministic=True)
env.close();evaluation_env.close()
print(json.dumps({'example':'dqn','device':str(model.device),'requested_env_steps':args.steps,'actual_env_steps':model.num_timesteps,'evaluation_episodes':2,'evaluation_mean_return':mean_return,'scope':'Short training pipeline check, not a reliable performance estimate or paper reproduction'}))
