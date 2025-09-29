import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sequential_social_dilemma_games.DQN_model.single_agent import dqn
from replay import ReplayMemory, Transition
env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


num_episodes = 600
algo = dqn(env = env, epsilon_decay=[EPS_START, EPS_END, EPS_DECAY], conv_filters = None, fc_filters=[128,128], lr = LR, replay_capacity=10000,batch_size=BATCH_SIZE, gamma=GAMMA, grad_clip=100, num_episode=600, tau=TAU,use_cnn=False)
algo.train()

