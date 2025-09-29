from DQN_model.single_agent import dqn
import gymnasium as gym
import math
import random
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from DQN_model.module import DQN_Module
from DQN_model.replay import ReplayMemory, Transition
import matplotlib
import matplotlib.pyplot as plt

class mult_agent_dqn:
    def __init__(self,device,env, epsilon_decay, conv_filters, fc_filters, lr, replay_capacity, batch_size,gamma,grad_clip,num_episode,tau,episode_len,log_file,use_cnn = False, debug = False,epsilon_decay_style="LINEAR"):
        self.device = device
        obs, info = env.reset()
        self.agent_ids = list(env.agents.keys())
        self.env = env
        self.dqn_lst = {}
        obs_shape = obs[self.agent_ids[0]].shape
        action_shape = env.action_space.n
        self.num_episodes = num_episode

        self.episode_len = episode_len
        self.log_file = log_file
        self.log = []
        with open(self.log_file, 'w', newline='') as f:
            f.write("")

        for id in self.agent_ids:
            mod = dqn(self.env, device = device,epsilon_decay=epsilon_decay, obs_shape=obs_shape, action_shape=action_shape,conv_filters=conv_filters,fc_filters=fc_filters,lr=lr,replay_capacity=replay_capacity,batch_size=batch_size,gamma=gamma,grad_clip=grad_clip,num_episode=num_episode,tau=tau, episode_len=episode_len,log_file=None, use_cnn=use_cnn,debug=debug,epsilon_decay_style=epsilon_decay_style)
            self.dqn_lst[id] = mod

    def multi_agent_select_action(self):
        action_dict = {}
        for id in self.agent_ids:
            action = self.dqn_lst[id].select_action()
            action = action.item()
            action_dict[id] = action
        return action_dict

    def multi_agent_set_state(self, state):
        for id in self.agent_ids:
            id_state = state[id]
            id_state = torch.tensor(id_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.dqn_lst[id].set_state(id_state)

    def train(self):
        state, info = self.env.reset()
        self.multi_agent_set_state(state)
        for i_episode in range(self.num_episodes):
            for t in range(self.episode_len):
                action_dict = self.multi_agent_select_action()
                observation, reward, terminated, truncated, _ = self.env.step(action_dict)
                for id in self.agent_ids:
                    obs = observation[id]
                    rew = reward[id]
                    term = terminated[id]
                    trunc = truncated[id]
                    act = action_dict[id]
                    self.dqn_lst[id].train_one_step(act,obs,rew,term,trunc)
            state, info = self.env.reset()
            self.log_harvest(i_episode, info)
            self.multi_agent_set_state(state)
        state, info = self.env.reset()
        self.log_harvest(i_episode, info, force_log=True)
        print("complete")
    
    # TODO make this customizable
    def log_harvest(self,i_episode, info,force_log = False):
        if not force_log:
            line = ""
            for id in sorted(self.agent_ids):
                reward, time_out, beam_attempt,beam_success,apples=info[id]
                line += f"{reward},{time_out},{beam_attempt},{beam_success},{apples}"
            self.log.append(line[:-1]+"\n")
        if i_episode % 10 == 0 or force_log:
            with open(self.log_file, 'a', newline='') as f:
                f.writelines(self.log)
            self.log = []
