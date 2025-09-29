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


class dqn:
    def __init__(self,env, device,epsilon_decay, conv_filters, fc_filters, lr, replay_capacity, batch_size,gamma,grad_clip,num_episode,tau,episode_len,log_file,obs_shape = None, action_shape = None, use_cnn = False, debug = False,epsilon_decay_style="LINEAR"):
        # if GPU is to be used
        self.device = device
        self.env = env
        if obs_shape is None:
            obs,info = self.env.reset()
            obs_shape = obs.shape
            action_shape = env.action_space.n
        # initialize target and policy network
        self.policy_net = DQN_Module(env=self.env,conv_filters=conv_filters,fc_filters=fc_filters,use_cnn=use_cnn,obs_shape=obs_shape,action_shape=action_shape).to(self.device)
        self.target_net = DQN_Module(env=self.env,conv_filters=conv_filters,fc_filters=fc_filters,use_cnn=use_cnn,obs_shape=obs_shape,action_shape=action_shape).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        # initialize replay memory
        self.memory = ReplayMemory(replay_capacity)
        self.steps_done = 0 # num of steps so far
        self.eps_start, self.eps_end, self.eps_decay = epsilon_decay # exploration epsilon
        self.epsilon = self.eps_start
        self.batch_size = batch_size
        self.gamma = gamma # gamma for discounted reward
        self.grad_clip = grad_clip
        self.num_episodes= num_episode
        self.tau = tau
        self.episode_durations = []
        self.episode_len = episode_len
        # TODO do debug stuff
        self.debug = debug # prints debug messages if true
        self.epsilon_decay_style = epsilon_decay_style # whether to decrease epsilon exponentially or linearly
        self.log_file = None
        if log_file is not None:
            self.log_file = log_file
            self.log = []
            with open(self.log_file, 'w', newline='') as f:
                f.write("")
        self.state = None

    def update_epsilon_linear(self):
        if self.steps_done >= self.eps_decay:
            return
        else:
            self.epsilon = self.eps_start - (self.eps_start - self.eps_end) * (self.steps_done / self.eps_decay)

    def update_epsilon_exp(self):
        if self.steps_done >= self.eps_decay:
            return
        else:
            self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.steps_done / self.eps_decay)
        
    def select_action(self):
        sample = random.random()
        if self.epsilon_decay_style == "LINEAR":
            self.update_epsilon_linear()
        elif self.epsilon_decay_style == "EXPONENTIAL":
            self.update_epsilon_exp()
        self.steps_done += 1
        if sample > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                vals = self.policy_net(self.state)
                if self.debug and self.steps_done % 1000 == 0:
                    print("=======Q VALUES======")
                    print(f"step: {self.steps_done}")
                    print(f"epsilon: {self.epsilon}")
                    print(vals)

                return vals.max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def set_state(self, state):
        self.state = state

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # print(f"state_batch:{state_batch}")
        # print(f"action_batch:{action_batch}")
        # print(f"reward_batch:{reward_batch}")
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

    def train_one_step(self, action, observation, reward, terminated, truncated):
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor([[action]],device=self.device)
        self.memory.push(self.state, action, next_state, reward)

        # Move to the next state
        self.set_state(next_state)
        self.optimize_model()
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


    def train(self):
        state, info = self.env.reset()
        state = state
        info = info
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.set_state(state)
        for i_episode in range(self.num_episodes):
            # Initialize the environment and get its state

            for t in range(self.episode_len):
                action = self.select_action()
                observation, reward, terminated, truncated, _ = self.env.step(action)
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(self.state, action, next_state, reward)

                # Move to the next state
                self.state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
            state, info = self.env.reset()
            self.state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            # TODO hard code
            if self.log_file is not None:
                self.log_results_harvest_single_agent(info,i_episode)
        if self.log_file is not None:
            self.log_results_harvest_single_agent(info,i_episode)
        print('Complete')
    
    def log_results_cartpole(self, info, show_result):
        if not show_result:
            self.episode_durations.append(info + 1)
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
    
    def log_results_harvest_single_agent(self, info,i_episode,force_log=False):
        # TODO hard code
        if not force_log:
            rew = info["agent-0"]
            self.log.append(f"{rew}\n")
        if i_episode % 10 == 0 or force_log:
            with open(self.log_file, 'a', newline='') as f:
                f.writelines(self.log)
            self.log = []
