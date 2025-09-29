import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from DQN_model.multi_agent import mult_agent_dqn
from social_dilemmas.envs.harvest_simple import HarvestSimpleEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.maps import HARVEST_TEST_MAP,HARVEST_MAP_PAPER, HARVEST_SINGLE_AGENT
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set seeds for reproducible results"""
    # Python random module
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
# Use it at the beginning of your script
set_seed(3407)

def main():
    device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )   
    batch_size = 50
    lr = 0.0001
    gamma = 0.99
    grad_clip = 0.5
    epsilon = [1,0.1,100000]
    replay_capacity =10000
    conv_filter = [ [32, [3, 3], 1],[16,[3,3],1] ]
    fc_hidden = [32,16]
    tau = 0.005
    # env = HarvestSimpleEnv(
    #             num_agents=1,
    #             # return_agent_actions=False,
    #             return_agent_actions=False,
    #             use_collective_reward=False,
    #             inequity_averse_reward=False,
    #             alpha=0,
    #             beta=0,
    #             use_reputation = False,
    #         )
    env = HarvestEnv(
                num_agents=1,
                # return_agent_actions=False,
                return_agent_actions=False,
                use_collective_reward=False,
                inequity_averse_reward=False,
                alpha=0,
                beta=0,
                use_reputation = False,
                ascii_map = HARVEST_SINGLE_AGENT
            )
    num_episodes = 100000
    episode_length = 1000
    num = 6
    log_file=f"/home/yw180/sequential_social_dilemma_games/results/harvest_sa_result_{num}.csv"
    algo = mult_agent_dqn(device=device,env = env, epsilon_decay=epsilon, conv_filters=conv_filter,fc_filters=fc_hidden,lr=lr, replay_capacity=replay_capacity, batch_size=batch_size,gamma=gamma,grad_clip=grad_clip, num_episode=num_episodes,tau=tau,episode_len=episode_length,use_cnn=True, log_file=log_file, debug= True)
    algo.train()
if __name__ == "__main__":
    main()
