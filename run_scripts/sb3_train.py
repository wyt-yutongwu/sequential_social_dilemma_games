import argparse
import gymnasium
# from gymnasium.spaces import Discrete

import supersuit as ss
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from torch import nn
from supersuit import dtype_v0
import numpy as np
from social_dilemmas.envs.pettingzoo_env import parallel_env, CustomStepWrapper
from stable_baselines3.common.vec_env import VecEnvWrapper

from stable_baselines3.common.vec_env import SubprocVecEnv
# from social_dilemmas.envs.pettingzoo_env import parallel_env
# from supersuit import pettingzoo_env_to_vec_env_v1
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN
torch.cuda.set_device(3)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def parse_args():
    parser = argparse.ArgumentParser("Stable-Baselines3 PPO with Parameter Sharing")
    parser.add_argument(
        "--env-name",
        type=str,
        default="harvest",
        choices=["harvest", "cleanup"],
        help="The SSD environment to use",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=5,
        help="The number of agents",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        default=1000,
        help="length of training rollouts AND length at which env is reset",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=5e8,
        help="Number of environment timesteps",
    )
    parser.add_argument(
        "--use-collective-reward",
        type=bool,
        default=False,
        help="Give each agent the collective reward across all agents",
    )
    parser.add_argument(
        "--inequity-averse-reward",
        type=bool,
        default=False,
        help="Use inequity averse rewards from 'Inequity aversion \
            improves cooperation in intertemporal social dilemmas'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Disadvantageous inequity aversion factor",
    )
    args = parser.parse_args()
    return args

class CustomInfoLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.time_out_list = np.zeros(60)
        self.reward_list = np.zeros(60)
        self.rep_list = np.zeros(60)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            if 'total_time_out_steps' in info:
                self.time_out_list[i] = info['total_time_out_steps']
            if 'reward_this_episode' in info:
                self.reward_list[i] = info['reward_this_episode']
            if 'reputation' in info:
                self.rep_list[i] = info['reputation']
        return True
    
    def _on_rollout_end(self):
        for i in range(0, len(self.reward_list)):
            self.logger.record(f'custom/agent_{i}/reward', float(self.reward_list[i]))
            self.logger.record(f'custom/agent_{i}/time_out', float(self.time_out_list[i]))
            self.logger.record(f'custom/agent_{i}/reputation', float(self.rep_list[i]))
        return True


# Use this with lambda wrapper returning observations only
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        features_dim=128,
        view_len=7,
        num_frames=6,
        fcnet_hiddens=[1024, 128],
    ):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        flat_out = num_frames * 6 * (view_len * 2 - 1) ** 2
        self.conv = nn.Conv2d(
            in_channels=num_frames * 3,  # Input: (3 * 4) x 15 x 15
            out_channels=num_frames * 6,  # Output: 24 x 13 x 13
            kernel_size=3,
            stride=1,
            padding="valid",
        )
        self.fc1 = nn.Linear(in_features=flat_out, out_features=fcnet_hiddens[0])
        self.fc2 = nn.Linear(in_features=fcnet_hiddens[0], out_features=fcnet_hiddens[1])

    def forward(self, observations) -> torch.Tensor:
        # Convert to tensor, rescale to [0, 1], and convert from B x H x W x C to B x C x H x W
        observations = observations.permute(0, 3, 1, 2)
        features = torch.flatten(F.relu(self.conv(observations)), start_dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return features
    

class VecActionCastWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        # Force casting to uint8 to match DiscreteWithDType
        actions = actions.astype(np.uint8)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, terminations, truncations, infos = self.venv.step_wait()

        # Convert to single done flag
        dones = np.array([t or u for t, u in zip(terminations, truncations)], dtype=np.bool_)
        return obs, rew, dones, infos



def main(args):
    # Config
    env_name = args.env_name
    num_agents = args.num_agents
    rollout_len = args.rollout_len
    total_timesteps = args.total_timesteps
    use_collective_reward = args.use_collective_reward
    inequity_averse_reward = args.inequity_averse_reward
    alpha = args.alpha
    beta = args.beta

    # Training
    num_cpus = 4  # number of cpus
    num_envs = 12  # number of parallel multi-agent environments
    num_frames = 6  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    features_dim = (
        128  # output layer of cnn extractor AND shared layer for policy and value functions
    )
    fcnet_hiddens = [1024, 128]  # Two hidden layers for cnn extractor
    ent_coef = 0.001  # entropy coefficient in loss
    batch_size = rollout_len * num_envs // 2  # This is from the rllib baseline implementation
    lr = 0.0001
    n_epochs = 30
    gae_lambda = 1.0
    gamma = 0.99
    target_kl = 0.01
    grad_clip = 40
    verbose = 3

    env = parallel_env(
        max_cycles=rollout_len,
        env=env_name,
        num_agents=num_agents,
        use_collective_reward=use_collective_reward,
        inequity_averse_reward=inequity_averse_reward,
        alpha=alpha,
        beta=beta,
    )
    env = CustomStepWrapper(env, max_cycles=rollout_len)  # After Supersuit wraps

    env = ss.observation_lambda_v0(env, lambda x, _: x["curr_obs"], lambda s: s["curr_obs"])

    env = dtype_v0(env, np.uint8)

    env = ss.frame_stack_v1(env, num_frames)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_envs, num_cpus=0, base_class="stable_baselines3"
    )
    original_reset = env.reset  # Save original
    env.reset = lambda **kwargs: original_reset(**kwargs)[0]  
    env = VecActionCastWrapper(env) 

    env = VecMonitor(env)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(
            features_dim=features_dim, num_frames=num_frames, fcnet_hiddens=fcnet_hiddens
        ),
        net_arch=[features_dim],
    )

    tensorboard_log = "./results/sb3/cleanup_ppo_paramsharing"

    model = PPO(
        "CnnPolicy",
        env=env,
        learning_rate=lr,
        n_steps=rollout_len,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        max_grad_norm=grad_clip,
        target_kl=target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    model.learn(total_timesteps=total_timesteps,callback=CustomInfoLoggerCallback())

    logdir = model.logger.dir
    model.save(logdir + "/model")
    del model
    model = PPO.load(logdir + "/model")  # noqa: F841


if __name__ == "__main__":
    args = parse_args()
    main(args)
