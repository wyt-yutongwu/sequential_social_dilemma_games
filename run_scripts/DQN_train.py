import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
import torch
from social_dilemmas.envs.pettingzoo_env import multi_agent_env

from ray.rllib.algorithms.callbacks import RLlibCallback

from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig

from custom_rl_mod import CustomCNNTorchRLModule

torch.cuda.set_device(1)
device = torch.device("cuda")# if torch.cuda.is_available() else torch.device("cpu")

HARVEST_VIEW_SIZE = 7

def parse_args():
    parser = argparse.ArgumentParser("Stable-Baselines3 PPO with Parameter Sharing")
    parser.add_argument(
        "--env-name",
        type=str,
        default="harvest",
        choices=["harvest", "cleanup","harvest_test","harvest_paper"],
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
        default=0.6,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Disadvantageous inequity aversion factor",
    )
    parser.add_argument(
        "--use_reputation",
        type=bool,
        default=False,
        help="Give agent other agents' reputation",
    )
    args = parser.parse_args()
    return args

class CustomLoggingCallback(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.ep_count = 0
    def on_episode_created(self, *, episode, **kwargs):
        self.ep_count += 1
    # Log custom information
    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        num_apples = episode.get_infos()["agent-0"][-1]["num_apples"]
        metrics_logger.log_value("min_num_apples", value=num_apples, reduce='mean', window=1)
        for agent_id, agent_eps in episode.agent_episodes.items():
            info = agent_eps.get_infos()[-1]
            if info is None:
                continue
            if "total_time_out_steps" in info:
                time_out= info["total_time_out_steps"]
                metrics_logger.log_value(f"{agent_id}/time_out", value=time_out, reduce='mean', window=1)
            if "reward_this_episode" in info:
                rew = info["reward_this_episode"]
                metrics_logger.log_value(f"{agent_id}/rew", value=rew, reduce='mean', window=1)
            if "beam_attempt" in info:
                beam = info["beam_attempt"]
                metrics_logger.log_value(f"{agent_id}/beam_attempt", value=beam, reduce='mean', window=1)
            if "successful_hit" in info:
                succ_hit = info["successful_hit"]
                metrics_logger.log_value(f"{agent_id}/succ_hit", value=succ_hit, reduce='mean', window=1)

def env_creator(config):
    env = multi_agent_env(
        max_cycles=config.get("rollout_len", 1000),
        env=config.get("env_name", "harvest"),
        num_agents=config.get("num_agents", 5),
        use_collective_reward=config.get("use_collective_reward", False),
        inequity_averse_reward=config.get("inequity_averse_reward", False),
        alpha=config.get("alpha", 5),
        beta=config.get("beta", 0.05),
        use_reputation=config.get("use_reputation", False),
    )
    return env

def policy_mapping_fn(agent_id, *args, **kwargs):
    idx = int(agent_id.split("-")[-1])  # assumes agent ids are like agent_0, agent_1...
    return f"policy-{idx}"


def main(args):
    register_env("mapenv", env_creator)
    # Config
    env_name = args.env_name
    num_agents = args.num_agents
    rollout_len = args.rollout_len
    total_timesteps = args.total_timesteps
    use_collective_reward = args.use_collective_reward
    inequity_averse_reward = args.inequity_averse_reward
    alpha = args.alpha
    beta = args.beta
    use_reputation = args.use_reputation

    # Training
    num_envs = 4  # number of parallel multi-agent environments
    num_frames = 6  # number of frames to stack together; use >4 to avoid automatic VecTransposeImage
    batch_size = rollout_len * num_envs // 2  # This is from the rllib baseline implementation
    lr = 0.0001
    gamma = 0.995

    env = env_creator({
    "env_name": env_name,
    "num_agents": num_agents,
    "rollout_len": rollout_len,
    "use_collective_reward": use_collective_reward,
    "inequity_averse_reward": inequity_averse_reward,
    "alpha": alpha,
    "beta": beta,
    "use_reputation": use_reputation,
    "num_frames": num_frames,
    })
    obs, info = env.reset()

    agent_ids = env.agents
    obs_space = env.observation_spaces[agent_ids[0]]
    act_space = env.action_spaces[agent_ids[0]]
    policies = {f"policy-{i}" for i in range(num_agents)}


    config = (
        DQNConfig()
        .environment("mapenv", env_config={
            "env_name": env_name,
            "num_agents": num_agents,
            "rollout_len": rollout_len,
            "use_collective_reward": use_collective_reward,
            "inequity_averse_reward": inequity_averse_reward,
            "alpha": alpha,
            "beta": beta,
            "use_reputation": use_reputation,
            "num_frames": num_frames,
        })    
        .env_runners(num_env_runners=4,
                     explore=True,
                     rollout_fragment_length=1000,
                    )
        .learners(num_learners=1)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            replay_buffer_config={
            "type": "MultiAgentReplayBuffer",  
            "capacity": 10000,
            },
            lr=lr,
            train_batch_size=batch_size,
            gamma=gamma,
        )

        # .rl_module(rl_module_spec=MultiRLModuleSpec(
        #         rl_module_specs={p: RLModuleSpec(module_class=DefaultDQNTorchRLModule,
        #         model_config={
        #             "conv_filters": [
        #                 [32, [3, 3], 1],  
        #                 [64, [3, 3], 1],  
        #             ],
        #             "conv_activation": "relu",
        #             "post_fcnet_hiddens": [256],
        #             "post_fcnet_activation": "relu",
        #         }) for p in policies},
        #     ),
        #     )
        .rl_module(rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={p: RLModuleSpec(
                module_class=CustomCNNTorchRLModule,
                model_config={
            # Custom configuration specific to your environment/module
            "custom_model_config": {
                "view_len": HARVEST_VIEW_SIZE,
                "num_agents": num_agents,
                "return_agent_actions": True,
            },
            # CNN configuration for image processing
            "conv_filters": [
            [256, [3, 3], 1],
            [512, [3, 3], 1],
            [512, [3, 3], 1],
            [256, [3, 3], 1],
            [128, [3, 3], 1]   # 5 conv layers total
        ],
            "conv_activation": "relu",
            "agent_fc_hiddens": [512, 256, 128, 64], 
            "post_fcnet_hiddens": [2048, 1024, 512, 256], 
            "post_fcnet_activation": "relu",
        }) for p in policies},
        ))
        .callbacks(CustomLoggingCallback)
    )


    checkpoint_dir = os.path.abspath("checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    algo = config.build_algo()
    for i in range(int(total_timesteps)):
        result = algo.train()
        if i % 100 == 0:
            print(i)
    save_path = os.path.join(checkpoint_dir, f"PPO_{args.env_name}_{i}")
    algo.save(save_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
