from functools import lru_cache

from gym.utils import EzPickle
from supersuit import pad_observations_v0, pad_action_space_v0, flatten_v0
import numpy as np

# from pettingzoo.utils import wrappers
# from pettingzoo.utils.conversions import from_parallel_wrapper
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.wrappers import AssertOutOfBoundsWrapper, OrderEnforcingWrapper

from social_dilemmas.envs.env_creator import get_env_creator

MAX_CYCLES = 1000


def parallel_env(max_cycles=MAX_CYCLES, **ssd_args):
    return _parallel_env(max_cycles, **ssd_args)

# def parallel_env(max_cycles=MAX_CYCLES, **ssd_args):
#     env_creator = get_env_creator(**ssd_args)
#     env = env_creator(None)
    
#     # Assume env is already a ParallelEnv
#     env = pad_observations_v0(env)
#     env = pad_action_space_v0(env)
#     env = flatten_v0(env)

#     return env

# def parallel_env(max_cycles=MAX_CYCLES, **ssd_args):
#     env = _parallel_env(max_cycles, **ssd_args)

#     # Ensure it is a valid ParallelEnv
#     if not isinstance(env, ParallelEnv):
#         raise TypeError(f"_parallel_env must return a ParallelEnv, got {type(env)}")

#     # Apply supersuit wrappers
#     env = pad_observations_v0(env)
#     env = pad_action_space_v0(env)
#     env = flatten_v0(env)

#     return env

def raw_env(max_cycles=MAX_CYCLES, **ssd_args):
    # Get the ParallelEnv
    parallel = parallel_env(max_cycles, **ssd_args)

    # Convert to AECEnv using built-in method
    aec = parallel.parallel_to_aec()

    return aec

# def raw_env(max_cycles=MAX_CYCLES, **ssd_args):
#     return from_parallel_wrapper(parallel_env(max_cycles, **ssd_args))


# def env(max_cycles=MAX_CYCLES, **ssd_args):
#     aec_env = raw_env(max_cycles, **ssd_args)
#     aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
#     aec_env = wrappers.OrderEnforcingWrapper(aec_env)
#     return aec_env

def env(max_cycles=MAX_CYCLES, **ssd_args):
    aec_env = raw_env(max_cycles, **ssd_args)

    aec_env = AssertOutOfBoundsWrapper(aec_env)
    aec_env = OrderEnforcingWrapper(aec_env)

    return aec_env

class ssd_parallel_env(ParallelEnv):
    def __init__(self, env, max_cycles):
        self.ssd_env = env
        self.max_cycles = max_cycles
        self.possible_agents = list(self.ssd_env.agents.keys())
        self.ssd_env.reset()
        self.observation_space = lru_cache(maxsize=None)(lambda agent_id: env.observation_space)
        self.observation_spaces = {agent: env.observation_space for agent in self.possible_agents}
        self.action_space = lru_cache(maxsize=None)(lambda agent_id: env.action_space)
        self.action_spaces = {agent: env.action_space for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        self.dones = {agent: False for agent in self.agents}
        obs = self.ssd_env.reset()
        return obs, {}

    def seed(self, seed=None):
        return self.ssd_env.seed(seed)

    def render(self, mode="human"):
        return self.ssd_env.render(mode=mode)

    def close(self):
        self.ssd_env.close()

    def step(self, actions):  

        obss, rews, self.dones, infos = self.ssd_env.step(actions)
        del self.dones["__all__"]
        self.num_cycles += 1
        if self.num_cycles >= self.max_cycles:
            self.dones = {agent: True for agent in self.agents}
        self.agents = [agent for agent in self.agents if not self.dones[agent]]
        return obss, rews, self.dones, infos


class _parallel_env(ssd_parallel_env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, max_cycles, **ssd_args):
        EzPickle.__init__(self, max_cycles, **ssd_args)
        env = get_env_creator(**ssd_args)(ssd_args["num_agents"])
        super().__init__(env, max_cycles)

class CustomStepWrapper(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "CustomStepWrapper"}

    def __init__(self, env, max_cycles):
        self.env = env
        self.max_cycles = max_cycles
        self.possible_agents = env.possible_agents
        self.agents = self.possible_agents[:]  
        # self.render_mode = getattr(env, "render_mode", None)
        self.observation_spaces = env.observation_spaces
        self.action_spaces = env.action_spaces
        self.num_cycles = 0
        self.dones = {agent: False for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.num_cycles = 0
        self.dones = {agent: False for agent in self.possible_agents}
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        # Make sure each agent has an empty info dict
        if not info or not isinstance(info, dict):
            info = {}
        for agent in self.agents:
            if agent not in info:
                info[agent] = {}

        return obs, info

    def step(self, actions):

        obs, rew, done, info = self.env.step(actions)
        trunc = {agent: False for agent in self.agents}
        for agent in self.agents:
            if agent not in info:
                info[agent] = {}
        self.num_cycles += 1
        if self.num_cycles >= self.max_cycles:
            done = {agent: True for agent in self.possible_agents}
        return obs, rew, done, trunc, info

    def render(self, *args, **kwargs): return self.env.render(*args, **kwargs)
    def close(self): return self.env.close()
    def state(self): return self.env.state()