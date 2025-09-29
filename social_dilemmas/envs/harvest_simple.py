import numpy as np
import random

from social_dilemmas.envs.agent import HarvestAgent
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.maps import HARVEST_SIMPLE
import math

APPLE_RADIUS = 2
TIME_OUT_DURATION = 25

# Add custom actions to the agent
_HARVEST_ACTIONS = {}#{"FIRE": 5}  # length of firing range
FEASIBLE_POINTS=[[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]]
# SPAWN_PROB = [0.001, 0.005, 0.02, 0.05]
APPLE_SPAWN = [[1,2], [2,1],[2,3],[3,2]]
HARVEST_VIEW_SIZE = 2


class HarvestSimpleEnv(MapEnv):
    def __init__(
        self,
        ascii_map=HARVEST_SIMPLE,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        use_reputation = False,
        alpha=0.0,
        beta=0.0,
    ):
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
        )
        self.apple_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

    @property
    def action_space(self):
        return DiscreteWithDType(7, dtype=np.uint8)

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation =  self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE, time_out_duration=TIME_OUT_DURATION)
            self.agents[agent_id] = agent

    def custom_reset(self):
        reset_apple_pos = APPLE_SPAWN[random.randint(0,len(APPLE_SPAWN) - 1)]

        """Initialize the walls and the apples"""
        self.single_update_map(reset_apple_pos[0], reset_apple_pos[1], b"A")

    # def custom_action(self, agent, action):
    #     updates, hit_agent_rep = self.update_map_fire(
    #         agent.pos.tolist(),
    #         agent.get_orientation(),
    #         self.all_actions["FIRE"],
    #         fire_char=b"F",
    #     )
    #     agent.fire_beam(b"F", hit_agent_rep)
    #     return updates
    

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

    def spawn_apples(self):
        """Apple only spawns if there is no apple left

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """
        new_apple_points = []
        empty_space = []
        for i,j in FEASIBLE_POINTS:
                if self.world_map[i,j] == b"A":
                    new_apple_points.append((i, j, b"A"))
                    self.num_spawn_apples += 1
                    return new_apple_points
                # do not spawn at where the agent is standing
                elif [i,j] not in self.agent_pos and self.world_map[i,j] != b"@":
                    empty_space.append([i,j])
        new_apple_pos = empty_space[random.randint(0,len(empty_space) - 1)]
        new_apple_points.append((new_apple_pos[0], new_apple_pos[1], b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples



