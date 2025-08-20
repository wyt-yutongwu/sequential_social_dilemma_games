from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.switch import SwitchEnv
from social_dilemmas.maps import HARVEST_TEST_MAP,HARVEST_MAP_PAPER


def get_env_creator(
    env,
    num_agents,
    use_collective_reward=False,
    inequity_averse_reward=False,
    alpha=0.0,
    beta=0.0,
    num_switches=6,
    use_reputation = False
):
    if env == "harvest":

        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                # return_agent_actions=False,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
                use_reputation = use_reputation
            )

    elif env == "cleanup":

        def env_creator(_):
            return CleanupEnv(
                num_agents=num_agents,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
            )

    elif env == "switch":

        def env_creator(_):
            return SwitchEnv(num_agents=num_agents, num_switches=num_switches)
    elif env == "harvest_test":
        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                # return_agent_actions=False,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
                use_reputation = use_reputation,
                ascii_map = HARVEST_TEST_MAP
            )
    elif env == "harvest_paper":
        def env_creator(_):
            return HarvestEnv(
                num_agents=num_agents,
                # return_agent_actions=False,
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
                use_reputation = use_reputation,
                ascii_map = HARVEST_MAP_PAPER
            )
    else:
        raise ValueError(f"env must be one of harvest, cleanup, switch, not {env}")

    return env_creator
