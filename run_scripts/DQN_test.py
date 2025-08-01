"""Simple example of setting up an agent-to-module mapping function.

How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --num-agents=2`

Control the number of agents and policies (RLModules) via --num-agents and
--num-policies.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`
"""

from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
)
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig
parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=100000,
    default_reward=600.0,
)
# TODO (sven): This arg is currently ignored (hard-set to 2).
parser.add_argument("--num-policies", type=int, default=2)


if __name__ == "__main__":
    args = parser.parse_args()

    # Register our environment with tune.
    register_env(
        "env",
        lambda _: MultiAgentCartPole(config={"num_agents": 2}),
    )
    config = (
            DQNConfig()
            .environment("env")    
            .env_runners(num_env_runners=1,
                        explore=True,
                        )

            .framework("torch")
            .multi_agent(
                policies={f"p{i}" for i in range(2)},
                policy_mapping_fn=lambda aid, *a, **kw: f"p{aid}",
            )
            .training(
                replay_buffer_config={
                "type": "MultiAgentReplayBuffer", 
                },
                lr=0.001,
                train_batch_size=100,
                gamma=0.99,
            )
        )

    algo = config.build_algo()
    for i in range(10):
        result = algo.train()
