from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from util import create_env


# Get the observation and action spaces from the environment
env_instance = create_env({})
obs_space = env_instance.observation_space
act_space = env_instance.action_space

# Register Env
register_env("SoccerTwosEnv", create_env)

# Create config
config = (
    PPOConfig()
    .environment("SoccerTwosEnv")
    .framework("torch")
    .env_runners(
        num_env_runners=1,  # Number of parallel workers
        num_envs_per_env_runner=1,  # Environments per worker
    )
    .training(
        # Training params
        lr=0.0001,
        train_batch_size=4000,
        minibatch_size=128,
        num_epochs=2,

        # PPO params
        gamma=0.99,
        clip_param=0.2,
        entropy_coeff=0.01,
    )
    .multi_agent(
        policies={
            "shared_policy": (
                None,       # Default policy class
                obs_space,  # Use observation space
                act_space,  # Use action space
                {},         # Policy config
            )
        },
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",  # Map all agents to "shared_policy"
        policies_to_train=["shared_policy"]   # Train only the shared policy
    )
    .debugging(log_level="INFO")
)