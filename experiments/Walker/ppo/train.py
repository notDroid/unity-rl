from rlkit.templates import PPOTrainer
from config import *
from env import create_env
from init_state import init_state

if __name__ == "__main__":
    state = init_state()
    env_fn = lambda: create_env(time_scale=TIME_SCALE)
    ppo = PPOTrainer(env_fn, TRAIN_CONFIG)
    ppo.load_state(**state)
    ppo.train()