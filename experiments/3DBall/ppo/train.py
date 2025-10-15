from rlkit.templates import PPOWithDynamicEntropy
from config import *
from env import create_randomized_env
from init_state import init_state

if __name__ == "__main__":
    state = init_state()
    env_fn = lambda: create_randomized_env(time_scale=TIME_SCALE)
    train_config = TRAIN_CONFIG.copy()
    start_generation = state["start_generation"]
    
    # # Exploration Phase
    # train_config["generations"] = GENERATIONS // 3
    # ppo = PPOWithDynamicEntropy(env_fn, train_config)
    # ppo.load_state(**state)
    # ppo.train()

    # Exploitation Phase
    train_config["generations"] = GENERATIONS
    state["start_generation"] = start_generation # max(start_generation, GENERATIONS // 3)
    state["entropy_module"].target_entropy = EXPLOITATION_TARGET_ENTRPOY
    ppo = PPOWithDynamicEntropy(env_fn, train_config)
    ppo.load_state(**state)
    ppo.train()