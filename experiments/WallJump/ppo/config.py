'''
config.py
'''

### ENV INFO

ENV_PATH = "../../../envs/WallJump"
N_ENVS = 24

ROOT_KEY = "agents"
OBSERVATION_KEY = "observation"
ACTION_KEY = "action"
REWARD_KEY = "reward"
DONE_KEY = "done"
TERMINATED_KEY = "terminated"

LOG_KEYS = [
    "timestep", "generation", "time", "collection_time", "train_time",  # Training Progress Metrics
    "return", "episode_length",                                         # Performance Metrics
    "entropy",                                                          # Exploration Metrics
    "policy_loss", "kl_approx", "clip_fraction", "ESS",                 # Policy Metrics
    "value_loss", "explained_variance",                                 # Value Metrics
]
LOG_INDEX = "timestep"
BEST_METRIC_KEY = "return"

### MODEL CONFIGS

OBSERVATION_SHAPE = 444
ACTION_SHAPE = 54

### TRAIN CONFIGS

MODEL_PATH = 'models'
LOG_PATH = 'logs'
CKPT_PATH = 'ckpt'
RESULTS_PATH = 'results'