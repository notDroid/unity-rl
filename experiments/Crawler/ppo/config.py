### ENVIRONMENT CONFIG
ENV_PATH = "../../../envs/Crawler"
N_ENVS = 10

OBSERVATION_DIM = 158
ACTION_DIM = 20


### CHECKPOINT AND LOGGER CONFIG
LOG_KEYS = [
    "timestep", "generation", "time", "collection_time", "train_time",  # Training Progress Metrics
    "return", "episode_length",                                         # Performance Metrics
    "entropy",                                                          # Exploration Metrics
    "policy_loss", "kl_approx", "clip_fraction", "ESS",                 # Policy Metrics
    "value_loss", "explained_variance",                                 # Value Metrics
]
LOG_INDEX = "timestep"
BEST_METRIC_KEY = "return"

MODEL_PATH = 'models'
LOG_PATH = 'logs'
CKPT_PATH = 'ckpt'
RESULTS_PATH = 'results'


### MODEL CONFIG

MODEL_CONFIG = {
    "hidden_dim": 1024,
    "n_blocks": 3,
    "in_features": OBSERVATION_DIM,
    "out_features": ACTION_DIM,
}

### RL CONFIG

# ENV Params
TIME_SCALE = 5

# PPO Params
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON = 0.2
ENTROPY_COEF = 1e-2
VALUE_COEF = 1

### TRAIN CONFIG

# Device Specific Stuff
import os, math, torch
from rlkit.util import round_up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = os.cpu_count() // 2
COLLECTOR_BUFFER_SIZE = 1000 * WORKERS * N_ENVS

# Config that Often Changes
GENERATION_SIZE = round_up(COLLECTOR_BUFFER_SIZE, COLLECTOR_BUFFER_SIZE)
GENERATIONS = 1500
LR = 3e-4


TRAIN_CONFIG = {
    # Device
    "device": device,
    "storage_device": "cpu",

    ### Collection phase
    "workers": WORKERS,
    "collector_buffer_size": COLLECTOR_BUFFER_SIZE,
    "generation_size": GENERATION_SIZE,
    "generations": GENERATIONS,

    ### Advantage phase
    "slice_len": 256,
    "n_slices": 128,

    ### Train phase
    "epochs": 3,
    "minibatch_size": 1024,
    "lr": LR,
    "max_grad_norm": 0.5,

    # Early Stop
    "kl_soft_clip": 0.03,
    "early_stop_threshold": 3 * math.ceil(GENERATION_SIZE / 1024) // 10,
    "kl_hard_clip": 0.05,

    ### Checkpointing and Logging
    "checkpoint_interval": 1,
    "log_interval": 1,
    "model_path": MODEL_PATH,
    "best_metric_key": BEST_METRIC_KEY
}

### Run Config
NAME = "run1"
CONTINUE = True