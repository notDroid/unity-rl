'''
config.py contains config files for the model and training. Select both for a run.
Still need to set params yourself (run/computer dependent):
    - name
    - workers
    - device, storage_device
'''

### ENV INFO

ENV_PATH = "../../../envs/Basic"

ROOT_KEY = ('group_0', 'agent_0')
OBSERVATION_KEY = "Basic"
ACTION_KEY = "discrete_action"

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

OBSERVATION_SHAPE = 20
ACTION_SHAPE = 3

### TRAIN CONFIGS

MODEL_PATH = 'models'
LOG_PATH = 'logs'
CKPT_PATH = 'ckpt'
RESULTS_PATH = 'results'