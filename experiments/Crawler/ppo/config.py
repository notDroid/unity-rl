ENV_PATH = "../../../envs/Crawler"
N_ENVS = 10

OBSERVATION_DIM = 158
ACTION_DIM = 20

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