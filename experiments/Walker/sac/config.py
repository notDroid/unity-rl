### ENVIRONMENT CONFIG
ENV_PATH = "../../../envs/Walker"
N_ENVS = 10

OBSERVATION_DIM = 243
ACTION_DIM = 39

MODEL_PATH = 'models'
LOG_PATH = 'logs'
CKPT_PATH = 'ckpt'
RESULTS_PATH = 'results'

LOG_KEYS = [
    "timestep", "time", "collect_wait_time", "train_time",  # Training Progress Metrics
    "score",                                                # Latest Metrics
    "policy_loss", "qvalue_loss", "alpha",                             # Loss Metrics      
    "entropy", "td_error",                              # Batch Metrics (computed with loss)  
    "return", "episode_length", "eval_entropy",             # Eval Performance Metrics                       
]
LOG_INDEX = "timestep"


### MODEL CONFIG

MODEL_CONFIG = {
    "hidden_dim": 128,
    "n_blocks": 3,
    "in_features": OBSERVATION_DIM,
    "out_features": ACTION_DIM,
}

BENCH_AVG_REWARD = 0.025
