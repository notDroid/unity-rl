### ENVIRONMENT CONFIG
ENV_PATH = "../../../envs/Walker"
N_ENVS = 10

OBSERVATION_DIM = 243
ACTION_DIM = 39

MODEL_PATH = 'models'
LOG_PATH = 'logs'
CKPT_PATH = 'ckpt'
RESULTS_PATH = 'results'


### MODEL CONFIG

MODEL_CONFIG = {
    "hidden_dim": 512,
    "n_blocks": 3,
    "in_features": OBSERVATION_DIM,
    "out_features": ACTION_DIM,
}