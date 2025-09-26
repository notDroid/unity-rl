'''
config.py contains config files for the model and training. Select both for a run.
Still need to set params yourself (run/computer dependent):
    - name
    - workers
    - device, storage_device
'''

### ENV INFO

OBSERVATION_KEY = "VectorSensor_size8"
ACTION_KEY = "continuous_action"

LOG_KEYS = [
    "generation", "time", "collection_time", "train_time",  # Training Progress Metrics
    "return", "episode_length",                             # Performance Metrics
    "entropy", "action_std",                                # Exploration Metrics
    "policy_loss", "kl_approx", "clip_fraction", "ESS",     # Policy Metrics
    "value_loss", "explained_variance",                     # Value Metrics
]
LOG_INDEX = "generation"
BEST_METRIC_KEY = "return"

### MODEL CONFIGS

OBSERVATION_SHAPE = 8
ACTION_SHAPE = 2

DEFAULT_MODEL_CONFIG = {
    "hidden_dim": 256,
    "n_blocks": 3,
    "in_features": OBSERVATION_SHAPE,
    "out_features": ACTION_SHAPE,
}

MODEL_CONFIGS = [DEFAULT_MODEL_CONFIG]

### TRAIN CONFIGS

MODEL_PATH = 'models'
LOG_PATH = 'logs'
CKPT_PATH = 'ckpt'
RESULTS_PATH = 'results'

DEFAULT_TRAIN_CONFIG = {
    # Train Params
    "timestamps": 200_000,
    "generation_size": 2000,

    # Inner Train Loop Params
    "epochs": 10,
    "minibatch_size": 64,

    # PPO Params
    "epsilon": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "entropy_coef": 1e-5,

    # Optimizer Params
    "lr": 3e-4,
    "max_grad_norm": 0.5,

    # Checkpoint and Log Params
    "checkpoint_interval": 1,
    "log_interval": 1,
    "ckpt_path": CKPT_PATH,
    "log_path": LOG_PATH,
    "model_path": MODEL_PATH,
}

TRAIN_CONFIGS = [DEFAULT_TRAIN_CONFIG]