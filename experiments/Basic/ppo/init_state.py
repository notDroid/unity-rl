import torch
from torch import optim
from rlkit.utils import Checkpointer, Logger, MultiVersionCheckpointer
from train_util import make_loss_module, create_policy, create_value

from config import *

def init_state():
    policy, value = create_policy(MODEL_CONFIG).to(device), create_value(MODEL_CONFIG).to(device)

    # Device
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_dtype   = torch.float16 if device_type == "cuda" else torch.float32

    # Loss + Optimizer
    loss_module = make_loss_module(policy, value, epsilon=EPSILON, entropy_coef=ENTROPY_COEF, gamma=GAMMA, lmbda=GAE_LAMBDA, value_coef=VALUE_COEF)
    optimizer = optim.Adam(loss_module.parameters(), lr=LR)
    # only need scaler with float16, float32 and bfloat16 have wider exponent ranges.
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # Logger + Checkpointer
    logger = Logger(keys = LOG_KEYS, log_path=LOG_PATH, name=NAME)
    checkpointer = Checkpointer(ckpt_path=CKPT_PATH, name=NAME, metric_key=BEST_METRIC_KEY)

    # Continue/Reset
    start_generation = 0
    if not CONTINUE:
        logger.reset()
        checkpointer.reset()
    else:
        checkpoint = checkpointer.load_progress()
        if checkpoint:
            start_generation = int(checkpoint["generation"])
            policy.load_state_dict(checkpoint["policy_state_dict"])
            value.load_state_dict(checkpoint["value_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print("CHECKPOINT FOUND, STARTING FROM GENERATION:", start_generation)
            logger.revert("generation", start_generation)
        else:
            print("CHECKPOINT NOT FOUND, STARTING FROM SCRATCH")
            logger.reset()

    return {
        "policy": policy,
        "value": value,
        "optimizer": optimizer,
        "loss_module": loss_module,
        "checkpointer": checkpointer,
        "logger": logger,
        "scaler": scaler,
        "start_generation": start_generation,
    }