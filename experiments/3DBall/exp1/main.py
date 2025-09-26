'''
Usage: python main [train/test/plot/play] name
'''

import os
import torch
import argparse
from env import create_env
from model import create_policy, create_value
from train import train
from train_util import load_model
from result import plot, play, test
import config

DEFAULT_TEST_TIMESTAMPS = 1000

def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO training on 3DBallEnv"
    )

    parser.add_argument("method", type=str, default="train", help="\"train\" or \"test\"", choices=("train", "test", "plot", "play"))
    parser.add_argument("name", type=str, default="testrun0", help="model to use (from ckpt/model directory)")
    parser.add_argument("--model_config", required=False, type=str, default="default_model_config", 
        help="select model config from config.py (add your own)", choices=config.MODEL_CONFIG_LIST)
    parser.add_argument("--train_config", required=False, type=str, default="default_train_config", 
        help="select train config from config.py (add your own)", choices=config.TRAIN_CONFIG_LIST)
    parser.add_argument("--continue_training", required=False, type=bool, default=True, help="continue training or reset")
    parser.add_argument("--test_timestamps", required=False, type=str, default=DEFAULT_TEST_TIMESTAMPS, help="specify max test timestamps. default: 100")
    parser.add_argument("--workers", required=False, type=int, help="default inferred")
    parser.add_argument("--device", required=False, type=str, help="specify device: \"cpu\" or \"cuda\". default inferred", choices=("cpu", "cuda"))
    parser.add_argument("--storage_device", required=False, type=str, default="cpu",
        help="specify storage device: \"cpu\" or \"cuda\". default cpu", choices=("cpu", "cuda"))

    for key, value in config.DEFAULT_TRAIN_CONFIG.items():
        parser.add_argument(f"--{key}", required=False, type=type(value), help=f"set train config {key}")

    return vars(parser.parse_args())

def start_train(args):
    # Create trian config
    train_config = config.TRAIN_CONFIGS[args["train_config"]].copy()
    # Overwrite
    for key in args:
        if args[key] is not None:
            train_config[key] = args[key]
    # Include run information
    for key in ("name", "workers", "device", "storage_device"):
        train_config[key] = args[key]

    # Create models
    model_config = config.MODEL_CONFIGS[args["model_config"]]
    device = args["device"]
    policy, value = create_policy(model_config).to(device), create_value(model_config).to(device)

    train(create_env, policy, value, train_config, continue_training=args["continue_training"])

if __name__ == "__main__":
    args = parse_args()
    def default(arg, value): 
        if args[arg] is None: args[arg] = value

    # Infer some args
    default("workers", os.cpu_count())
    if args["device"] is not None:
        args["device"] = torch.device(args["device"])
    default("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Add some config for convenience
    default("log_path", config.LOG_PATH)
    default("ckpt_path", config.CKPT_PATH)
    default("model_path", config.MODEL_PATH)
    args["log_keys"] = config.LOG_KEYS
    args["log_index"] = config.LOG_INDEX
    args["best_metric_key"] = config.BEST_METRIC_KEY
    

    # 1. train
    method = args["method"]
    if method == "train":
        start_train(args)
    # 2. plot
    elif method == "plot":
        plot(args["name"])
    # MODEL EVALUATION
    else:
        # Create policy
        model_config = config.MODEL_CONFIGS[args["model_config"]]
        device = args["device"]
        policy = create_policy(model_config).to(device)

        # Load policy
        load_model(args["model_path"], args["name"], policy, None)

        # 3. play
        if method == "play":
            play(lambda: create_env(graphics=True), policy, timestamps=args["test_timestamps"])
        # 4. test
        elif method == "test":
            test(create_env, policy, timestamps=args["test_timestamps"], workers=args["workers"])