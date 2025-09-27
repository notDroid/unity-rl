import os
import time
import tempfile
import pandas as pd

import torch
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.collectors.utils import split_trajectories
from torch.distributions import Normal, Independent

import config

#
##
#### LOSS FUNCTION UTILS (Specific to this project, see train.py for explanation)
##
#

def make_loss_module(policy, value, epsilon, entropy_coef, gamma, lmbda):
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value,
        clip_epsilon=epsilon,
        entropy_coeff=entropy_coef,
        normalize_advantage=True,
    )
    
    loss_module.set_keys(
        action=("agents", config.ACTION_KEY),
        sample_log_prob=("agents", "log_prob"),
        value=("agents", "state_value"),

        advantage=("agents", "advantage"),
        value_target=("agents", "value_target"),

        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
        # no truncated needed
    )

    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)

    return loss_module

# Uses loss module output tensordict and converts into a python dict with metrics
def loss_dict(loss_data, weight):
    # Hard coded keys, values
    keys = ["value_loss", "explained_variance", "policy_loss", "kl_approx", "clip_fraction", "ESS"]
    values = ["loss_critic", "explained_variance", "loss_objective", "kl_approx", "clip_fraction", "ESS"]

    return {
        key: (loss_data[value].detach().mean().item(), weight) for key, value in zip(keys, values)
    }

'''
COMPUTES ENVIRONMENT METRICS:
averages of: return, episode_length, entropy, action_std
'''
def compute_trajectory_metrics(tensordict_data):
    traj_data = split_trajectories(tensordict_data)

    reward = traj_data["next", "agents", "reward"] # [Tr, T, A, 1]
    if "collector" in traj_data:
        mask = traj_data["collector", "mask"].to(reward.dtype) # [Tr, T]
    elif "mask" in traj_data:
        mask = traj_data["mask"].to(reward.dtype) # [Tr, T]
    else:
        raise KeyError("No mask field found in:", tensordict_data)

    # 1. [Tr, T, A, 1] --(per episode return)--> [Tr, A] --(average return)--> float
    average_return = (reward * (mask.unsqueeze(-1).unsqueeze(-1))).sum(dim=(-3,-1)).mean().cpu().item()

    # 2. [Tr, T] --(episode length per trajectory (minumum of all agents))--> [Tr] --(average episode length)
    average_episode_length = mask.sum(dim=1).mean().cpu().item()
    
    # 3. Calculate entropy from dist, instead of sampled log_p, more accurate approximation
    loc, scale = traj_data["agents", "loc"], traj_data["agents", "scale"].clamp_min(1e-6) # clamp avoids nans in log(scale)
    loc, scale = loc.reshape(-1, loc.shape[-1]), scale.reshape(-1, scale.shape[-1]) # [Tr, T, A, action_dim]
    # Multivariate Normal [B, action_dim]
    dist = Independent(Normal(loc, scale), 1) 
    # Masked entropy [Tr, T, A]
    entropy = dist.entropy().reshape(*mask.shape, -1) * mask.unsqueeze(-1)
    # [Tr, T, A] --(mean entropy per agent (along the same timestep))--> [Tr, T]
    entropy = entropy.mean(dim=-1)
    # Total timesteps = mask.sum(), [Tr, T] --(average entropy)--> float
    entropy = (entropy.sum() / mask.sum()).cpu().item()

    # 4. action_std [T, A, action_dim]
    action_std = tensordict_data["agents", config.ACTION_KEY].std().cpu().item()

    metrics = {
        "return": average_return,
        "episode_length": average_episode_length,
        "entropy": entropy,
        "action_std": action_std,
    }

    return metrics

class Stopwatch:
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()

    def end(self):
        return time.time() - self.start_time

#
##
#### SAVE PROGRESS UTIL
##
#

# Do the temporary replace trick for an atomic checkpoint update (will not get corrupted on sudden crashes)
def atomic_torch_save(state_obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path)) as tmp:
        torch.save(state_obj, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)

class Checkpointer:
    '''
    Checkpoints training progress. Uses atomic ops so no corruption of files occur.
    
    Args:
        - ckpt_path, name

    Usage:

    checkpointer = Checkpointer(ckpt_path=ckpt_path, name=name)
    checkpointer.reset()
    # training ...
    checkpointer.save_progress(metric_key="metric",
    state_obj={
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metric": metric
    })
    checkpoint = checkpointer.load_progress()
    # this is a dict containing values of (generation, metric) and state dict of others

    # save model, choose 'best' or 'latest, model_path is the directory, whats written is name.pt containing the keys specified from that state_obj.
    checkpointer.copy_model(ckpt_type='best', model_path=model_path, keys=("model",))
    '''
    def __init__(self, ckpt_path, name):
        self.path = ckpt_path
        self.name = name
        self.best_path = os.path.join(ckpt_path, f"{name}_best.ckpt")
        self.latest_path = os.path.join(ckpt_path, f"{name}_latest.ckpt")
        self.best_metric = self.get_best_metric()

    def reset(self):
        if os.path.exists(self.best_path):
            os.remove(self.best_path)
        if os.path.exists(self.latest_path):
            os.remove(self.latest_path)
        self.best_metric = float("-inf")
        
    def get_best_metric(self):
        if not os.path.exists(self.best_path):
            return float("-inf")
        try:
            checkpoint = torch.load(self.best_path, weights_only=True)
            return checkpoint["metric"]
        except:
            return float("-inf")

    def save_progress(self, metric_key, state_obj):
        atomic_torch_save(state_obj, self.latest_path)

        # Check if new best
        metric = state_obj[metric_key]
        if metric >= self.best_metric:
            atomic_torch_save(state_obj, self.best_path)
            self.best_metric = metric

    def load_progress(self):
        if not os.path.exists(self.latest_path):
            return None
        return torch.load(self.latest_path, weights_only=True)

    def copy_model(self, ckpt_type, model_path, keys):
        if ckpt_type == 'best':
            state_obj = torch.load(self.best_path, weights_only=True)
        elif ckpt_type == 'latest':
            state_obj = torch.load(self.latest_path, weights_only=True)
        else:
            raise KeyError(f"Invalid ckpt_type: {ckpt_type}. Expected: best or latest")

        state_obj = {key: state_obj[key] for key in keys}

        atomic_torch_save(state_obj, os.path.join(model_path, f"{self.name}.pt"))

def load_model(path, name, policy, value):
    path = os.path.join(path, f"{name}.pt")
    if not os.path.exists(path):
        raise KeyError(f"Path does not exist: {path}")
    model_states = torch.load(path, weights_only=True)
    if policy:
        policy.load_state_dict(model_states["policy_state_dict"])
    if value:
        value.load_state_dict(model_states["value_state_dict"])

def save_model(path, name, policy, value):
    path = os.path.join(path, f"{name}.pt")
    state_obj = {
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value.state_dict(),
        }
    atomic_torch_save(state_obj, path)

#
##
#### LOG PROGRESS UTIL
##
#

class WEWMA:
    def __init__(self, beta):
        self.N = 0.0
        self.D = 0.0
        self.beta = beta

    def update(self, w, x):
        self.N = w*x + self.beta * self.N
        self.D =  w  + self.beta * self.D
        return self.N / self.D

class Logger:
    '''
    Simple CSV Logger
    
    Args:
        - keys
        - log_path, name 
            - (optional) log_path is the log directory and must exist beforehand.
        - beta (for ewmas)

    Usage:

    logger = Logger(keys=keys, log_path=log_path, name=name) # (full_log_path = log_path/name.csv)
    # the history is retrieved on creation of the logger
    logger.reset() # reset the logger and log file if present
    # ...
    # Do some log ops
    # ...
    logger.next(print_row=True) # Writes this row, optionally prints it
    # ...
    history_df = logger.dataframe()

    Log Ops:
        logger.add({key: value}): sets column
        logger.sum({key: value}): adds from previous column
        logger.accumulate({key: value}): (weighted) EWMA of calls (this row only, does not weight over previous rows)
            - in accumulate providing a value tuple of (value, weight) uses that weight, otherwise no tuple means weight=1
    '''

    def __init__(self, keys, log_path=None, name=None, beta=0.90):
        # Data State
        self.keys = list(keys)
        self.set_keys = set(keys)
        self.df = pd.DataFrame(columns=self.keys)

        # For Building
        self.prev_row = {} # Helps with sums
        self.row = {}
        self.wewmas = {}
        self.beta = beta
        

        # Check for log path + name
        self.log_path = log_path
        if log_path and not name:
            raise KeyError("Log path provided without name")
        self.full_log_path = None

        # Check for existing logs
        if log_path:
            self.full_log_path = os.path.join(log_path, f"{name}.csv")

            # If it exists, read current data
            if os.path.exists(self.full_log_path):
                try:
                    self.df = pd.read_csv(self.full_log_path)
                    self._set_prev_row()
                except:
                    print("FAILED TO READ LOG FILE, STARTING FROM SCRATCH.")

    ### START ###

    def _set_prev_row(self):
        if len(self.df) == 0: 
            self.prev_row = {}
            return
        self.prev_row = self.df.iloc[-1].to_dict()

    def reset(self):
        # Ignore if no path
        if self.full_log_path is None:
            print("NO LOG PATH GIVEN: SKIPPING RESET")
            return

        # Try empty/create file (fails)
        with open(self.full_log_path, "w") as f:
            pass
        self.df = pd.DataFrame(columns=self.keys)
        self._set_prev_row()

    ### INTERNAL HELPERS ###

    def _check(self, key):
        if key not in self.set_keys:
            print(f"key: {key} not in keys: {self.keys}. SKIPPING KEY")
            return False
        return True
    
    @staticmethod
    def _add(dict_to, dict_from, key, value):
        dict_to[key] = (value + dict_from[key]) if key in dict_from else value

    def _update_wewma(self, key, w, x):
        if key not in self.wewmas: 
            self.wewmas[key] = WEWMA(self.beta)

        self.row[key] = self.wewmas[key].update(w, x)

    ### LOG OPS ###

    def sum(self, entries):
        for key, value in entries.items():
            if not self._check(key): continue
            
            if key in self.row: 
                # Add to current row
                self.row[key] += value
            else:
                # Add from prev row (no entry in current row)
                self._add(self.row, self.prev_row, key, value)


    def accumulate(self, entries):
        for key, value_data in entries.items():
            if not self._check(key): continue
            
            value, weight = value_data if isinstance(value_data, tuple) else (value_data, 1)
            self._update_wewma(key, weight, value)

    def add(self, entries):
        for key, value in entries.items():
            if not self._check(key): continue
            self.row[key] = value

    ### NEXT ###

    def next(self, print_row=False):
        # Append next row
        df_next_row = pd.DataFrame(self.row, columns=self.keys, index=[len(self.df)])

        if len(self.df) == 0:
            self.df = df_next_row
        else:
            self.df = pd.concat((self.df, df_next_row))
        
        # Reset state
        self.prev_row = self.row
        self.row = {}
        self.wewmas = {}

        # Print
        if print_row:
            print(df_next_row)

        # Write to csv
        if self.full_log_path:
            df_next_row.to_csv(self.full_log_path, mode='a', header=(len(self.df) == 1), index=False)

    ### ACCESS LOGS ###

    def dataframe(self):
        return self.df
    
    def last(self):
        return self.row
    
