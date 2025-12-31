import os
import numpy as np
import tempfile
import torch
from .utils import hf_hub_download_and_copy
from huggingface_hub import upload_file

def atomic_torch_save(state_obj, path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path)) as tmp:
        torch.save(state_obj, tmp, **kwargs)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)

class Checkpointer:
    '''
    Checkpoints training progress. 
    Uses atomic ops so no corruption of files occur.
    Saves latest and best checkpoints.
    
    Args:
        - ckpt_path, name

    Usage:

    ```
    checkpointer = Checkpointer(ckpt_path=ckpt_path, name=name, metric_key="metric")
    checkpointer.reset()
    
    # training ...

    checkpointer.save_progress(
        state_obj={
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metric": metric
        }
    )


    # Retrieve state object
    checkpoint = checkpointer.load_progress(generation="latest", weights_only=True, map_location="cuda")

    # copy to model directory
    checkpointer.copy_model(model_path=model_path, generation="best", keys=("model_state_dict",))
    ```
    '''
    def __init__(self, ckpt_path, name, metric_key = None):
        self.path = ckpt_path
        self.name = name
        self.best_path = os.path.join(ckpt_path, f"{name}_best.ckpt")
        self.latest_path = os.path.join(ckpt_path, f"{name}_latest.ckpt")
        self.metric_key = metric_key
        self.best_metric = float("-inf")

        # Make dir
        os.makedirs(ckpt_path, exist_ok=True)

    def reset(self):
        ls = os.listdir(self.path)
        for file_name in ls:
            # Look for files with prefix
            if not file_name.startswith(self.name + '_'): continue

            # Delete
            os.remove(os.path.join(self.path, file_name))
        self.best_metric = float("-inf")
        
    def _get_best_metric(self):
        if not os.path.exists(self.best_path):
            return float("-inf")
        checkpoint = torch.load(self.best_path, weights_only=False, map_location='meta')
        return checkpoint[self.metric_key]

    def save_progress(self, state_obj, **kwargs):
        # Add new latest
        atomic_torch_save(state_obj, self.latest_path, **kwargs)

        # Check if new best
        if not self.metric_key: return
        if self.metric_key not in state_obj: return
        metric = state_obj[self.metric_key]
        if metric is None: return
        if metric >= self._get_best_metric():
            atomic_torch_save(state_obj, self.best_path, **kwargs)

    def load_progress(self, generation="latest", **kwargs):
        if generation == "latest": path = self.latest_path
        elif generation == "best": path = self.best_path
        else: raise RuntimeError(f"Invalid ckpt type: {generation} not in (\"latest\", \"best\")")

        if not os.path.exists(path): return None
        return torch.load(path, weights_only=False, **kwargs)

    def copy_model(self, model_path, generation="latest", keys = None, **kwargs):
        if generation == "latest": path = self.latest_path
        elif generation == "best": path = self.best_path
        else: raise RuntimeError(f"Invalid ckpt type: {generation} not in (\"latest\", \"best\")")

        state_obj = torch.load(path, weights_only=False, **kwargs)

        if keys is not None:
            state_obj = {key: state_obj[key] for key in keys}

        atomic_torch_save(state_obj, os.path.join(model_path, f"{self.name}.pt"))

class MultiVersionCheckpointer:
    '''
    Checkpoints training progress. 
    Uses atomic ops so no corruption of files occur.
    Keeps multiple past versions, optionally with a log scale.
    
    Args:
        - ckpt_path, name, levels, base_interval, interval_scale
            - levels is how many checkpoints to maintain (excluding best)
            - base is size of base interval
            - scale is how much larger the next interval is
        
    interval sizes = [1, B, B(s), B(s^2), ...]
    intervals = ckpt_gen - [0, 1), [1, 1+B), [1+B, 1+B(1+s)), [1+B(1+s), 1+B(1+s+s^2)), ...

    Updated Usage (past what's detailed in Checkpointer above):

    ```
    # Go back to version 1234 associated with ckpt_path/name_1234.ckpt
    checkpointer.revert(generation=1234) 
    ```
    '''
    def __init__(self, ckpt_path, name, levels=0, base_interval=1, interval_scale=1, metric_key=None):
        self.path = ckpt_path
        self.name = name
        self.base_interval = base_interval
        self.interval_scale = interval_scale
        self.levels = levels

        self.path_fn = lambda x: os.path.join(self.path, f"{self.name}_{x}.ckpt")
        self.best_path = self.path_fn('best')
        self.best_metric = float("-inf")
        self.metric_key = metric_key

    def _intervals(self, generation):
        '''Returns intervals in [closed, open) format from latest to oldest (descending in gen)'''
        levels = np.arange(self.levels)
        sizes = self.base_interval * np.pow(self.interval_scale, levels)
        sizes = np.insert(sizes, 0, 1)
        sizes = np.insert(sizes, 0, 0)
        intervals = generation - sizes.cumsum()
        intervals = intervals[intervals > 0]
        intervals = np.insert(intervals, len(intervals), 0)
        # print("intervals:", intervals)
        return intervals

    def _gens(self):
        '''Gets list of generations from ckpt dir in ascending order'''
        ls = os.listdir(self.path)
        gens = []
        for file_name in ls:
            # Only from our name
            if not file_name.startswith(self.name + '_'): continue
            gen = file_name.split('_')[-1].split('.')[0]

            # Add only integer generations (so excluding best)
            if gen.isnumeric():
                gens.append(int(gen))
        gens = np.sort(gens)
        # print("gens:", gens)
        return gens

    def _latest(self):
        gens = self._gens()
        if len(gens) == 0: return 0 
        return max(gens)

    def _enforce(self, generation):
        '''Enforces 1 ckpt per interval condition, best is ignored'''
        
        gens = self._gens()
        intervals = self._intervals(generation)
        i = len(intervals) - 1
        occupied = False

        for gen in gens:
            gen_path = self.path_fn(gen)

            while True:
                # CASE 1: above max generation -> delete
                if gen > generation: 
                    os.remove(gen_path)
                    break

                # CASE 2: within interval [start, end) (where start > end)
                start, end = intervals[i-1], intervals[i]
                if start >= gen and gen > end:
                    # CASE 1: oldest in interval -> keep
                    if not occupied: 
                        occupied = True
                        break
                    # CASE 2: not oldest in interval -> delete
                    os.remove(gen_path)
                    break

                # CASE 3: not within interval -> must be in a future interval
                i -= 1
                occupied = False

    def _get_best_metric(self):
        if not os.path.exists(self.best_path):
            return float("-inf")
        checkpoint = torch.load(self.best_path, weights_only=False, map_location='meta')
        return checkpoint[self.metric_key]

    def reset(self):
        ls = os.listdir(self.path)
        for file_name in ls:
            # Look for files with prefix
            if not file_name.startswith(self.name + '_'): continue

            # Delete
            os.remove(os.path.join(self.path, file_name))
        self.best_metric = float("-inf")

    def revert(self, generation="latest"):
        '''
        Deletes checkpoints past current one
        '''
        path = self.path_fn(generation)
        
        # Check for existance
        if (generation != 'latest') and (not os.path.exists(path)): 
            print(f"No file found: {path}")
            return

        # Numeric or best
        if generation != 'latest':
            # Retrieve checkpoint if best
            if generation == 'best':
                checkpoint = torch.load(path, weights_only=False, map_location='meta')
                generation = int(checkpoint["checkpoint_generation"])
            
            # Enforce invariant
            self._enforce(generation)
        
        # Choose latest
        generation = self._latest()
        if generation == 0: 
            print(f"No checkpoints, starting at 0")
            self.reset() # Clears best if present
            return
        
        # Enforce invariant
        self._enforce(generation)
        return

    def save_progress(self, state_obj, **kwargs):
        # Add new checkpoint
        generation = self._latest() + 1
        state_obj["checkpoint_generation"] = generation
        atomic_torch_save(state_obj, self.path_fn(generation), **kwargs)

        # Enforce invariant
        self._enforce(generation)

        # Check if new best
        if not self.metric_key: return
        if self.metric_key not in state_obj: return
        metric = state_obj[self.metric_key]
        if metric is None: return
        if metric >= self._get_best_metric():
            atomic_torch_save(state_obj, self.best_path, **kwargs)

    def load_progress(self, generation="latest", **kwargs):
        if generation == "latest": path = self.path_fn(self._latest())
        elif generation == "best": path = self.best_path
        else: path = self.path_fn(generation)

        if not os.path.exists(path): return None
        return torch.load(path, weights_only=False, **kwargs)

    def copy_model(self, model_path, generation="latest", keys=None, **kwargs):
        if generation == "latest": path = self.path_fn(self._latest())
        elif generation == "best": path = self.best_path
        else: path = self.path_fn(generation)

        state_obj = torch.load(path, weights_only=False, **kwargs)

        if keys is not None:
            state_obj = {key: state_obj[key] for key in keys}

        atomic_torch_save(state_obj, os.path.join(model_path, f"{self.name}.pt"))

class HFCheckpointer(Checkpointer):
    '''
    Checkpoints training progress using HuggingFace Hub.
    Saves latest checkpoint.
    
    Args:
        - repo_id, ckpt_path, name, (optional) metric_key
    '''

    def __init__(self, repo_id, ckpt_path, name, metric_key=None):
        super().__init__(ckpt_path, name, metric_key)
        self.repo_id = repo_id

    def sync_to_hub(self, **kwargs):
        upload_file(path_or_fileobj=self.latest_path, path_in_repo=self.latest_path, repo_id=self.repo_id, repo_type="model", **kwargs)

    def sync_from_hub(self, **kwargs):
        hf_hub_download_and_copy(local_path=self.latest_path, repo_id=self.repo_id, remote_path=self.latest_path, **kwargs)