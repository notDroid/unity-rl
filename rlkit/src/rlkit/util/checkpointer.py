import os
import numpy as np
import tempfile
import torch

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

class MultiVersionCheckpointer:
    '''
    Checkpoints training progress. 
    Uses atomic ops so no corruption of files occur.
    Keeps multiple past versions, optionally with a log scale.
    
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
    def __init__(self, ckpt_path, name, levels=1, base_interval=1, interval_scale=1):
        self.path = ckpt_path
        self.name = name
        self.base_interval = base_interval
        self.interval_scale = interval_scale
        self.levels = levels

        self.path_fn = lambda x: os.path.join(self.path, f"{self.name}_{x}")

        self.generation = 0

    def _gens(self):
        ls = os.listdir(self.path)
        gens = []
        for file_name in ls:
            if file_name.startswith(self.name + '_'):
                gen = int(file_name.split('_')[-1])
                gens.append(gen)
        return sorted(gens)

    def revert(self, generation='latest'):
        path = self.path_fn(generation)
        
        # Check for existance
        if (generation != 'latest') and (not os.path.exists(path)): 
            self.generation = 0
            print(f"No file found: {path}")

        if generation != 'latest':
            # Retrieve checkpoint
            if generation == 'best':
                checkpoint = torch.load(path, weights_only=True)
                self.generation = checkpoint["checkpoint_generation"]
            else:
                self.generation = generation
            
            # Delete generations past current one
            gens = self._gens()
            for gen in gens:
                if gen <= self.generation: continue
                os.remove(self.path_fn(gen))
            return
        
        gens = self._gens()
        if len(gens) == 0: 
            self.generation = 0
            print(f"No checkpoints, starting at 0")
            return
        
        self.generation = max(gens)
        return
        
    def _interval_ends(self):
        levels = np.arange(self.levels)
        sizes = self.base_interval * np.pow(self.interval_scale, levels)
        ends = self.generation - sizes
        ends = ends[ends >= 0]
        ends = np.insert(ends, 0, -1)
        return ends

    def save_progress(self, metric_key, state_obj):
        self.generation += 1
        

        # Shift and delete checkpoints
        gens = self._gens()
        ends = self._interval_ends()
        i = 0
        for end in ends:
            if i >= len(gens): break
            occupied = True
            i += 1
            if i >= len(gens): break
            occupied = True



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