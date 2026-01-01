import numpy as np
import pandas as pd

import os
import tempfile
import shutil

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from huggingface_hub import HfApi, snapshot_download

def atomic_replace_df(df, path):
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path)) as tmp:
        df.to_csv(tmp.name, mode='w', header=True, index=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)

'''
LOG PROGRESS UTIL
'''

class WEWMA:
    def __init__(self, beta):
        self.N = 0.0
        self.D = 0.0
        self.beta = beta
        if beta is None: raise KeyError("Expected EWMA argument: beta")

    def update(self, w, x):
        self.N = w*x + self.beta * self.N
        self.D =  w  + self.beta * self.D
        return self.N / self.D

class WAVG:
    def __init__(self):
        self.n = 0
        self.x = 0

    def update(self, w, x):
        n_prev = self.n
        self.n += w
        self.x = self.x * (n_prev / self.n) + x * (w / self.n)
        return self.x

class LoggerBase:
    '''
    Simple Logger
    
    Args:
        - keys
        - beta (optional, for ewmas)

    Usage:

    ```
    logger = Logger(keys=keys)
    
    # ...
    # Do some log ops
    # ...
    logger.next(print_row=True) # Writes this row, optionally prints it
    history_df = logger.dataframe()

    Log Ops:
        logger.add({key: value}): sums across previous rows
        logger.acc({key: value} or {key: (value, weight)}, mode='ema' or 'avg' or 'sum'):
            -  if weight not provided, defaults to 1
    ```
    '''

    def __init__(self, keys):
        # Data State
        self.keys = list(keys)
        self.set_keys = set(keys)
        self.df = pd.DataFrame(columns=self.keys)

        # For Building
        self.prev_row = {} # Helps with sums
        self.row = {}
        self.wewmas = {}
        self.wavgs = {}
        self.sums = {}

    ### INTERNAL HELPERS ###

    def _check(self, key):
        if key not in self.set_keys:
            print(f"key: {key} not in keys: {self.keys}. SKIPPING KEY")
            return False
        return True
    
    @staticmethod
    def _add(dict_to, dict_from, key, value):
        dict_to[key] = (value + dict_from[key]) if key in dict_from else value

    def _update_wewma(self, key, w, x, beta=None):
        if key not in self.wewmas: 
            self.wewmas[key] = WEWMA(beta)

        self.row[key] = self.wewmas[key].update(w, x)

    def _update_wavg(self, key, w, x):
        if key not in self.wavgs: 
            self.wavgs[key] = WAVG()

        self.row[key] = self.wavgs[key].update(w, x)

    def _update_sum(self, key, x):
        if key not in self.sums: 
            self.sums[key] = x
        else:
            self.sums[key] += x
        self.row[key] = self.sums[key]

    ### LOG OPS ###

    def acc(self, entries, mode='sum', **kwargs):
        for key, value_data in entries.items():
            if not self._check(key): continue
            
            value, weight = value_data if isinstance(value_data, tuple) else (value_data, 1)
            if mode == 'ema': self._update_wewma(key, weight, value, **kwargs)
            elif mode == 'avg': self._update_wavg(key, weight, value)
            elif mode =='sum': self._update_sum(key, value)
            else: raise KeyError(f"Expected a mode in (\"ema\", \"avg\", \"sum\") not {mode}")

    def add(self, entries):
        for key, value in entries.items():
            if not self._check(key): continue
            self._add(self.row, self.prev_row, key, value)

    ### NEXT ###

    def next(self, print_row=False):
        # Append next row
        row = self.row
        df_next_row = pd.DataFrame(self.row, columns=self.keys, index=[len(self.df)])

        if len(self.df) == 0:
            self.df = df_next_row
        else:
            self.df = pd.concat((self.df, df_next_row))
        
        # Reset state
        self.prev_row = self.row
        self.row = {}
        self.wewmas = {}
        self.wavgs = {}
        self.sums = {}

        # Print
        if print_row:
            print(df_next_row)

        return row

    ### ACCESS LOGS ###

    def dataframe(self):
        return self.df
    
    def last(self):
        return self.row
    
    def reset(self): raise NotImplementedError("Incorrect Usage")
    def revert(self, key=None, value=None): raise NotImplementedError("Incorrect Usage")
    
class CSVLogger(LoggerBase):
    '''
    Log to CSV FILE at log_path/name

    Usage:
        - revert(), to latest index or specified (key, value)
        - reset(), reset log file
        - log use next() + log ops
    '''
    def __init__(self, keys, log_path, name=None):
        super().__init__(keys)
        self.log_path = os.path.join(log_path, name) if name else log_path


    def next(self, print_row=False):
        row = super().next(print_row)
        df_row = self.df.iloc[[-1]]
        df_row.to_csv(self.log_path, mode='a', header=(len(self.df) == 1), index=False)

    ### START ###

    def _set_prev_row(self):
        if len(self.df) == 0: 
            self.prev_row = {}
            return
        self.prev_row = self.df.iloc[-1].to_dict()

    def reset(self):
        # Empty/create file
        with open(self.log_path, "w") as f:
            pass
        self.df = pd.DataFrame(columns=self.keys)
        self._set_prev_row()

    def revert(self, key=None, value=None):
        # Doesn't exist
        if not os.path.exists(self.log_path):
            print("FAILED TO FIND LOG FILE, STARTING FROM SCRATCH.")
            self.reset() 
            return
        
        # Attempt to read file
        try:
            self.df = pd.read_csv(self.log_path)
        except:
            print("FAILED TO READ LOG FILE, STARTING FROM SCRATCH.")
            self.reset()
            return
        if len(self.df) == 0: 
            print("NOTHING TO READ IN LOG FILE, STARTING FROM SCRATCH.")
            self.reset()
            return

        # Use latest
        if key is None:
            self._set_prev_row()
            return
        
        # Revert to some specified (key, value) search dataframe backwards
        index = np.where(self.df[key] == value)[0]

        # Not found
        if len(index) == 0: 
            raise KeyError("(key, value) not found in dataframe")
        
        # Save
        index = index[-1]
        self.df = self.df.iloc[:index + 1, :]
        self._set_prev_row()
        atomic_replace_df(self.df, self.log_path)

class TensorBoardLogger(LoggerBase):
    '''
    Log to tensorboard at log_dir

    Usage:
        - revert(), to latest index or specified (key, value)
        - reset(), reset log dir
        - log use next() + log ops
    '''
    def __init__(self, keys, log_dir):
        # Infer keys from existing log dir if not provided
        if keys is None:
            if not os.path.exists(log_dir):
                raise KeyError("log_dir does not exist, must provide keys to create new logger")
            try:
                event_acc = EventAccumulator(path=log_dir)
                event_acc.Reload()
                keys = event_acc.Tags()['scalars']
            except:
                raise KeyError("Failed to get keys from existing log_dir, must provide keys to create new logger")
        
        super().__init__(keys)
        
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)

    def next(self, print_row=False):
        row = super().next(print_row)

        # Log to tensorboard
        step = len(self.df) - 1
        for key, value in row.items():
            self.writer.add_scalar(key, value, step)
        self.writer.flush()

        return row
    
    def _set_prev_row(self):
        if len(self.df) == 0: 
            self.prev_row = {}
            return
        self.prev_row = self.df.iloc[-1].to_dict()

    def reset(self):
        self.writer.close()
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.df = pd.DataFrame(columns=self.keys)
        self._set_prev_row()

    def revert(self, key=None, value=None):
        # Doesn't exist
        if not os.path.exists(self.log_dir):
            print("FAILED TO FIND LOG DIR, STARTING FROM SCRATCH.")
            self.reset() 
            return
        
        # Attempt to read events
        ea = EventAccumulator(self.log_dir)
        ea.Reload()
        tags = ea.Tags()['scalars']
        if not tags:
            print("NOTHING TO READ IN LOG DIR, STARTING FROM SCRATCH.")
            self.reset()
            return

        # Find max step
        max_step = max(max(e.step for e in ea.Scalars(tag)) for tag in tags)

        # Build df
        self.df = pd.DataFrame(index=range(max_step + 1), columns=self.keys)
        for tag in tags:
            if tag not in self.set_keys: continue
            events = ea.Scalars(tag)
            for e in events:
                self.df.at[e.step, tag] = e.value

        # Use latest
        if key is None:
            self._set_prev_row()
            return
        
        # Revert to some specified (key, value) search dataframe backwards
        if key not in self.df.columns:
            raise KeyError(f"Key {key} not found in logs")

        index = np.where(self.df[key] == value)[0]

        # Not found
        if len(index) == 0: 
            raise KeyError("(key, value) not found in dataframe")
        
        # Save
        index = index[-1]
        self.df = self.df.iloc[:index + 1, :]
        self._set_prev_row()

        # Rewrite tensorboard logs
        self.writer.close()
        shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            for k, v in row.items():
                if pd.notna(v):
                    self.writer.add_scalar(k, v, i)

class HFTBLogger(TensorBoardLogger):
    '''
    Log to tensorboard at local log_dir and sync with huggingface

    Usage:
        - revert(), to latest index or specified (key, value)
        - reset(), reset log dir
        - log use next() + log ops

        NEW:
        - sync_from_hub, get logs from hf
        - sync_to_hub, send logs to hf
    '''
    def __init__(self, keys, log_dir, repo_id, repo_subfolder):
        super().__init__(keys, log_dir)

        self.repo_id = repo_id
        self.repo_subfolder = repo_subfolder
        self.api = HfApi()

    def sync_from_hub(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="model",
                local_dir=temp_dir,
                local_dir_use_symlinks=False,
                allow_patterns=f"{self.repo_subfolder}/**"  # Download only subfolder contents
            )
            subfolder_path = os.path.join(temp_dir, self.repo_subfolder)
            if os.path.exists(subfolder_path):
                # Reset log_dir and tensorboard
                super().reset()

                # Copy contents to self.log_dir
                for item in os.listdir(subfolder_path):
                    shutil.move(os.path.join(subfolder_path, item), self.log_dir)

                # Update tensorboard writer
                super().revert()
            else:
                print(f"Subfolder '{self.repo_subfolder}' not found in repo.")
    
    def sync_to_hub(self):
        self.api.upload_folder(
            folder_path=self.log_dir,
            path_in_repo=self.repo_subfolder,
            repo_id=self.repo_id,
            commit_message=f"Log sync: Step {len(self.df)}",
        )

    