import os
import numpy as np
import pandas as pd
import tempfile

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

class Logger:
    '''
    Simple CSV Logger
    
    Args:
        - keys
        - (optional) log_path, name 
            - log_path is the log directory and must exist beforehand.
        - beta (for ewmas)

    Usage:

    ```
    logger = Logger(keys=keys, log_path=log_path, name=name) # (full_log_path = log_path/name.csv)
    # Retrieve history with revert, you can also revert to a past state by specifying key, value
    logger.revert()
    # Reset logger
    logger.reset()
    
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

    def __init__(self, keys, log_path=None, name=None, beta=0.95):
        # Data State
        self.keys = list(keys)
        self.set_keys = set(keys)
        self.df = pd.DataFrame(columns=self.keys)

        # For Building
        self.prev_row = {} # Helps with sums
        self.row = {}
        self.wewmas = {}
        self.beta = beta
        self.wavgs = {}
        self.sums = {}
        

        # Check for log path + name
        self.log_path = log_path
        if log_path and not name:
            raise KeyError("Log path provided without name")
        self.full_log_path = None

        if log_path:
            self.full_log_path = os.path.join(log_path, f"{name}.csv")

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

    def revert(self, key=None, value=None):
        if self.full_log_path is None: return

        # Doesn't exist
        if not os.path.exists(self.full_log_path):
            print("FAILED TO READ LOG FILE, STARTING FROM SCRATCH.")
            self.reset() 
            return
        
        # Attempt to read file
        try:
            self.df = pd.read_csv(self.full_log_path)
        except:
            print("FAILED TO READ LOG FILE, STARTING FROM SCRATCH.")
            self.reset()
            return

        # Use latest
        if key is None:
            self._set_prev_row()
            return
        
        # Revert to some specified (key, value) search dataframe backwards
        index = np.where(self.df[key] == value)

        # Not found
        if len(index) == 0: 
            raise KeyError("(key, value) not found in dataframe")
        
        # Save
        index = index[-1]
        self.df = self.df.iloc[:index + 1, :]
        atomic_replace_df(self.df, self.full_log_path)

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

    def acc(self, entries, mode='sum'):
        for key, value_data in entries.items():
            if not self._check(key): continue
            
            value, weight = value_data if isinstance(value_data, tuple) else (value_data, 1)
            if mode == 'ema': self._update_wewma(key, weight, value)
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

        # Write to csv
        if self.full_log_path:
            df_next_row.to_csv(self.full_log_path, mode='a', header=(len(self.df) == 1), index=False)

    ### ACCESS LOGS ###

    def dataframe(self):
        return self.df
    
    def last(self):
        return self.row
    
