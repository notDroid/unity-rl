from torchrl.envs import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance
import random

class UnityRandomizerTransform(Transform):
    def __init__(self, interval, env_param_channel, params, verbose=False):
        super().__init__()
        self._interval = interval
        self._env_param_channel = env_param_channel
        self._params = params
        self.t = 0
        self._verbose = verbose

    def _call(self, next_tensordict):
        if (self.t % self._interval) == 0:
            for key, (start, end) in self._params.items():
                value = random.uniform(start, end)
                self._env_param_channel.set_float_parameter(key, value)
                if self._verbose:
                    print(f"Set {key} to {value}")
        self.t += 1
        return next_tensordict
    
    forward = _call
    
    def _reset(self, tensordict, tensordict_reset):
        self.t = 0
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset