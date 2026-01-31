from torchrl.envs import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance

class UnnestTransform(Transform):
    def __init__(self, nested_key, out_keys, in_keys_inv):
        self.nested_key = nested_key
        self.out_keys = out_keys
        self.in_keys_inv = in_keys_inv
        self.in_keys = [(nested_key, key) for key in out_keys]
        self.out_keys_inv = [(nested_key, key) for key in in_keys_inv]
        super().__init__(in_keys=self.in_keys, out_keys=out_keys, in_keys_inv=self.in_keys_inv, out_keys_inv=self.out_keys_inv)        

    def _call(self, tensordict):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key in tensordict.keys(True):
                tensordict.rename_key_(in_key, out_key)
        if self.nested_key in tensordict.keys(True):
            tensordict.exclude(self.nested_key, inplace=True)
        return tensordict
    
    forward = _call

    def _reset(self, tensordict, tensordict_reset):
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _inv_call(self, tensordict):
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            if in_key in tensordict.keys(True):
                tensordict.rename_key_(in_key, out_key)
        return tensordict
    

    def _transform_spec(self, spec):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            if in_key in spec.keys(True):
                spec[out_key] = spec[in_key]
                del spec[in_key]
        if self.nested_key in spec.keys(True):
            del spec[self.nested_key]
        return spec
    
    def _transform_spec_inv(self, spec):
        for out_key, in_key in zip(self.in_keys_inv, self.out_keys_inv):
            if in_key in spec.keys(True):
                spec[out_key] = spec[in_key]
                del spec[in_key]
        return spec

    transform_observation_spec = _transform_spec
    transform_action_spec = _transform_spec_inv
    transform_reward_spec = _transform_spec
    transform_done_spec = _transform_spec
