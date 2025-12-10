from torchrl.envs import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance

class RenameAction(Transform):
    invertible=True
    def __init__(self, prev_key, new_key):
        super().__init__(in_keys_inv=[new_key], out_keys_inv=[prev_key])
        self.prev_key = prev_key
        self.new_key = new_key

    def _inv_call(self, in_tensordict):
        in_tensordict.rename_key_(self.new_key, self.prev_key)
        return in_tensordict

    def transform_action_spec(self, action_spec):
        action_spec[self.new_key] = action_spec[self.prev_key]
        del action_spec[self.prev_key]
        return action_spec
    
class InvertibleRename(Transform):
    # Only for forward direction (not for actions, use RenameAction for that instead)
    invertible=True

    def __init__(self, prev_keys, new_keys):
        assert len(prev_keys) == len(new_keys), f"invalid InvertibleRename input keys: {prev_keys} -> {new_keys}"
        super().__init__(in_keys=prev_keys, out_keys=new_keys, in_keys_inv=new_keys, out_keys_inv=prev_keys)
        self.combined = list(zip(prev_keys, new_keys))

    def _call(self, out_tensordict):
        for prev_key, new_key in self.combined:
            if prev_key in out_tensordict.keys(True):
                out_tensordict.rename_key_(prev_key, new_key)
        return out_tensordict
    
    forward = _call

    def _reset(self, tensordict, tensordict_reset):
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _inv_call(self, in_tensordict):
        for prev_key, new_key in self.combined:
            if new_key in in_tensordict.keys(True):
                in_tensordict.rename_key_(new_key, prev_key)
        return in_tensordict

    def _transform_spec(self, spec):
        for prev_key, new_key in self.combined:
            if prev_key in spec.keys(True):
                spec[new_key] = spec[prev_key]
                del spec[prev_key]
        return spec
    
    transform_observation_spec = _transform_spec
    transform_reward_spec = _transform_spec
    transform_done_spec = _transform_spec