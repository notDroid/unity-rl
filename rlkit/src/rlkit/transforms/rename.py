from torchrl.envs import Transform

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