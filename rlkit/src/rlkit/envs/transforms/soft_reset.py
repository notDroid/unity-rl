from torchrl.envs import EnvBase

class SoftResetWrapper(EnvBase):
    """Only the very first reset() hits the wrapped env.
    Later reset() calls return a cached 'reset-like' TensorDict."""
    def __init__(self, env: EnvBase):
        super().__init__(device=env.device, batch_size=env.batch_size)
        self.env = env
        self._passthrough_specs()
        self._last = None

    def _step(self, tensordict):
        self._last = self.env._step(tensordict)
        return self._last

    def _reset(self, tensordict=None, **kwargs):
        # Real Unity reset (only first reset)
        if self._last is None or tensordict is None or "_reset" not in tensordict:
            self._last = self.env._reset(tensordict, **kwargs)
            return self._last
        
        # Soft Reset: Unity Autoresets
        return self._make_reset_out(self._last.copy())
    
    def _make_reset_out(self, tensordict):
        # 1. Clear Dones
        tensordict = tensordict.update(tensordict.select(*self.env.done_keys) * False)
        # 2. Exclude Rewards
        tensordict = tensordict.exclude(*self.env.reward_keys, inplace=True)
        return tensordict
    
    # Passthrough
    def _set_seed(self, *args, **kwargs): return self.env.set_seed(*args, **kwargs)
    def _passthrough_specs(self):
        self.observation_spec = self.env.observation_spec
        self.action_spec = self.env.action_spec
        self.reward_spec = self.env.reward_spec
        self.done_spec = self.env.done_spec