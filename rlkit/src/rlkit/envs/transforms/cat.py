import torch
from torchrl.envs import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance
from torchrl.data.tensor_specs import Composite, Unbounded


class InvertibleCatTensors(Transform):
    invertible = True

    def __init__(self, in_keys, out_key, dim=-1, del_keys = True):
        super().__init__(in_keys=in_keys, out_keys=[out_key])
        if out_key in in_keys: raise Exception(f"out_key={out_key} cannot be in in_keys={in_keys}")
        self.in_keys = in_keys
        self.out_key = out_key
        self.dim = dim
        self._del_keys = del_keys
        self.in_key_dims = None # Set in transform_observation_spec

    def _call(self, next_tensordict):
        # Get in keys
        values = [next_tensordict.get(key, None) for key in self.in_keys]
        if any(value is None for value in values):
            raise Exception(
                f"InvertibleCatTensor failed, as it expected input keys ="
                f" {self.in_keys} but got a TensorDict with keys"
                f" {next_tensordict.keys(include_nested=True)}"
            )

        # Concat them
        out_tensor = torch.cat(values, dim=self.dim)
        next_tensordict.set(self.out_keys[0], out_tensor)
        if self._del_keys:
            next_tensordict.exclude(*self.in_keys, inplace=True)
        return next_tensordict

    forward = _call

    def _reset(self, tensordict, tensordict_reset):
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset
    
    def _inv_call(self, in_tensordict):
        cat = in_tensordict[self.out_keys[0]]
        index = 0

        # Set in keys (unconcat) if they were deleted
        if self._del_keys:
            for key, dim in zip(self.in_keys, self.in_key_dims):
                idx = torch.arange(index, index + dim)
                in_tensordict[key] = torch.index_select(cat, self.dim, idx)
                index += dim
        # Delete out key (concat)
        in_tensordict.exclude(self.out_keys[0], inplace=True)

        return in_tensordict

    def transform_observation_spec(self, observation_spec):
        # check that all keys are in observation_spec
        if len(self.in_keys) > 1 and not isinstance(observation_spec, Composite):
            raise ValueError(
                "CatTensor cannot infer the output observation spec as there are multiple input keys but "
                "only one observation_spec."
            )

        if isinstance(observation_spec, Composite) and len(
            [key for key in self.in_keys if key not in observation_spec.keys(True)]
        ):
            raise ValueError(
                "CatTensor got a list of keys that does not match the keys in observation_spec. "
                "Make sure the environment has an observation_spec attribute that includes all the specs needed for CatTensor."
            )

        if not isinstance(observation_spec, Composite):
            # by def, there must be only one key
            return observation_spec

        # Input keys
        keys = self.in_keys
        # Save dimensions for inverse
        self.in_key_dims = [
            observation_spec[key].shape[self.dim]
            if observation_spec[key].shape
            else 1
            for key in keys
        ]
        # Concat length
        sum_shape = sum(self.in_key_dims)

        # Set new spec
        spec0 = observation_spec[keys[0]]
        out_key = self.out_keys[0]
        shape = list(spec0.shape)
        device = spec0.device
        shape[self.dim] = sum_shape
        shape = torch.Size(shape)
        observation_spec[out_key] = Unbounded(
            shape=shape,
            dtype=spec0.dtype,
            device=device,
        )
        # Delete old
        if self._del_keys:
            for key in self.in_keys:
                if key in observation_spec.keys(True):
                    del observation_spec[key]
        return observation_spec