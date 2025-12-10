import torch
import torch.nn.functional as F
from torchrl.envs import Transform
from torchrl.data.tensor_specs import BoxList, OneHot, Composite
from torchrl.envs.transforms.transforms import _apply_to_composite_inv

class FlattenMultiOneHot(Transform):
    def __init__(self, action_key):
        super().__init__(in_keys_inv=[action_key], out_keys_inv=[action_key])
        self._initialized = False

    def _inv_call(self, tensordict):
        if not self._initialized: self._set_dims()

        # Get One Hot Actions
        action_key = self.parent.action_key
        action_data = tensordict[action_key]

        # Convert to Categorical (get index)
        index = torch.argmax(action_data, dim=-1, keepdims=True)

        # Unravel to each mini category index, D x [B, 1]
        multi_cat = torch.unravel_index(index, self._dims)

        # Set each mini category one hot vector, [B, C], C=sum(C_i)
        multi_one_hot = []
        for cat, dim in zip(multi_cat, self._dims):
            # [B, C_i]
            one_hot = F.one_hot(cat.squeeze(-1), num_classes=dim)
            multi_one_hot.append(one_hot)

        tensordict[action_key] = torch.cat(multi_one_hot, dim=-1)
        return tensordict

    def _set_dims(self):
        action_spec = self.parent.action_spec
        if isinstance(action_spec, Composite):
            moh = action_spec[self.in_keys_inv[0]]
        else:
            moh = action_spec
        self._dims = [box.n for box in moh.space.boxes]
        self._dim_sum = sum(self._dims)
        self._initialized = True
    
    @_apply_to_composite_inv
    def transform_input_spec(self, action_spec):
        # Passthrough if already flat (or not the right key)
        if not isinstance(action_spec.space, BoxList): return action_spec

        # Get dims
        if not self._initialized: self._set_dims()
        n = 1
        for dim in self._dims: n *= dim

        # Set dims
        shape = list(action_spec.shape)
        shape[-1] = n
        action_spec = OneHot(n, shape=torch.Size(shape), device=action_spec.device, dtype=torch.int64)
        
        return action_spec