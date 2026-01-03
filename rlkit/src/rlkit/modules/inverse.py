from __future__ import annotations

import contextlib
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, TensorDictParams
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.utils import _reduce

from torchrl.modules.tensordict_module.probabilistic import SafeProbabilisticModule, SafeProbabilisticTensorDictSequential
from torchrl.modules.tensordict_module.sequence import SafeSequential

class SiameseWrapper(nn.Module):
    def __init__(self, trunk: nn.Module, num_inputs: int):
        super().__init__()
        self.trunk = trunk
        self.num_inputs = num_inputs

    def forward(self, *args):
        # 1. Stack/Cat inputs: (B, ...) -> (2B, ...)
        assert len(args) == 2 * self.num_inputs, \
            f"Expected {2 * self.num_inputs} inputs, got {len(args)}"
        
        current_inputs = args[:self.num_inputs]
        next_inputs = args[self.num_inputs:]

        combined_inputs = []
        for curr, next_ in zip(current_inputs, next_inputs):
            combined_inputs.append(torch.cat([curr, next_], dim=0))
        
        # 2. Combined Forward pass
        combined_features = self.trunk(*combined_inputs)
        
        # 3. Split outputs: (2B, ...) -> (B, ...), (B, ...)
        features, next_features = torch.chunk(combined_features, 2, dim=0)
        return features, next_features

class ActorValueInverseOperator(SafeSequential):
    def __init__(
        self,
        trunk_net: nn.Module,
        policy_operator: TensorDictModule,
        value_operator: TensorDictModule,
        inverse_operator: TensorDictModule,
        observation_keys=["observation"],
        hidden_keys=["hidden_features"],
    ):
        # Only do state prediction
        trunk_no_inverse = TensorDictModule(
            trunk_net,
            in_keys=observation_keys,
            out_keys=hidden_keys,
        )
        # Do state and next state prediction
        common_operator = TensorDictModule(
            SiameseWrapper(trunk_net, num_inputs=len(observation_keys)),
            in_keys=observation_keys + [("next", key) for key in observation_keys],
            out_keys=hidden_keys + [("next", key) for key in hidden_keys],
        )

        super().__init__(
            common_operator,
            policy_operator,
            value_operator,
            inverse_operator,
        )

        self.trunk_no_inverse = trunk_no_inverse
        self.common_operator = common_operator

    def get_policy_operator(self) -> SafeSequential:
        """Returns a standalone policy operator that maps an observation to an action."""
        if isinstance(self.module[1], SafeProbabilisticTensorDictSequential):
            return SafeProbabilisticTensorDictSequential(
                self.trunk_no_inverse, *self.module[1].module
            )
        return SafeSequential(self.trunk_no_inverse, self.module[1])

    def get_value_operator(self) -> SafeSequential:
        """Returns a standalone value network operator that maps an observation to a value estimate."""
        return SafeSequential(self.trunk_no_inverse, self.module[2])
    
    def get_inverse_operator(self) -> SafeSequential:
        """Returns a standalone inverse dynamics operator that maps (observation, next_observation) to action."""
        return SafeSequential(self.module[0], self.module[3])

    def get_policy_head(self) -> SafeSequential:
        """Returns the policy head."""
        return self.module[1]

    def get_value_head(self) -> SafeSequential:
        """Returns the value head."""
        return self.module[2]
    
    def get_inverse_head(self) -> SafeSequential:
        """Returns the inverse dynamics head."""
        return self.module[3]

class ClipPPOInverseLoss(ClipPPOLoss):
    """
    Clipped PPO Loss with an auxiliary Inverse Dynamics Loss.
    
    This loss adds an auxiliary term: L_total = L_PPO + inverse_coeff * L_Inverse
    Where L_Inverse predicts the action taken between observation and next_observation.
    """

    @dataclass
    class _AcceptedKeys(ClipPPOLoss._AcceptedKeys):
        predict_action: str = "predict_action" 

    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams | None
    critic_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams | None
    target_critic_network_params: TensorDictParams | None

    inverse_dynamics_network: TensorDictModule
    inverse_dynamics_network_params: TensorDictParams | None
    target_inverse_dynamics_network_params: TensorDictParams | None
        
    tensor_keys: _AcceptedKeys

    def __init__(
        self,
        actor_network=None,
        critic_network=None,
        inverse_dynamics_network: TensorDictModule = None,
        *,
        inverse_dynamics_coeff: float = 1.0,
        inverse_loss_fn: str = "cross_entropy",
        **kwargs,
    ):
        super().__init__(actor_network, critic_network, **kwargs)
        
        if inverse_dynamics_network is None:
            raise ValueError("inverse_dynamics_network must be provided")

        self.inverse_dynamics_coeff = inverse_dynamics_coeff
        
        if self.functional:
            self.convert_to_functional(
                inverse_dynamics_network, 
                "inverse_dynamics_network"
            )
        else:
            self.inverse_dynamics_network = inverse_dynamics_network
            self.inverse_dynamics_network_params = None

        if inverse_loss_fn == "cross_entropy":
            self.inverse_loss_fn = nn.CrossEntropyLoss()
        elif inverse_loss_fn == "mse":
            self.inverse_loss_fn = nn.MSELoss()
        else:
            self.inverse_loss_fn = inverse_loss_fn

    def _set_in_keys(self):
        super()._set_in_keys()
        keys = self._in_keys
        if self.inverse_dynamics_network is not None:
            keys += list(self.inverse_dynamics_network.in_keys)
        self._in_keys = list(set(keys))

    @property
    def out_keys(self):
        if self._out_keys is None:
            super().out_keys
            self._out_keys.append("loss_inverse_dynamics")
        return self._out_keys

    def loss_inverse_dynamics(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Computes the inverse dynamics loss."""
        
        with self.inverse_dynamics_network_params.to_module(
            self.inverse_dynamics_network
        ) if self.functional else contextlib.nullcontext():
            preds_td = self.inverse_dynamics_network(tensordict)
        
        preds = preds_td.get(self.tensor_keys.predict_action)
        true_action = tensordict.get(self.tensor_keys.action).to(preds.dtype)
        
        loss = self.inverse_loss_fn(preds, true_action)
        return loss

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        loss_td = super().forward(tensordict)
        loss_inverse = self.loss_inverse_dynamics(tensordict)
        loss_td.set("loss_inverse_dynamics", loss_inverse * self.inverse_dynamics_coeff)
        
        loss_td.set(
            "loss_inverse_dynamics", 
            _reduce(loss_inverse, reduction=self.reduction)
        )
        
        return loss_td