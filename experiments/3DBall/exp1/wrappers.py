from pettingzoo.utils import BaseParallelWrapper
import gymnasium as gym
import functools
import numpy as np

def flatten_mask(masks):
    '''
    Flattens multi-dimensions mask.
    - Masks is a tuple containing np.array of dtype some int

    Ex: [1, 0] x [1, 1] = [1, 1, 0, 0]
    This corresponds to the cross product with & of (first input is row, second is column):
    * 1 1
    1 1 1
    0 0 0
    Then reduce the 2D array [[1, 1], [0, 0]] to [1, 1, 0, 0].
    For more than 2 masks, use procedure recursively.
    '''
    assert isinstance(masks, tuple), "Mask array must be a tuple"
    flat = masks[0]
    for mask in masks[1:]:
        # Flat is along the row dim, the next mask is along the column. & 2D Product -> Flatten
        flat = (flat[:, None] & mask[None, :]).reshape(-1)
    return flat


class ConcatParallelEnv(BaseParallelWrapper):
    def __init__(self, env):
        """
        - Concats observation space. (Tuple of boxes, else passthrough)
        - Flattens multi dim discrete action space. (only works with MultiDiscrete, else passthrough)
        """
        super().__init__(env)

    def reset(self, seed=None, options=None):
        # Flatten output obs
        obs, info = super().reset(seed, options)
        return self._flatten_obs(obs), info
    
    def step(self, actions):
        # Unflatten input actions, flatten output obs
        obs, rew, term, trunc, info = super().step(self._unflatten_actions(actions))
        return self._flatten_obs(obs), rew, term, trunc, info

    def _flatten_obs(self, obs):
        for agent, agent_obs in obs.items():
            # concat obs
            observation_space = super().observation_space(agent)
            if isinstance(observation_space, gym.spaces.Tuple) and "observation" in agent_obs:
                agent_obs["observation"] = np.concatenate(agent_obs["observation"])

            # flatten action_mask
            action_space = super().action_space(agent)
            if isinstance(action_space, gym.spaces.MultiDiscrete) and "action_mask" in agent_obs:
                agent_obs["action_mask"] = flatten_mask(agent_obs["action_mask"])
        return obs
    
    def _unflatten_actions(self, actions):
        unflat_actions = {}
        for agent, agent_action in actions.items():
            # Single index -> Multi dim index
            action_space = super().action_space(agent)
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                unflat_actions[agent] = np.array(np.unravel_index(agent_action, action_space.nvec))
            else:
                unflat_actions[agent] = agent_action
        return unflat_actions
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        observation_space = super().observation_space(agent)

        # Flatten (else passthrough)
        if isinstance(observation_space, gym.spaces.Tuple):
            observation_space = gym.spaces.utils.flatten_space(observation_space)
        
        # Turn into a dict{observation, action_mask}
        observation_space = gym.spaces.Dict({
            "observation": observation_space,
            "action_mask": gym.spaces.MultiBinary(int(self.action_space(agent).n))
        })
        return observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        action_space = super().action_space(agent)
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            return gym.spaces.Discrete(np.prod(action_space.nvec))
        return action_space
    
    # OVERRIDE BaseParallelWrapper which prohibts accessing attributes starting with _
    def __getattr__(self, name):
        return getattr(self.env, name)
    

