### Imports
from ray.rllib.algorithms.ppo import PPOConfig
from config import config

# Training Loop
ppo = config.build_algo()
for _ in range(10):
    print(ppo.train())
    
ppo.save("soccer_twos_shared_policy")