import supersuit as ss
from pettingzoo.sisl import waterworld_v4
from stable_baselines3 import PPO

# 1) Build a ParallelEnv
env = waterworld_v4.parallel_env()
# If agents can die/respawn, keep agent count fixed for vectorization:
# env = ss.black_death_v3(env)  # uncomment when needed

# 2) Convert to an SB3 VecEnv, and optionally concat for more workers
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=8, num_cpus=2, base_class="stable_baselines3")

# 3) Single shared policy controls all agents (parameter sharing)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_pz_shared.zip")

model = PPO.load("ppo_pz_shared.zip")
env = waterworld_v4.env(render_mode=None)  # AEC for eval

for ep in range(5):
    env.reset(seed=ep)
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        act = None if terminated or truncated else model.predict(obs, deterministic=True)[0]
        env.step(act)
env.close()