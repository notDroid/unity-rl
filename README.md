# Unity ML-Agents with TorchRL
This repo contains examples of solving reinforcement learning scenarios from unity [mlagents](https://github.com/Unity-Technologies/ml-agents) with TorchRL. 

<p align="center">
  <img src="assets/3DBallModel.gif" alt="Row 1 Col 1" width="32%" style="margin-right: 0px; margin-bottom: 0px;">
  <img src="assets/PushBlockModel.gif" alt="Row 1 Col 2" width="32%" style="margin-right: 0px; margin-bottom: 0px;">
  <img src="assets/WallJumpModel.gif" alt="Row 1 Col 3" width="32%" style="margin-bottom: 0px;">
  <br>
  <img src="assets/CrawlerModel.gif" alt="Row 2 Col 1" width="32%" style="margin-right: 0px;">
  <img src="assets/WormModel.gif" alt="Row 2 Col 2" width="32%" style="margin-right: 0px;">
  <img src="assets/WalkerModel.gif" alt="Row 2 Col 3" width="32%">
</p>

## **Quickstart**

**1. Installation**

Clone the repo and run the auto-install script. This handles the complex dependency conflicts (mlagents vs numpy) for you. *Requires Conda.*

```bash
git clone https://github.com/notDroid/unity-rl.git
cd unity-rl

bash install.sh

conda activate mlagents
```
**2. CLI Usage** 

You can list available models and run them immediately from the command line.

```bash
# List all available environments and models
python play.py ls

# Run a specific environment (auto-downloads model from HF)
python play.py Crawler ppo conf1 run9 --graphics
```

**3. Python Usage** 

Minimal example to load an agent and run a rollout:

```python
from utils import PPOAgent
from rlkit.envs import UnityEnv

# 1. Load Agent (Auto-downloads from Hugging Face)
agent = PPOAgent('Crawler', 'conf1', 'run9')
policy = agent.get_policy_operator()

# 2. Run Environment (Auto-downloads from mlagents registry)
env = UnityEnv(name='Crawler', graphics=True)
with torch.no_grad():
  env.rollout(1000, policy=policy, break_when_any_done=False)
```
Check `quickstart.ipynb` for a complete walkthrough.

## **Architecture & Usage**

### **Organization**

There are 2 main components: 
**1. rlkit**
  - rlkit contains algorithms (like ppo, sac), unity environments (with torchrl transforms), and other utility.

```python
env = UnityEnv(name='Crawler', path=None, graphics=True, time_scale=1, seed=1)
agent = PPOAgent('Crawler', 'conf1', 'run9')
```

**2. experiment runner**
  - I use hydra to manage configs for (environment, algorithm, config) tuples. 
  - Experiment results are under experiments/, configs under configs/, and the code for the experiment runner has its entry point at run_experiment.py. 
```bash
python run_experiment.py -cn "config_name" +verbose=True +continue_=False run_name="run_name"
```

Both have huggingface integration to upload/download models, checkpoints, logs automatically at https://huggingface.co/notnotDroid/unity-rl (default).

### **Unity Environments**
You can either use the built in unity environments or download them manually. The manual download ones look better and may be necessary if the unity registry is down.

**Manual Download**
-  [Download](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Installation.html) the repo containing the environments.

- Then open the project in the unity editor (select the Project/ folder from mlagents), select a scene from an environment and build it for whatever platform you're on.
- Create an env/ folder at the root of this repo and place compiled environments in it.

### **Python Environment**

Either run install.sh or manually install the dependencies.

**Manual Install**

First of all conda is required (something weird about grpcio, wheel won't build) so make sure its properly setup. Then run these at project root:

```Bash
# Create conda environment
conda create -n mlagents python=3.10.12
conda activate mlagents

# Install mlagents python interface
conda install "grpcio=1.48.2" -c conda-forge
python -m pip install mlagents==1.1.0
python -m pip install numpy==2.2.6

# Install toolkit
python -m pip install pandas matplotlib ipykernel hydra-core seaborn huggingface_hub torchinfo
python -m pip install torch torchrl 
python -m pip install -e rlkit
```
Note that the numpy version conflicts with mlagents because of gym (deprecated), but we don't use gym anyways so we are safe to use the latest version of numpy. This also means we have to manual download everything (no requirements.txt).

### **RLKit**
This package contains reusable resources:
1. mlagent environments (with torchrl transforms)
2. training templates (ppo/sac)
    - The training templates are meant to be used as templates rather than robust algorithms (customize them).
3. utils (checkpointer/logger)
4. models (mlp/cnn)


### **TODO**

1. Finish models for vector environments (3DBall, Crawler, PushBlock, Walker, WallJump, Worm).
2. Add support for visual environments (GridWorld, Match 3)
3. Add support for multi agent environments (food collector, soccer twos, striker vs. goalie, co-op pushblock, dungeon escape)
4. Add SAC
5. Add support for sparse reward environments (hallway, pryamids)
6. Add support for variable length observation environments (sorter)

- Also add Docker support

## **Environments**

### **Summary of Results**

My configs are likely not optimal, but they work reasonably well. Feel free to open an issue or PR if you have better hyperparameters or training tricks.

**Reproducibility**

Models and training logs (with plots) are available on [Hugging Face](https://huggingface.co/notnotDroid/unity-rl). Training runs can be reproduced with:

```bash
python run_experiment.py -cn <config_name> +verbose=True +continue_=False run_name=<run_name>
```

**Results**

The following table summarizes the average returns achieved by each (environment, algorithm, config) tuple.
- For environments with truncation the window is 1000 timesteps.


| Environment | Algorithm      | Config File    | Average Return | Episode Length | Timesteps Trained |
|-------------|----------------|----------------|----------------| ----------------|-------------------|
| 3DBall      | PPO            | 3dball_ppo     | 100            | 1000            | 400k              |
| PushBlock   | PPO            | pushblock_ppo  | 4.9            | 48.2            | 50M               |
| WallJump    | PPO            | walljump_ppo   | 0.96           | 29.7            | 500M              |
| Crawler     | PPO            | crawler_ppo    | 360            | 1000            | 400M              |
| Worm        | PPO            | worm_ppo       | 100            | 1000            | 100M              |
| Walker      | PPO            | walker_ppo     | 25             | 1000            | 1.6B              |

