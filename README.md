# Unity RL Solutions
This repo contains examples of solving reinforcement learning scenarios from unity with torchRL. 
It contains environments from the unity [mlagents](https://github.com/Unity-Technologies/ml-agents) repo.

The projects in this repo are under experiments/ and organized as: [environment]/[experiment]/. 
For example "experiments/3DBall/ppo".

## **Usage**

### **Unity Environments**:
You can either use the built in unity environments or download them manually. I reccomend using the built in ones at first, but the manual download ones look better.

**Manual Download**:
-  [Download](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Installation.html) the repo containing the environments:

- Then open the project in the unity editor (select the Project/ folder from mlagents), select a scene from an environment and build it for whatever platform you're on.
- Create an env/ folder at the root of this repo and place compiled environments in it.

### **Python Environment**:

First of all conda is required (something weird about grpcio, wheel won't build) so make sure its properly setup. Then run these at project root:

```
# Create conda environment
conda create -n mlagents python=3.10.12
conda activate mlagents

# Install mlagents python interface
conda install "grpcio=1.48.2" -c conda-forge
python -m pip install mlagents==1.1.0
python -m pip install numpy==2.2.6

# Install toolkit
python -m pip install pandas matplotlib ipykernel
python -m pip install torch torchrl 
python -m pip install -e rlkit
```

### **RLKit:**
This contains reusable resources. Like models (mlp/cnn), utils (checkpointer/logger), and training templates (ppo/sac).

If you're working on an environment, I reccomend copy and pasting the template in and customizing it from there. It is not made for general use, just as a starting point. 

## **TODO List:**
- SAC
    - Add GPU support (with amp)
    - Customize loss module: separate entropy and reward heads, separate discount factor for entropy reward.
- Add rest of the unity environments.
    - Visual and MARL environments.
- Set up docker for benchmark training runs.
- Learn about how to do distributed training.
