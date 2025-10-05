# Unity RL Solutions
This repo contains examples of solving reinforcement learning scenarios from unity with torchRL. 
It contains environments from the unity [mlagents](https://github.com/Unity-Technologies/ml-agents) repo.

The projects in this repo are under experiments/ and organized as: [environment]/[experiment]/. 
Each experiment folder is self contained (a full project on its own). 

## **Usage**

### **Unity Environments**:

In order to use this repo you need to [download](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Installation.html) the repo containing the environments:

Then open the project in the unity editor (select the Project/ folder from mlagents), select a scene from an environment and build it for whatever platform you're on.
Create an env/ folder at the root of this repo and place compiled environments in it.

### **Python Environment**:

First of all conda is required (something weird about grpcio) so make sure its properly setup. Then run these at project root:

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
python -m pip install -e rldk
```
