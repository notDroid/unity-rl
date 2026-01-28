#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "--- Creating Conda Environment 'mlagents' ---"
conda create -n mlagents python=3.10.12 -y

echo ""
echo "--- Installing Grpcio (Conda Forge) ---"
conda install -n mlagents -c anaconda --override-channels grpcio=1.48.2 -y

echo ""
echo "--- Installing ML-Agents (Sequential install to bypass resolver) ---"
conda run -n mlagents python -m pip install mlagents==1.1.0

echo ""
echo "--- Overwriting Numpy ---"
conda run -n mlagents python -m pip install -U numpy

echo ""
echo "--- Installing Toolkit ---"
conda run -n mlagents python -m pip install pandas matplotlib ipykernel hydra-core seaborn huggingface_hub torchinfo

echo ""
echo "--- Installing PyTorch & TorchRL ---"
conda run -n mlagents python -m pip install torch torchrl

echo ""
echo "--- Installing RLKit (Editable) ---"
conda run -n mlagents python -m pip install -e rlkit

echo ""
echo "--- DONE! Activate with: conda activate mlagents ---"
