# CUDA
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Magic "best practices" stuff
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Copy pasted bare minimum installs
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget curl ca-certificates bzip2 git \
      build-essential pkg-config \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Mamba (mini version of conda)
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
 && bash Miniforge3-Linux-x86_64.sh -b -p $CONDA_DIR \
 && rm -f Miniforge3-Linux-x86_64.sh
ENV PATH=$CONDA_DIR/bin:$PATH


# Bash shell
SHELL ["/bin/bash", "-lc"]

# Create conda env
RUN mamba create -y -n mlagents python=3.10.12 \
 && conda clean -afy
ENV CONDA_DEFAULT_ENV=mlagents
ENV PATH=$CONDA_DIR/envs/mlagents/bin:$PATH

### Install Requirements
# 1. pip requirements layer
RUN python -m pip install --no-cache-dir "grpcio==1.48.2" \
 && python -m pip install --no-cache-dir "mlagents==1.1.0" \
 && python -m pip install --no-cache-dir "numpy==2.2.6" \
 && python -m pip install --no-cache-dir pandas matplotlib ipykernel \
 && python -m pip install --no-cache-dir torch torchrl

# 2, rlkit layer
WORKDIR /app
COPY rlkit ./rlkit
RUN python -m pip install --no-cache-dir -e rlkit

# 3. Project layer
COPY . .

CMD ["/bin/bash"]
