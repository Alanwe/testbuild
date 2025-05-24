# 1) Base on NVIDIA CUDA 12.1.0 + cuDNN8 runtime (Ubuntu 20.04)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

# 2) Install prerequisites *and* build tools (gcc, make, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      bzip2 \
      ca-certificates \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libssl-dev \
      libffi-dev \
      zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 3) Install Miniconda (use the “latest” installer)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /tmp/conda.sh && \
    bash /tmp/conda.sh -b -p $CONDA_DIR && \
    rm /tmp/conda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# 4) Copy and create your exact conda env
WORKDIR /app
COPY conda.yaml /app/conda.yaml
RUN conda config --add channels conda-forge \
 && conda config --add channels anaconda \
 && conda env create -f conda.yaml \
 && conda clean -afy

# 5) Activate the env for all future commands
SHELL ["conda", "run", "-n", "project_environment", "/bin/bash", "-l", "-c"]

# 6) (Optional) Copy in your source
# COPY . /app

# 7) Default to an activated bash shell
CMD ["bash"]


