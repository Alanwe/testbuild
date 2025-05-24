# Docker Setup for FacadeAI Inference Component

This document provides detailed instructions for setting up and running the FacadeAI Inference component using Docker.

## Introduction

Docker provides an isolated, reproducible environment for running the FacadeAI inference component. It ensures that all dependencies are consistent and properly configured regardless of the host system.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine
- At least 8GB of RAM allocated to Docker
- At least 10GB of free disk space

## Setup Instructions

### Step 1: Clone the Repository

If you haven't already done so, clone the repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Understanding the Dockerfile

The included Dockerfile is based on NVIDIA CUDA 12.1.0 with cuDNN8, optimized for deep learning inference. Key components include:

1. Base image with CUDA support for GPU acceleration
2. Installation of essential system libraries
3. Miniconda setup for Python environment management
4. Creation of a conda environment with required dependencies
5. Configuration of the working directory

### Step 3: Build the Docker Image

Build the Docker image with the following command:

```bash
docker build -t facadeai:latest .
```

This process may take several minutes as it downloads base images and installs dependencies.

### Step 4: Prepare Data and Models

Ensure you have:
- Input images in the `data/images` directory
- Model files in the `models` directory

The directory structure should look like:

```
data/
  images/
    B3/             # Batch directory
      cam/          # Camera images directory
        image1.jpg
        image2.jpg
models/
  Glazing-Defects/  # Model directory
    MLmodel
    conda.yaml
    python_env.yaml
    python_model.pkl
    artifacts/
    ...
```

### Step 5: Run the Docker Container

Run the Docker container with volumes mounted for data and models:

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  facadeai:latest
```

#### Key Docker Run Arguments

- `-it`: Run in interactive mode with a terminal
- `--rm`: Remove the container when it exits
- `-v $(pwd)/data:/app/data`: Mount the local data directory into the container
- `-v $(pwd)/models:/app/models`: Mount the local models directory into the container

### Step 6: Run the Inference Component

Inside the container, run the inference script:

```bash
./runlocal.sh
```

This will process all images in the specified input directory (default: `data/images`) and save results to the output directory (default: `data/output`).

## Customizing the Docker Environment

### Modifying the Conda Environment

If you need to add additional dependencies, you can edit the `environment.yml` file before building the Docker image.

### Using GPU Acceleration

To use GPU acceleration, you need:
1. NVIDIA drivers installed on your host machine
2. NVIDIA Container Toolkit installed
3. Add `--gpus all` to your docker run command:

```bash
docker run -it --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  facadeai:latest
```

### Docker Compose (Optional)

For convenience, you can create a `docker-compose.yml` file:

```yaml
version: "3.8"
services:
  facadeai:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    tty: true
    command: bash
```

And then run:

```bash
docker-compose up -d
docker-compose exec facadeai ./runlocal.sh
```

## Troubleshooting Docker Issues

### Container Exits Immediately

If the container exits immediately, check:
- Docker logs for error messages: `docker logs <container-id>`
- Ensure your data and model directories are accessible

### Memory Issues

If you encounter memory errors:
- Increase Docker's memory allocation in Docker Desktop settings
- Try using a smaller batch size or reducing the number of concurrent processes

### Permission Issues

If you encounter permission issues accessing mounted volumes:
- Check the permissions on your local data and models directories
- Try adding the `--user $(id -u):$(id -g)` flag to your docker run command

### Unable to Connect to NVIDIA GPU

If GPU acceleration isn't working:
- Verify that the NVIDIA Container Toolkit is properly installed
- Check that your GPU drivers are compatible with CUDA 12.1.0
- Run `nvidia-smi` on the host to verify GPU availability