# FacadeAI Development Environment Setup

This document provides instructions for setting up and using the FacadeAI development environment for CPU-based inference.

## Overview

The FacadeAI development environment includes:

1. A conda environment with all required dependencies for CPU-based inference
2. A dummy model for testing
3. Scripts to set up the environment and run inference

## Setup Options

### Option 1: Docker Setup (Easiest)

For a completely isolated and ready-to-use development environment:

1. Make sure Docker and Docker Compose are installed on your system

2. Run the Docker development environment:

```bash
chmod +x run_docker_dev.sh
./run_docker_dev.sh
```

3. Access the container:

```bash
docker exec -it testbuild_facadeai-dev_1 bash
```

4. Inside the container, run the inference:

```bash
./dev_runlocal.sh
```

5. When finished, stop the container:

```bash
docker-compose -f docker-compose.dev.yml down
```

### Option 2: Conda Setup (Recommended for local development)

1. Run the setup script:

```bash
chmod +x setup_dev.sh
./setup_dev.sh
```

2. Activate the conda environment:

```bash
conda activate facadeai_dev
```

3. Run the inference script:

```bash
chmod +x dev_runlocal.sh
./dev_runlocal.sh
```

### Option 3: Python venv Setup

1. Create the Python virtual environment:

```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

2. Activate the virtual environment:

```bash
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Run the inference script:

```bash
./dev_runlocal.sh
```

### Option 4: Manual Setup

1. Create the conda environment:

```bash
conda env create -f dev_environment.yml
conda activate facadeai_dev
```

2. Install required packages using pip:

```bash
pip install -r requirements.txt
```

3. Create necessary directories:

```bash
mkdir -p data/images/B3/cam
mkdir -p data/output
mkdir -p models/Dev-Model
```

4. Create a dummy model using the script in `src/old/create_dummy_model.py`:

```bash
python src/old/create_dummy_model.py models/Dev-Model
```

5. Run the inference script:

```bash
chmod +x dev_runlocal.sh
./dev_runlocal.sh
```

### Option 3: Setup without Conda

If you don't want to use conda, you can set up a virtual environment with pip:

```bash
./dev_runlocal.sh
```

## Directory Structure

The development environment follows this structure:

```
/
├── component.py           # Main inference component code
├── dev_runlocal.sh        # Development version of the run script
├── dev_environment.yml    # Conda environment for development
├── requirements.txt       # Pip requirements for non-conda setup
├── setup_dev.sh           # Setup script for development environment
├── setup_venv.sh          # Setup script for venv environment
├── verify_env.py          # Environment verification script
├── Dockerfile.dev         # Dockerfile for development environment
├── docker-compose.dev.yml # Docker Compose for development
├── run_docker_dev.sh      # Script to run Docker development environment
├── data/                  # Input/output data directory
│   ├── images/            # Input images
│   └── output/            # Output results
└── models/                # MLflow model directories
    └── Dev-Model/         # Dummy model for development
├── requirements.txt       # Pip requirements for non-conda setup
├── setup_dev.sh           # Setup script for development environment
├── data/                  # Input/output data directory
│   ├── images/            # Input images
│   └── output/            # Output results
├── src/                   # Source code directory
│   └── utils.py           # Utility functions
└── models/                # MLflow model directories
    └── Dev-Model/         # Dummy model for development
```

## Testing the Environment

After setting up the environment:

1. Ensure the `data/images/B3/cam` directory contains at least one image file
   - The setup script creates a sample image if none exists
   - You can add your own test images to this directory

2. Run the inference script:

```bash
./dev_runlocal.sh
```

3. Check the output in `data/output`

## Troubleshooting

### Common Issues

1. **ImportError for dependencies**:
   - Ensure you've activated the conda environment: `conda activate facadeai_dev`
   - Try reinstalling the specific package: `pip install <package-name>`

2. **Model loading errors**:
   - Check that the Dev-Model structure is correct with MLmodel file, conda.yaml, and python_model.pkl
   - You can recreate the dummy model using: `python src/old/create_dummy_model.py models/Dev-Model`

3. **File not found errors**:
   - Ensure all required directories exist: `data/images/B3/cam` and `data/output`
   - Check that the paths in `dev_runlocal.sh` match your directory structure

4. **Permission issues**:
   - Make scripts executable: `chmod +x setup_dev.sh dev_runlocal.sh`

5. **Python version conflicts**:
   - Ensure you're using Python 3.9 as specified in the environment

## Using a Different Model

To use a different model:

1. Update the MODEL1 parameter in `dev_runlocal.sh`:

```bash
MODEL1="models/Your-Model-Name"
```

2. Run the script as normal

For models requiring GPU, additional dependencies would be needed.

## For GitHub Copilot

This development environment is designed to be compatible with GitHub Copilot, allowing you to:

1. Run and test the inference code locally
2. Debug and fix dependency issues
3. Make code changes and test them immediately

When working with Copilot, refer to the `good.py` file for reference implementation of model loading and prediction.