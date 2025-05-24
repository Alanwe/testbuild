# Conda Environment Setup for FacadeAI Inference Component

This document provides detailed instructions for setting up and running the FacadeAI Inference component directly on your host machine using a Conda environment.

## Introduction

While Docker provides the most consistent environment, you may prefer to run the component directly on your host machine. This guide helps you set up a suitable Conda environment with all required dependencies.

## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system
- Python 3.8, 3.9, or 3.10 (Required for compatibility with Azure ML packages)
- At least 4GB of RAM
- 5GB of free disk space

## Setup Instructions

### Step 1: Clone the Repository

If you haven't already done so, clone the repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Create the Conda Environment

The repository includes a `environment.yml` file that defines the necessary dependencies. Create a conda environment from this file:

```bash
conda env create -f environment.yml
```

This will create a new environment named `project_environment` (or whatever name is specified in the environment.yml file).

### Step 3: Activate the Environment

Activate the newly created environment:

```bash
conda activate project_environment
```

### Step 4: Install Additional Packages

Some packages may need to be installed separately because they are not available in the conda repositories or have specific version requirements:

```bash
pip install azure-cosmos azure-identity azure-keyvault-secrets
pip install opencv-python-headless==4.8.1.78
pip install azureml-automl-dnn-vision==1.60.0
```

### Step 5: Install PyTorch (if not included in environment.yml)

If your specific machine has GPU support and you want to use it:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio
```

### Step 6: Verify Environment

Verify that all key packages are installed:

```bash
pip list | grep -E 'opencv|numpy|azure|mlflow|torch'
```

You should see output listing these packages and their versions.

### Step 7: Prepare Data and Models

Ensure you have:
- Input images in the `data/images` directory
- Model files in the `models` directory

The expected directory structure is:

```
data/
  images/
    B3/             # Batch directory
      cam/          # Camera images directory
        image1.jpg
        image2.jpg
        ...
models/
  Glazing-Defects/  # Model directory
    MLmodel
    conda.yaml
    python_env.yaml
    python_model.pkl
    ...
```

If you don't have the model files, you may need to download them separately or create a dummy model for testing:

```bash
# Create directories if they don't exist
mkdir -p data/images/B3/cam data/output models/Glazing-Defects
```

### Step 8: Make the Run Script Executable

Make sure the run script has execute permissions:

```bash
chmod +x runlocal.sh
```

### Step 9: Run the Component

Run the inference component using the provided script:

```bash
./runlocal.sh
```

The script will process all images in the input directory and save results to the output directory.

## Common Issues with Conda Environment

### Python Version Conflicts

Azure ML packages have specific Python version requirements. If you encounter errors, check that your Python version is compatible:

```bash
python --version
```

If needed, create a new environment with a specific Python version:

```bash
conda create -n facadeai python=3.9
conda activate facadeai
# Then install required packages
```

### Package Conflicts

If you encounter package conflicts, try installing packages in this order:

1. Core Python packages first
2. PyTorch and related packages
3. Azure ML packages
4. OpenCV and other dependencies

```bash
pip install numpy pandas pillow pyyaml
pip install torch torchvision
pip install azureml-automl-dnn-vision==1.60.0
pip install opencv-python-headless==4.8.1.78
```

### GPU Support Issues

If you're trying to use GPU acceleration:

1. Ensure you have compatible NVIDIA drivers installed
2. Check CUDA compatibility with PyTorch version
3. Verify GPU is detected:

```bash
# For PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### Environment Variable Setup (if needed)

Some components may require environment variables to be set:

```bash
# For Azure authentication
export AZURE_CLIENT_ID="your-client-id"
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_SECRET="your-client-secret"

# For model directory (if needed)
export AZUREML_MODEL_DIR="./models"
```

## Alternative: Use Python venv

If you prefer using Python's built-in virtual environment system instead of Conda:

```bash
# Create a virtual environment
python -m venv facadeai_env

# Activate it (Linux/macOS)
source facadeai_env/bin/activate

# Activate it (Windows)
facadeai_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Then install additional packages as described above
```