# Repository Structure for FacadeAI Inference Component

This document provides a detailed overview of the repository structure and the purpose of each file and directory in the FacadeAI Inference project.

## Overview

The repository is organized to support both local development and Azure ML deployment, with clear separation of code, configuration, data, and models.

## Directory Structure

```
/
├── Dockerfile            # Docker configuration for containerization
├── environment.yml       # Conda environment definition
├── component.yml         # Azure ML component definition
├── component.py          # Main inference component code
├── utils.py              # Utility functions for model loading and inference
├── runlocal.sh           # Script to run the component locally
├── good.py               # Example code for implementing ML model loading/prediction
├── data/                 # Input/output data directory
│   ├── images/           # Input images directory
│   │   └── B3/           # Batch directory
│   │       └── cam/      # Camera images
│   └── output/           # Output results directory
├── models/               # MLflow model directories
│   └── Glazing-Defects/  # Example model directory
└── .github/              # GitHub-specific files
    └── copilot-instructions/  # Instructions for GitHub Copilot
```

## Core Files

### Dockerfile

The Dockerfile creates a container environment with all necessary dependencies, based on NVIDIA CUDA image for GPU support.

**Key components:**
- Base image: NVIDIA CUDA 12.1.0 with cuDNN8 on Ubuntu 20.04
- Python environment: Miniconda with Python 3.9
- Dependencies: Installs packages from conda.yaml
- Working directory: `/app`

### environment.yml

This file defines the Conda environment for local development and testing. 

**Key components:**
- Python version specification
- Core dependencies for ML inference
- Platform-specific packages

### component.yml

This is the Azure ML component definition file that describes how the component should be run in Azure ML pipelines.

**Key components:**
- Component name and version
- Input/output definitions
- Environment configuration
- Command to execute

### component.py

The main Python script that implements the inference logic.

**Key components:**
- Argument parsing for configuration
- Model loading from MLflow format
- Sliding window implementation for processing large images
- Detection using loaded models
- Result formatting and storage

### utils.py

Contains utility functions used by component.py.

**Key components:**
- Image loading and preprocessing functions
- Model loading and prediction helpers
- Sliding window implementation
- Coordinate transformation functions
- CosmosDB integration functions

### runlocal.sh

Bash script to run the component locally with default parameters.

**Key components:**
- Default parameter definitions
- Directory creation for outputs
- Component execution command

### good.py

An example implementation showing how to properly load and use ML models.

**Key components:**
- Model loading patterns
- Image preprocessing
- Prediction function implementation
- Result formatting

## Data and Models

### data/images/

Directory for input images, organized by batch and camera.

**Structure:**
- `B3/`: Batch directory
  - `cam/`: Camera images
    - `*.jpg`: Individual images

### data/output/

Directory for output results from the inference process.

**Contents:**
- JSON files with detection results
- Trace images if enabled
- results.json summary file

### models/

Directory containing MLflow model directories.

**Structure per model:**
- `MLmodel`: MLflow model definition file
- `conda.yaml`: Environment specification
- `python_env.yaml`: Python environment requirements
- `python_model.pkl`: Pickled Python model
- `artifacts/`: Model artifacts directory

## GitHub Configuration

### .github/copilot-instructions/

This directory contains detailed instructions for GitHub Copilot users to understand and work with the codebase.

**Files:**
- `README.md`: Main instructions overview
- `docker-setup.md`: Docker setup guide
- `conda-setup.md`: Conda environment setup guide
- `azure-ml-integration.md`: Azure ML integration guide
- `repository-structure.md`: This file
- `troubleshooting.md`: Common issues and solutions
- `running-component.md`: Instructions for running the component

## Code Structure and Flow

### Main Execution Flow

1. `runlocal.sh` is executed, which sets parameters and runs `component.py`
2. `component.py` parses arguments and initializes the environment
3. Models are loaded from the specified directories using MLflow
4. Input images are read and processed using the sliding window technique
5. Predictions are made for each window and results are combined
6. Results are saved locally and/or to CosmosDB

### Key Classes and Functions

#### In component.py:

- `parse_arguments()`: Parses command line arguments
- `load_models()`: Loads ML models using MLflow
- `process_image()`: Processes a single image using sliding window
- `run()`: Main function for parallel processing in Azure ML
- `run_parallel_inference()`: Orchestrates processing of multiple images

#### In utils.py:

- `sliding_window()`: Splits an image into overlapping windows
- `run_generic_model_predict()`: Standardizes model prediction interface
- `load_image_as_array()`: Loads and prepares images for processing
- `recalculate_coordinates()`: Adjusts coordinates based on window position
- `update_cosmos_db()`: Updates results in CosmosDB
- `save_local_results()`: Saves results to local filesystem

## Azure ML Integration

The repository is designed to work seamlessly with Azure ML through:

1. Component definition in component.yml
2. Parallel processing support in component.py
3. CosmosDB integration for result storage
4. Model versioning and tracking with MLflow

For more details on Azure ML integration, see [azure-ml-integration.md](azure-ml-integration.md).