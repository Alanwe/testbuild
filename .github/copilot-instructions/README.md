# FacadeAI Inference Environment Setup Guide

This guide provides comprehensive instructions for setting up and running the FacadeAI Inference component locally and on Azure ML pipelines.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Local Setup with Docker](#local-setup-with-docker)
4. [Local Setup without Docker](#local-setup-without-docker)
5. [Running the Component](#running-the-component)
6. [Azure ML Integration](#azure-ml-integration)
7. [Troubleshooting](#troubleshooting)

## Project Overview

FacadeAI is a computer vision system designed to detect defects in building facades. The system uses a sliding window technique to process large images with ML models to identify various types of defects. Key features include:

- Support for multiple ML models specialized for different defect types
- Local processing mode with result storage to disk
- Azure integration with CosmosDB for result persistence
- Parallel processing capability for high throughput
- Sliding window technique for handling large-resolution images

## Repository Structure

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
│   └── output/           # Output results directory
└── models/               # MLflow model directories
    └── Glazing-Defects/  # Example model directory
```

See [repository-structure.md](repository-structure.md) for more details on the organization and purpose of each file.

## Local Setup with Docker

Docker provides the most reliable way to run the component with consistent dependencies.

### Prerequisites

- Docker installed on your machine
- Git (to clone the repository)

### Setup Steps

1. Clone the repository (if not already done):

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Build the Docker image:

```bash
docker build -t facadeai:latest .
```

3. Run the Docker container with mounted volumes for data and models:

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  facadeai:latest
```

4. Inside the container, run the inference script:

```bash
./runlocal.sh
```

See [docker-setup.md](docker-setup.md) for more detailed Docker setup instructions.

## Local Setup without Docker

If you prefer to run the component directly on your host machine, you can set up a Conda environment.

### Prerequisites

- Anaconda or Miniconda installed
- Python 3.8-3.11 (the exact version depends on the specific azureml-automl-dnn-vision requirement)

### Setup Steps

1. Create a Conda environment using the provided environment.yml:

```bash
conda env create -f environment.yml
conda activate facade_ai_environment
```

2. Install additional required packages:

```bash
pip install azure-cosmos azure-identity azure-keyvault-secrets azureml-automl-dnn-vision==1.60.0 opencv-python-headless==4.8.1.78
```

3. Make the runlocal.sh script executable:

```bash
chmod +x runlocal.sh
```

4. Run the inference script:

```bash
./runlocal.sh
```

See [conda-setup.md](conda-setup.md) for more detailed setup instructions without Docker.

## Running the Component

The `runlocal.sh` script simplifies running the component by setting up default parameters.

### Basic Execution

```bash
./runlocal.sh
```

### Key Parameters

| Parameter    | Default Value       | Description                                |
|--------------|---------------------|--------------------------------------------|
| INPUT_DS     | data/images         | Location of input images                   |
| OUTPUT_DIR   | data/output         | Directory for output results               |
| BATCH_ID     | B3                  | Batch identifier                           |
| MODEL1       | models/Glazing-Defects | Path to the primary model               |
| LOCAL        | true                | Flag to run in local mode                  |
| TRACE        | true                | Enable visualization and JSON trace output |

### Customizing Parameters

You can edit the runlocal.sh script to modify any parameters before running:

```bash
# Example with modified parameters
INPUT_DS="data/my_images"
OUTPUT_DIR="data/my_results"
BATCH_ID="B5"
MODEL1="models/My-Custom-Model"
```

See [running-component.md](running-component.md) for more details on parameter configuration and output interpretation.

## Azure ML Integration

For production deployment, the component is designed to run on Azure ML.

### AzureML Setup

1. Register the component with Azure ML:

```python
from azure.ai.ml import load_component, MLClient
from azure.identity import DefaultAzureCredential

# Connect to Azure ML workspace
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential=credential)

# Load and register the component
component = load_component(path="./component.yml")
ml_client.components.create_or_update(component)
```

2. Use the component in an Azure ML pipeline:

```python
from azure.ai.ml import dsl, Input, Output

@dsl.pipeline()
def facade_pipeline():
    inference_job = component(
        input_ds=Input(type="uri_folder", path="azureml://datastores/datastore/paths/images"),
        model1=Input(type="mlflow_model", path="azureml://registries/models/Glazing-Defects/versions/1"),
        output_data=Output(type="uri_folder"),
        batch_id="B1",
        local=False
    )
    return {"output": inference_job.outputs.output_data}
```

See [azure-ml-integration.md](azure-ml-integration.md) for detailed instructions on setting up Azure ML pipelines.

## Troubleshooting

### Common Issues

1. **Missing dependencies**

   If you encounter errors about missing Python modules, ensure all dependencies in environment.yml are installed:
   
   ```bash
   pip install -r models/Glazing-Defects/requirements.txt
   ```

2. **Model loading errors**

   The error "Failed to load model: No module named 'azureml'" indicates that the Azure ML SDK is not installed:
   
   ```bash
   pip install azureml-automl-dnn-vision==1.60.0
   ```

3. **Python version mismatches**

   The Azure ML packages require specific Python versions. If you see compatibility errors, check that your Python version is compatible (usually 3.8-3.11 for azureml-automl-dnn-vision 1.60.0).

4. **CosmosDB connectivity issues**

   When running in cloud mode, ensure proper authentication for CosmosDB is set up. For local testing, you can keep local=true to skip CosmosDB operations.

See [troubleshooting.md](troubleshooting.md) for more detailed troubleshooting guidance.