# FacadeAI Inference Environment Setup Instructions

This document provides instructions for setting up and running the FacadeAI Inference component locally and on Azure ML pipelines.

## Project Overview

FacadeAI is a computer vision system for detecting defects in facades using machine learning models. The system:

1. Uses sliding window technique to process large images
2. Supports multiple ML models for different types of defect detection
3. Stores results in Azure CosmosDB and local storage
4. Can run in both local and cloud (Azure ML) environments

## Environment Setup

### Prerequisites

- Docker
- Python 3.9+
- Access to Azure ML (for cloud deployment)
- Required model files in the `/models` directory

### Setting Up Local Environment with Docker

1. Build the Docker image using the provided Dockerfile:

```bash
docker build -t facadeai:latest .
```

2. Run the container with mounted volumes for data and models:

```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  facadeai:latest
```

3. Inside the container, run the inference script:

```bash
./runlocal.sh
```

### Setting Up Local Environment without Docker

1. Create a Conda environment using the provided environment.yml:

```bash
conda env create -f environment.yml
conda activate facade_ai_environment
```

2. Install additional required packages:

```bash
pip install opencv-python-headless==4.8.1.78 azure-cosmos azure-identity azure-keyvault-secrets
```

3. Make sure the `runlocal.sh` script is executable:

```bash
chmod +x runlocal.sh
```

4. Run the inference script:

```bash
./runlocal.sh
```

## Directory Structure

```
/
├── Dockerfile            # Docker configuration 
├── environment.yml       # Conda environment definition
├── component.yml         # Azure ML component definition
├── component.py          # Main inference component code
├── utils.py              # Utility functions
├── runlocal.sh           # Script to run the component locally
├── data/                 # Input/output data directory
│   ├── images/           # Input images
│   └── output/           # Output results
└── models/               # MLflow model directories
    └── Glazing-Defects/  # Example model
```

## Running the Component

### Local Execution

The `runlocal.sh` script sets up default parameters and runs the component locally:

```bash
./runlocal.sh
```

Key parameters in runlocal.sh:
- `INPUT_DS`: Location of input images (default: "data/images")
- `OUTPUT_DIR`: Directory for output results (default: "data/output")
- `BATCH_ID`: Batch identifier (default: "B3")
- `MODEL1`: Path to the primary model (default: "models/Glazing-Defects")
- `LOCAL`: Flag to run in local mode (default: true)

You can modify these parameters in the script as needed.

### Azure ML Execution

For Azure ML deployment, the component.yml file defines the component for use in Azure ML pipelines.

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

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure all dependencies in the environment.yml file are installed.

2. **Model loading errors**: Verify that the model structure is compatible with MLflow loading mechanism.

3. **CosmosDB connectivity**: When running in cloud mode, ensure proper authentication for CosmosDB.

4. **Image loading failures**: Check that input images are in supported formats (JPG, PNG).

### Using the provided good.py

The `good.py` file provides an example implementation of model loading and prediction that works with the MLflow models. You can reference this file for proper implementation patterns.

## Azure ML Pipeline Integration

The component is designed to be integrated into Azure ML pipelines with these key features:

1. Parallel processing of images
2. Sliding window technique for large images
3. Seamless integration with Azure CosmosDB for results storage
4. Support for multiple models with different confidence thresholds

For Azure ML deployment, follow the Azure ML Pipeline documentation and use the component.yml file to define the component.