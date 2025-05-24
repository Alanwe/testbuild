# Azure ML Integration Guide for FacadeAI Inference

This document provides detailed instructions for integrating and running the FacadeAI Inference component in Azure Machine Learning.

## Introduction

Azure Machine Learning (Azure ML) provides a managed cloud service for training, deploying, and managing machine learning models at scale. The FacadeAI Inference component is designed to be deployed as an Azure ML component, allowing it to be run in Azure ML pipelines with scalable compute resources.

## Prerequisites

To use Azure ML with the FacadeAI Inference component, you'll need:

- An Azure subscription with access to Azure Machine Learning
- Azure ML workspace 
- Appropriate permissions to create and run pipelines
- Azure CLI with ML extension installed for some operations
- Python environment with Azure ML SDK v2 installed

## Setup Instructions

### Step 1: Install Azure ML SDK v2

Install the required Python packages for Azure ML integration:

```bash
pip install azure-ai-ml azure-identity
```

### Step 2: Authenticate with Azure

You need to authenticate with Azure using one of these methods:

#### Interactive Authentication

```python
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient

# Authenticate
credential = InteractiveBrowserCredential()

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="<subscription_id>",
    resource_group_name="<resource_group>",
    workspace_name="<workspace_name>"
)
```

#### Service Principal Authentication

For automation scenarios, use a service principal:

```python
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient

# Get credential
credential = ClientSecretCredential(
    tenant_id="<tenant_id>",
    client_id="<client_id>",
    client_secret="<client_secret>"
)

# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="<subscription_id>",
    resource_group_name="<resource_group>",
    workspace_name="<workspace_name>"
)
```

### Step 3: Register Models in Azure ML

Before running the inference component, register your models in Azure ML:

```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

# Register the model
model = Model(
    path="models/Glazing-Defects",
    type=AssetTypes.MLFLOW_MODEL,
    name="Glazing-Defects",
    description="Facade defect detection model"
)

registered_model = ml_client.models.create_or_update(model)
print(f"Registered model: {registered_model.name}, version: {registered_model.version}")
```

### Step 4: Register the Component

Register the FacadeAI Inference component in Azure ML:

```python
from azure.ai.ml import load_component

# Load component from YAML
inference_component = load_component(source="./component.yml")

# Register the component in the workspace
registered_component = ml_client.components.create_or_update(inference_component)
print(f"Registered component: {registered_component.name}, version: {registered_component.version}")
```

### Step 5: Create a Pipeline with the Component

Create and run an Azure ML pipeline that uses the FacadeAI Inference component:

```python
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml.entities import Pipeline
from azure.ai.ml import load_component

# Define the pipeline with the inference component
@dsl.pipeline(
    name="facadeai-inference-pipeline",
    description="Pipeline for Facade defect detection",
)
def facade_pipeline():
    inference = registered_component(
        input_ds=Input(type="uri_folder", path="azureml://datastores/blob_datastore/paths/images/"),
        model1=Input(type="mlflow_model", path="azureml://registries/models/Glazing-Defects/versions/1"),
        output_data=Output(type="uri_folder"),
        batch_id="B1",
        local=False,
        trace=True
    )
    return {"output": inference.outputs.output_data}

# Create the pipeline
pipeline = facade_pipeline()

# Submit the pipeline job
pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    experiment_name="facadeai-inference"
)
ml_client.jobs.stream(pipeline_job.name)
```

### Step 6: Configure Compute and Environment

For better control over the compute resources and environment, you can create and specify them explicitly:

#### Create Compute Target

```python
from azure.ai.ml.entities import AmlCompute

# Create compute target if it doesn't already exist
if not any(c.name == "gpu-cluster" for c in ml_client.compute.list()):
    compute = AmlCompute(
        name="gpu-cluster",
        size="Standard_NC6s_v3",  # GPU VM size
        min_instances=0,
        max_instances=4,
        tier="dedicated"
    )
    ml_client.compute.begin_create_or_update(compute).wait()

# Use the compute target in pipeline
pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    compute="gpu-cluster",
    experiment_name="facadeai-inference"
)
```

#### Create Custom Environment

```python
from azure.ai.ml.entities import Environment, BuildContext

# Create custom environment from Dockerfile
custom_env = Environment(
    name="facadeai-env",
    version="1.0.0",
    build=BuildContext(path="./"),  # Directory containing Dockerfile
    description="Environment for FacadeAI inference",
)

env = ml_client.environments.create_or_update(custom_env)

# Use the environment in the component
inference_component = load_component(source="./component.yml")
inference_component.environment = env.id
```

## Setting Up Data in Azure ML

### Step 1: Upload Images to Azure Blob Storage

First, upload your images to an Azure Blob storage container that is registered as a datastore in your Azure ML workspace.

Use the Azure Storage Explorer or Azure CLI:

```bash
az storage blob upload-batch --account-name <storage_account> \
    --auth-mode key --account-key <key> \
    --destination <container> --destination-path images/B3/cam \
    --source ./data/images/B3/cam
```

### Step 2: Register the Datastore in Azure ML

If not already registered, register the blob container as a datastore:

```python
from azure.ai.ml.entities import AzureBlobDatastore

blob_datastore = AzureBlobDatastore(
    name="blob_datastore",
    description="Datastore for facade images",
    account_name="<storage_account_name>",
    container_name="<container_name>",
    credentials={
        "account_key": "<account_key>"
    }
)

ml_client.datastores.create_or_update(blob_datastore)
```

## Monitoring and Managing Azure ML Pipelines

### Viewing Pipeline Jobs

Access your pipeline runs through:
- Azure ML Studio UI 
- Azure ML Python SDK

```python
# Get pipeline job by name
job = ml_client.jobs.get(name="<job_name>")

# List recent jobs
jobs = ml_client.jobs.list(max_results=10)
for job in jobs:
    print(job.name, job.status)
```

### Downloading Results

After the pipeline completes, download the results:

```python
import os
from azure.ai.ml.entities import Job

# Get the latest job
job = ml_client.jobs.get(name="<job_name>")

# Get output URI
output_uri = job.outputs.output_data.path

# Download the results
ml_client.jobs.download(
    name=job.name,
    output_name="output_data",
    download_path="./downloaded_results"
)
```

## Troubleshooting Azure ML Issues

### Common Problems and Solutions

1. **Authentication Errors**
   - Check that your credentials have the correct permissions
   - Verify tenant ID, subscription ID, and workspace information

2. **Model Loading Failures**
   - Ensure the model is registered correctly and accessible
   - Check for path errors in the model reference

3. **Compute Resource Limitations**
   - Verify quotas for the VM size you're requesting
   - Consider using a different region if resources are not available

4. **Environment Issues**
   - Check Docker build logs if using a custom environment
   - Verify that all dependencies are properly specified

### Viewing Detailed Logs

Access detailed logs to diagnose issues:

```python
# Get the driver log
logs = ml_client.jobs.get_logs(name="<job_name>", lines=100)
print(logs)

# Download all logs
ml_client.jobs.download_logs(name="<job_name>", download_path="./logs")
```