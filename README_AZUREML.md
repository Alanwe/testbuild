# FacadeAI on AzureML

This document provides an overview of deploying FacadeAI components to Azure Machine Learning (AzureML) and running them as part of AzureML pipelines.

## Overview

FacadeAI components can be deployed to Azure ML to leverage cloud computing resources for facade defect detection. This allows for:

- Processing large volumes of images in parallel
- Using GPU resources for faster inference
- Automating the pipeline workflow
- Centralized model management
- Result storage in Azure services like CosmosDB

## Getting Started

1. Follow the [SETUP.md](SETUP.md) guide to configure your development environment for AzureML integration.

2. Install Azure CLI and ML extension:

```bash
./install_azure_cli.sh
```

## Components

The FacadeAI AzureML integration consists of these key files:

- `component.yml` - Defines the FacadeAI inference component for AzureML
- `pipeline.yml` - Defines a pipeline that uses the FacadeAI component
- `azureml-job.yml` - Defines a command job that can be submitted directly via GitHub Actions
- `submit_job.sh` - Script for submitting pipeline jobs
- `collect_outputs.sh` - Script for downloading results from completed jobs
- `.github/workflows/submit-azureml-job.yml` - GitHub workflow for submitting jobs via GitHub Actions

## Running a Pipeline

To run the FacadeAI pipeline on AzureML:

1. Make sure you have set up your Azure ML connection as described in SETUP.md.

2. Submit a job:

```bash
./submit_job.sh -b B42 -m Glazing-Defects -v 1
```

3. Retrieve results when the job is completed:

```bash
./collect_outputs.sh -j <JOB_NAME> -o ./results
```

## Parameter Reference

### Pipeline Parameters

The AzureML pipeline supports these parameters:

- `batch_id` - ID of the image batch to process
- `input_dataset` - Path to input data in Azure ML
- `output_path` - Path where to store results
- `main_model` - Main defect detection model
- `second_model` - Optional secondary model
- `window_size` - Size of sliding window (default: 512)
- `overlap` - Overlap between windows (default: 64)
- `confidence_threshold` - Detection confidence threshold (default: 50)
- `processing_mode` - Whether to force reprocessing of all images ("force") or only process new ones ("auto")

## Customization

You can customize the pipeline behavior by modifying these files:

- `pipeline.yml` - To change compute resources, add more models, or adjust parameters
- `component.yml` - To modify component behavior and parameters
- `submit_job.sh` - To add more command-line options for job submission

## Monitoring and Debugging

- Use Azure ML Studio UI to monitor jobs
- Check job logs in Azure ML Studio
- Use `collect_outputs.sh` to download results and logs

## Testing

Use the verification job to test your AzureML setup:

```bash
az ml job create -f tests/verify_setup.yml
```

## Related Documentation

- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [FacadeAI Development Guide](DEV_SETUP.md)

## GitHub Actions Integration

FacadeAI can be integrated with GitHub Actions for automated job submission:

1. The `.github/workflows/submit-azureml-job.yml` workflow automatically submits jobs to AzureML 
   using the `azureml-job.yml` definition.

2. Required parameters for `azureml-job.yml`:
   - `batch_id` - ID of the batch to process
   - `model` - MLflow model reference (e.g., "azureml:Glazing-Defects:1")
   - `input_dataset` - URI folder path (e.g., "azureml:facade-inference-image-batches:1")

3. You can trigger the workflow manually from the GitHub Actions tab with custom parameters.