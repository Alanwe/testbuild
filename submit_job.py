#!/usr/bin/env python3
"""
Submit FacadeAI pipeline job to Azure ML using the Python SDK

This script demonstrates how to programmatically submit Azure ML pipeline
jobs using the Python SDK as an alternative to the CLI-based approach.
"""

import os
import sys
import argparse
from datetime import datetime
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, load_component, dsl, Output
from azure.ai.ml.entities import Component, JobInput
from azure.ai.ml.constants import AssetTypes

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Submit FacadeAI job to Azure ML')
    parser.add_argument('-b', '--batch-id', type=str, required=True,
                        help='Batch ID to process')
    parser.add_argument('-m', '--model', type=str, default="Glazing-Defects",
                        help='Primary model name (default: Glazing-Defects)')
    parser.add_argument('-v', '--version', type=str, default="1",
                        help='Primary model version (default: 1)')
    parser.add_argument('-c', '--confidence', type=int, default=50,
                        help='Confidence threshold (default: 50)')
    parser.add_argument('-e', '--experiment', type=str, default="FacadeAI",
                        help='Experiment name (default: FacadeAI)')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force reprocessing of all images')
    
    return parser.parse_args()

def main():
    """Main function to submit pipeline job"""
    # Parse arguments
    args = parse_arguments()
    
    # Configure job parameters
    batch_id = args.batch_id
    model_name = args.model
    model_version = args.version
    confidence = args.confidence
    experiment_name = args.experiment
    processing_mode = "force" if args.force else "auto"
    
    # Create job name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    job_name = f"facadeai_{batch_id}_{timestamp}"
    
    print(f"Submitting job: {job_name}")
    print(f"Batch ID: {batch_id}")
    print(f"Model: {model_name}:{model_version}")
    print(f"Confidence: {confidence}")
    print(f"Mode: {processing_mode}")
    print(f"Experiment: {experiment_name}")
    print("-" * 40)
    
    # Connect to AzureML workspace
    try:
        # Use DefaultAzureCredential which supports multiple authentication methods
        credential = DefaultAzureCredential()
        
        # Try to load configuration from .azureml/config.json first
        config_file = os.path.expanduser("~/.azureml/config.json")
        if os.path.exists(config_file):
            print(f"Loading configuration from {config_file}")
            ml_client = MLClient.from_config(credential=credential)
        else:
            # Alternatively, use environment variables or parameters
            subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
            resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
            workspace_name = os.environ.get("AZURE_WORKSPACE_NAME")
            
            if not all([subscription_id, resource_group, workspace_name]):
                print("Error: Azure ML configuration not found.")
                print("Either create a .azureml/config.json file or set environment variables:")
                print("  AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME")
                return 1
            
            ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
        
        print(f"Connected to workspace: {ml_client.workspace_name}")
    except Exception as e:
        print(f"Error connecting to Azure ML workspace: {str(e)}")
        return 1
    
    # Define pipeline inputs
    inputs = {
        "batch_id": batch_id,
        "input_dataset": Input(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/facades/images/"),
        "main_model": Input(
            type="mlflow_model",
            path=f"azureml:{model_name}:{model_version}"
        ),
        "confidence_threshold": confidence,
        "processing_mode": processing_mode,
    }
    
    # Define pipeline outputs
    outputs = {
        "pipeline_results": Output(
            type="uri_folder",
            path=f"azureml://datastores/workspaceblobstore/paths/facades/results/{batch_id}/{timestamp}"
        )
    }
    
    # Submit the pipeline job
    try:
        print("Submitting pipeline job...")
        # Load the pipeline YAML
        job = ml_client.jobs.create_or_update(
            job={"file": "pipeline.yml"},
            inputs=inputs,
            outputs=outputs,
            experiment_name=experiment_name,
            name=job_name
        )
        
        print(f"Job '{job_name}' submitted successfully.")
        print(f"Job ID: {job.id}")
        print(f"You can monitor the job at: {job.studio_url}")
        return 0
    except Exception as e:
        print(f"Error submitting job: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())