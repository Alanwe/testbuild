#!/usr/bin/env python3
"""
Run Azure ML Pipeline Job

This script is designed to be run from GitHub Actions to submit a job to 
Azure ML pipeline with the parameters specified in the issue.
"""

import os
import sys
from datetime import datetime
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.ai.ml import MLClient, Input, load_job

def main():
    """Main function to submit pipeline job with parameters from the issue"""
    
    # Get Azure credentials from environment variables
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "Facade-MLRG")
    workspace_name = os.environ.get("AZURE_WORKSPACE_NAME", "Facade-AML")
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    
    # Check if all required environment variables are set
    if not all([subscription_id, tenant_id, client_id, client_secret]):
        print("Error: Required Azure credentials not found in environment variables.")
        print("Please set AZURE_SUBSCRIPTION_ID, AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET")
        return 1
    
    # Get parameters from environment or use defaults from issue
    batch_id = os.environ.get("BATCH_ID", "3")
    mode = os.environ.get("MODE", "auto")
    window_size = int(os.environ.get("WINDOW_SIZE", "512"))
    overlap = int(os.environ.get("OVERLAP", "64"))
    confidence = int(os.environ.get("CONFIDENCE", "30"))
    model1_confidence = int(os.environ.get("MODEL1_CONFIDENCE", "50"))
    model2_confidence = int(os.environ.get("MODEL2_CONFIDENCE", "50"))
    model3_confidence = int(os.environ.get("MODEL3_CONFIDENCE", "50"))
    model4_confidence = int(os.environ.get("MODEL4_CONFIDENCE", "50"))
    model5_confidence = int(os.environ.get("MODEL5_CONFIDENCE", "50"))
    model6_confidence = int(os.environ.get("MODEL6_CONFIDENCE", "50"))
    model7_confidence = int(os.environ.get("MODEL7_CONFIDENCE", "50"))
    model8_confidence = int(os.environ.get("MODEL8_CONFIDENCE", "50"))
    
    # Constants from issue
    input_dataset = "azureml:facade-inference-image-batches:1"
    model1 = "azureml:Glazing-Defects:1"
    cosmos_db = os.environ.get("COSMOS_DB", "https://facadestudio.documents.azure.com:443/;AccountKey=******;")
    key_vault_url = os.environ.get("KEY_VAULT_URL", "https://facade-keyvault.vault.azure.net/")
    cosmos_db_name = os.environ.get("COSMOS_DB_NAME", "FacadeDB")
    cosmos_container_name = os.environ.get("COSMOS_CONTAINER_NAME", "Images")
    compute_target = os.environ.get("COMPUTE_TARGET", "cpu-cluster2")
    
    print(f"Connecting to Azure ML workspace: {workspace_name}")
    print(f"Resource Group: {resource_group}")
    print(f"Subscription ID: {subscription_id[:8]}...") # Show only first 8 chars for security
    print(f"Region: West Europe")
    
    try:
        # Create credential object
        credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        
        # Connect to Azure ML workspace
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        print(f"Successfully connected to workspace: {ml_client.workspace_name}")
    except Exception as e:
        print(f"Error connecting to Azure ML workspace: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Check if the error is related to authentication
        error_msg = str(e).lower()
        if "authentication" in error_msg or "credential" in error_msg:
            print("\nAuthentication error. Please check your Azure credentials.")
            print("Make sure AZURE_SUBSCRIPTION_ID, AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET are set correctly.")
        
        return 1
    
    # Create a unique job name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    job_name = f"github_facadeai_{run_id}_{timestamp}"
    
    # Print job parameters
    print(f"\nSubmitting job with parameters:")
    print(f"Job Name: {job_name}")
    print(f"Batch ID: {batch_id}")
    print(f"Model: {model1}")
    print(f"Mode: {mode}")
    print(f"Window Size: {window_size}")
    print(f"Overlap: {overlap}")
    print(f"Confidence: {confidence}")
    print(f"Model1 Confidence: {model1_confidence}")
    
    try:
        # Get the pipeline YAML file path
        pipeline_yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.yml")
        print(f"Loading pipeline YAML from: {pipeline_yaml_path}")
        
        if not os.path.exists(pipeline_yaml_path):
            print(f"Error: Pipeline YAML file not found at {pipeline_yaml_path}")
            return 1
        
        # Load the pipeline job YAML
        try:
            pipeline_job = load_job(pipeline_yaml_path)
            
            # Update pipeline job properties
            pipeline_job.display_name = "FacadeAI Pipeline Test Job"
            pipeline_job.description = "FacadeAI Pipeline Job submitted from GitHub Actions"
            pipeline_job.name = job_name
            
            # Update inputs
            pipeline_job.inputs.batch_id = batch_id
            pipeline_job.inputs.input_dataset = input_dataset
            pipeline_job.inputs.main_model = model1
            pipeline_job.inputs.processing_mode = mode
            pipeline_job.inputs.window_size = window_size
            pipeline_job.inputs.overlap = overlap
            pipeline_job.inputs.confidence_threshold = confidence
            
            # Update compute target
            pipeline_job.settings.default_compute = compute_target
            pipeline_job.jobs["facade_inference"].compute = f"azureml:{compute_target}"
            
            # Submit the job
            print("\nSubmitting pipeline job...")
            submitted_job = ml_client.jobs.create_or_update(pipeline_job)
            
            print(f"\nJob submitted successfully!")
            print(f"Job Name: {submitted_job.name}")
            print(f"Job ID: {submitted_job.id}")
            
            # Don't wait for the job to complete as per requirement
            print("\nJob has been submitted. Not waiting for completion.")
            return 0
            
        except Exception as e:
            print(f"Error preparing pipeline job: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
            
    except Exception as e:
        print(f"Error submitting job: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())