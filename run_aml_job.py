#!/usr/bin/env python3
"""
Run Azure ML Pipeline Job

This script is designed to be run from GitHub Actions to submit a job to 
Azure ML pipeline with the parameters specified in the issue.
"""

import os
import sys
import json
from datetime import datetime
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.ai.ml import MLClient, Input, load_job
from azure.core.exceptions import ClientAuthenticationError, ServiceRequestError


def check_azure_cli():
    """Check if Azure CLI is installed and logged in"""
    try:
        import subprocess
        version_result = subprocess.run(["az", "--version"], capture_output=True, text=True)
        if version_result.returncode != 0:
            return False, "Azure CLI not available"
        
        # Check if user is logged in
        login_result = subprocess.run(["az", "account", "show"], capture_output=True, text=True)
        if login_result.returncode != 0:
            return False, "Not logged in to Azure CLI"
        
        # Check if ML extension is installed
        ml_result = subprocess.run(["az", "ml", "-h"], capture_output=True, text=True)
        if ml_result.returncode != 0:
            return False, "Azure ML CLI extension not installed"
        
        # Get account details
        account_info = json.loads(login_result.stdout)
        return True, account_info
    except Exception as e:
        return False, str(e)


def verify_workspace(subscription_id, resource_group, workspace_name):
    """Verify that the workspace exists using Azure CLI"""
    try:
        import subprocess
        cmd = [
            "az", "ml", "workspace", "show",
            "--subscription", subscription_id,
            "--resource-group", resource_group,
def install_and_verify_azureml_cli():
    """Install and verify Azure ML CLI extension"""
    import subprocess
    try:
        # Check if Azure CLI is installed
        az_version = subprocess.run(["az", "--version"], capture_output=True, text=True)
        if az_version.returncode != 0:
            print("Azure CLI not installed or not found in PATH")
            return False
            
        # Check if ML extension is installed
        ml_help = subprocess.run(["az", "ml", "--help"], capture_output=True, text=True)
        if ml_help.returncode != 0:
            print("Azure ML CLI extension not found. Attempting to install...")
            install_result = subprocess.run(
                ["az", "extension", "add", "--name", "ml", "--yes"], 
                capture_output=True, 
                text=True
            )
            if install_result.returncode != 0:
                print(f"Failed to install ML extension: {install_result.stderr}")
                return False
            print("Successfully installed Azure ML CLI extension")
        
        # Try to list ML workspaces to validate extension works
        print("Testing Azure ML CLI extension...")
        list_workspaces = subprocess.run(
            ["az", "ml", "workspace", "list"], 
            capture_output=True, 
            text=True
        )
        if list_workspaces.returncode != 0:
            print(f"Failed to list workspaces: {list_workspaces.stderr}")
            return False
            
        print("Azure ML CLI extension is working properly")
        return True
    except Exception as e:
        print(f"Error verifying Azure ML CLI: {str(e)}")
        return False
            "--name", workspace_name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, result.stderr
        
        workspace_info = json.loads(result.stdout)
        return True, workspace_info
    except Exception as e:
        return False, str(e)


def validate_pipeline_yaml(yaml_path):
    """Validate pipeline YAML using Azure CLI"""
    try:
        import subprocess
        print(f"Validating pipeline YAML at {yaml_path}...")
        cmd = ["az", "ml", "job", "validate", "--file", yaml_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"YAML validation failed: {result.stderr}")
            return False, result.stderr
            
        print("YAML validation successful")
        return True, result.stdout
    except Exception as e:
        print(f"YAML validation error: {str(e)}")
        return False, str(e)

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
        
        # Check if we're running in GitHub Actions environment
        if os.environ.get("GITHUB_ACTIONS"):
            print("\nDetected GitHub Actions environment.")
            print("Make sure these secrets are configured in your repository:")
            print("  - AZURE_SUBSCRIPTION_ID")
            print("  - AZURE_TENANT_ID")
            print("  - AZURE_CLIENT_ID")
            print("  - AZURE_CLIENT_SECRET")
            
        print("\nChecking Azure CLI as fallback...")
        cli_available, cli_info = check_azure_cli()
        
        if cli_available:
            print("Azure CLI is available and logged in.")
            if isinstance(cli_info, dict):
                subscription_id = cli_info.get("id")
                print(f"Using subscription: {cli_info.get('name')} ({subscription_id})")
            
            # Verify workspace exists
            if subscription_id:
                print(f"\nVerifying workspace {workspace_name} in resource group {resource_group}...")
                workspace_exists, workspace_info = verify_workspace(
                    subscription_id, resource_group, workspace_name
                )
                
                if not workspace_exists:
                    print(f"Error: Workspace verification failed: {workspace_info}")
                    print(f"Please make sure the workspace {workspace_name} exists in resource group {resource_group}")
                    return 1
                    
                print(f"Workspace verified successfully!")
                if isinstance(workspace_info, dict):
                    print(f"Workspace location: {workspace_info.get('location')}")
        else:
            print(f"Azure CLI fallback failed: {cli_info}")
            print("Neither service principal credentials nor Azure CLI authentication are available.")
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
        if all([tenant_id, client_id, client_secret]):
            print("Using ClientSecretCredential for authentication...")
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
        else:
            print("Using DefaultAzureCredential for authentication...")
            credential = DefaultAzureCredential()
        
        # Connect to Azure ML workspace
        print(f"Connecting to Azure ML workspace: {workspace_name}")
        print(f"Resource Group: {resource_group}")
        if subscription_id:
            print(f"Subscription ID: {subscription_id[:8]}...") # Show only first 8 chars for security
        print(f"Region: West Europe")
        
        # Attempt to connect with more detailed error handling
        try:
            ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
            
            # Test connection by retrieving workspace info
            workspace_info = ml_client.workspaces.get(workspace_name)
            print(f"Successfully connected to workspace: {workspace_info.name}")
            print(f"Workspace location: {workspace_info.location}")
            
        except ClientAuthenticationError as auth_error:
            print(f"\nAuthentication error: {auth_error}")
            print("\nPlease check your Azure credentials.")
            print("For service principal authentication, verify:")
            print("  - AZURE_TENANT_ID is correct")
            print("  - AZURE_CLIENT_ID is correct")
            print("  - AZURE_CLIENT_SECRET is correct and not expired")
            print("  - The service principal has access to the subscription and workspace")
            
            # Try to get more info using Azure CLI
            if os.system("az account show > /dev/null 2>&1") == 0:
                print("\nTrying to list accessible workspaces using Azure CLI...")
                os.system(f"az ml workspace list --subscription {subscription_id} -o table")
            return 1
            
        except ServiceRequestError as req_error:
            print(f"\nService request error: {req_error}")
            print("\nThis could be due to:")
            print("  - Network connectivity issues")
            print("  - Azure service unavailability")
            print("  - Invalid subscription ID")
            print(f"  - Workspace '{workspace_name}' not existing in resource group '{resource_group}'")
            
            # Try to verify using Azure CLI
            if os.system("az account show > /dev/null 2>&1") == 0:
                print("\nVerifying workspace using Azure CLI...")
                os.system(f"az ml workspace show --name {workspace_name} --resource-group {resource_group} --subscription {subscription_id} -o table")
            return 1
            
        except Exception as e:
            print(f"Error connecting to Azure ML workspace: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Check if the error is related to authentication
            error_msg = str(e).lower()
            if "authentication" in error_msg or "credential" in error_msg:
                print("\nAuthentication error. Please check your Azure credentials.")
                print("Make sure AZURE_SUBSCRIPTION_ID, AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET are set correctly.")
                print("\nAlternatively, ensure you're logged in with 'az login' if using DefaultAzureCredential.")
            elif "not found" in error_msg or "does not exist" in error_msg:
                print(f"\nWorkspace '{workspace_name}' not found in resource group '{resource_group}'.")
                print("Please verify the workspace name and resource group.")
            
            return 1
            
    except Exception as e:
        print(f"Error initializing authentication: {str(e)}")
        import traceback
        traceback.print_exc()
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
        
        # Validate the YAML using Azure CLI if available
        if os.system("az --version > /dev/null 2>&1") == 0:
            if install_and_verify_azureml_cli():
                is_valid, validation_result = validate_pipeline_yaml(pipeline_yaml_path)
                if not is_valid:
                    print("WARNING: Pipeline YAML validation failed. This may cause job submission to fail.")
                    print("Attempting to continue with submission anyway...")
        
        # Load the pipeline job YAML
        try:
            # First validate the YAML file content
            with open(pipeline_yaml_path, 'r') as f:
                yaml_content = f.read()
                print(f"YAML file size: {len(yaml_content)} bytes")
                print(f"YAML file first line: {yaml_content.split('\\n')[0]}")
            
            # Load the job definition from YAML
            pipeline_job = load_job(pipeline_yaml_path)
            
            # Print some info about the loaded job
            print(f"Successfully loaded pipeline job from YAML.")
            print(f"Job type: {type(pipeline_job).__name__}")
            print(f"Job schema: {getattr(pipeline_job, '$schema', 'Not found')}")
            
            # Update pipeline job properties
            pipeline_job.display_name = "FacadeAI Pipeline Test Job"
            pipeline_job.description = "FacadeAI Pipeline Job submitted from GitHub Actions"
            pipeline_job.name = job_name
            
            # Print job inputs
            print("\nAvailable inputs:")
            for input_name in dir(pipeline_job.inputs):
                if not input_name.startswith('_'):
                    print(f"  - {input_name}")
                    
            # Update inputs
            try:
                pipeline_job.inputs.batch_id = batch_id
                pipeline_job.inputs.input_dataset = input_dataset
                pipeline_job.inputs.main_model = model1
                pipeline_job.inputs.processing_mode = mode
                pipeline_job.inputs.window_size = window_size
                pipeline_job.inputs.overlap = overlap
                pipeline_job.inputs.confidence_threshold = confidence
                
                # Print job settings
                print("\nSettings:")
                print(f"  - Default compute: {pipeline_job.settings.default_compute}")
                
                # Update compute target
                pipeline_job.settings.default_compute = compute_target
                
                # Print job structure
                print("\nJob structure:")
                for job_name, job in pipeline_job.jobs.items():
                    print(f"  - {job_name}: {type(job).__name__}")
                
                # Check if facade_inference is in the jobs dictionary
                if "facade_inference" not in pipeline_job.jobs:
                    print("\nWarning: 'facade_inference' job not found in the pipeline.")
                    print("Available jobs:", list(pipeline_job.jobs.keys()))
                
                # Print the keys available in the job
                first_job_name = next(iter(pipeline_job.jobs.keys()))
                print(f"\nFirst job ({first_job_name}) attributes:")
                first_job = pipeline_job.jobs[first_job_name]
                for attr_name in dir(first_job):
                    if not attr_name.startswith('_') and attr_name != 'jobs':
                        print(f"  - {attr_name}")
                
                # Update the compute target for the facade_inference job or the first job
                try:
                    target_job_name = "facade_inference" if "facade_inference" in pipeline_job.jobs else first_job_name
                    pipeline_job.jobs[target_job_name].compute = f"azureml:{compute_target}"
                    print(f"\nUpdated compute target for job '{target_job_name}' to '{compute_target}'")
                except AttributeError:
                    print(f"\nWarning: Could not update compute target for job. Will use default compute.")
                
                # Submit the job using SDK first, and fallback to CLI if that fails
                try:
                    print("\nSubmitting pipeline job using SDK...")
                    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
                    
                    print(f"\nJob submitted successfully!")
                    print(f"Job Name: {submitted_job.name}")
                    print(f"Job ID: {submitted_job.id}")
                    
                    # Don't wait for the job to complete as per requirement
                    print("\nJob has been submitted. Not waiting for completion.")
                    return 0
                    
                except Exception as sdk_error:
                    print(f"\nError submitting job using SDK: {str(sdk_error)}")
                    print("Attempting to submit using Azure CLI as fallback...")
                    
                    # Try Azure CLI submission if SDK fails
                    if install_and_verify_azureml_cli():
                        try:
                            import subprocess
                            import tempfile
                            
                            # Create a temporary modified YAML file for CLI submission
                            with tempfile.NamedTemporaryFile(suffix='.yml', delete=False, mode='w') as temp_file:
                                modified_yaml = yaml_content.replace("${{parent.inputs.", "${{inputs.")
                                temp_file.write(modified_yaml)
                                temp_yaml_path = temp_file.name
                            
                            cmd = [
                                "az", "ml", "job", "create",
                                "--file", temp_yaml_path,
                                "--name", job_name,
                                "--resource-group", resource_group,
                                "--workspace-name", workspace_name,
                                "--set", f"inputs.batch_id={batch_id}",
                                "--set", f"inputs.input_dataset={input_dataset}",
                                "--set", f"inputs.main_model={model1}",
                                "--set", f"inputs.processing_mode={mode}",
                                "--set", f"inputs.window_size={window_size}",
                                "--set", f"inputs.overlap={overlap}",
                                "--set", f"inputs.confidence_threshold={confidence}",
                                "--set", f"jobs.facade_inference.compute=azureml:{compute_target}",
                                "--set", f"settings.default_compute=azureml:{compute_target}"
                            ]
                            
                            print(f"Running CLI command: {' '.join(cmd)}")
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            
                            if result.returncode == 0:
                                cli_response = json.loads(result.stdout)
                                print(f"\nJob submitted successfully using CLI!")
                                print(f"Job Name: {cli_response.get('name')}")
                                print(f"Job ID: {cli_response.get('id')}")
                                return 0
                            else:
                                print(f"Azure CLI job submission failed: {result.stderr}")
                                return 1
                                
                        except Exception as cli_error:
                            print(f"Error during CLI submission: {str(cli_error)}")
                            return 1
                    else:
                        print("Azure CLI not available for fallback submission.")
                        return 1
                
            except AttributeError as ae:
                print(f"\nError updating pipeline inputs: {str(ae)}")
                print("This is likely due to a mismatch between the pipeline.yml structure and the code.")
                
                # Print detailed debug information
                import json
                try:
                    # Try to convert to dict and print structure
                    pipeline_dict = pipeline_job.to_dict()
                    print("\nPipeline structure:")
                    print(json.dumps(pipeline_dict, indent=2)[:500] + "...")  # Print first 500 chars
                except Exception as e:
                    print(f"Could not convert pipeline to dict: {str(e)}")
                
                return 1
                
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