#!/bin/bash
# =============================================================================
# submit_aml_job.sh - Submit Azure ML pipeline job
# =============================================================================
# This script submits a pipeline job to Azure ML using the Azure CLI
# =============================================================================
set -e

# Display header
echo "========================================"
echo "Azure ML Pipeline Job Submission"
echo "========================================"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
  echo "Error: Azure CLI is not installed."
  echo "Please install with: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
  exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
  # Try to login with service principal if environment variables are set
  if [ -n "$AZURE_CLIENT_ID" ] && [ -n "$AZURE_CLIENT_SECRET" ] && [ -n "$AZURE_TENANT_ID" ]; then
    echo "Logging in with service principal..."
    az login --service-principal \
      --username "$AZURE_CLIENT_ID" \
      --password "$AZURE_CLIENT_SECRET" \
      --tenant "$AZURE_TENANT_ID"
  else
    echo "Not logged in to Azure. Please run 'az login' or provide service principal credentials."
    exit 1
  fi
fi

# Set default parameters
SUBSCRIPTION_ID=${AZURE_SUBSCRIPTION_ID}
RESOURCE_GROUP=${AZURE_RESOURCE_GROUP:-"Facade-MLRG"}
WORKSPACE_NAME=${AZURE_WORKSPACE_NAME:-"Facade-AML"}
BATCH_ID=${BATCH_ID:-"3"}
MODE=${MODE:-"auto"}
WINDOW_SIZE=${WINDOW_SIZE:-"512"}
OVERLAP=${OVERLAP:-"64"}
CONFIDENCE=${CONFIDENCE:-"30"}
MODEL1_CONFIDENCE=${MODEL1_CONFIDENCE:-"50"}
COMPUTE_TARGET=${COMPUTE_TARGET:-"cpu-cluster2"}

# Create a timestamp for the job name
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
JOB_NAME=${JOB_NAME:-"cli_facadeai_${TIMESTAMP}"}

# Set Azure CLI defaults
echo "Setting Azure defaults..."
az configure --defaults group="$RESOURCE_GROUP" workspace="$WORKSPACE_NAME"

# Install ML extension if not present
if ! az ml -h &> /dev/null; then
  echo "Installing Azure ML CLI extension..."
  az extension add --name ml --yes
fi

# Ensure the pipeline.yml file exists
if [ ! -f "pipeline.yml" ]; then
  echo "Error: pipeline.yml file not found!"
  exit 1
fi

# Display job parameters
echo "Job Parameters:"
echo "========================================"
echo "Subscription ID:    $SUBSCRIPTION_ID"
echo "Resource Group:     $RESOURCE_GROUP"
echo "Workspace:          $WORKSPACE_NAME"
echo "Job Name:           $JOB_NAME"
echo "Batch ID:           $BATCH_ID"
echo "Mode:               $MODE"
echo "Window Size:        $WINDOW_SIZE"
echo "Overlap:            $OVERLAP"
echo "Confidence:         $CONFIDENCE"
echo "Model1 Confidence:  $MODEL1_CONFIDENCE"
echo "Compute Target:     $COMPUTE_TARGET"
echo "========================================"

# Submit the job
echo "Submitting job..."
JOB_OUTPUT=$(az ml job create \
  --file pipeline.yml \
  --subscription "$SUBSCRIPTION_ID" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE_NAME" \
  --name "$JOB_NAME" \
  --set inputs.batch_id="$BATCH_ID" \
  --set inputs.input_dataset="azureml:facade-inference-image-batches:1" \
  --set inputs.main_model="azureml:Glazing-Defects:1" \
  --set inputs.processing_mode="$MODE" \
  --set inputs.window_size="$WINDOW_SIZE" \
  --set inputs.overlap="$OVERLAP" \
  --set inputs.confidence_threshold="$CONFIDENCE" \
  --set settings.default_compute="azureml:$COMPUTE_TARGET" \
  --set jobs.facade_inference.compute="azureml:$COMPUTE_TARGET" \
  --output json)

# Extract and display job details
JOB_ID=$(echo "$JOB_OUTPUT" | jq -r '.id')
CREATED_JOB_NAME=$(echo "$JOB_OUTPUT" | jq -r '.name')

echo "========================================"
echo "Job submitted successfully!"
echo "Job Name: $CREATED_JOB_NAME"
echo "Job ID: $JOB_ID"
echo "========================================"