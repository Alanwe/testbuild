#!/bin/bash
# =============================================================================
# run_test_job.sh - Run a test FacadeAI pipeline job on Azure ML
# =============================================================================
# This script runs a test pipeline job on Azure ML with the parameters
# specified in the issue.
# =============================================================================
set -e

# Display header
echo "========================================"
echo "FacadeAI Test Pipeline Job Submission"
echo "========================================"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
  echo "Error: Azure CLI is not installed."
  echo "Please run ./install_azure_cli.sh first."
  exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
  echo "You are not logged in to Azure. Please run 'az login' first."
  exit 1
fi

# Configure Azure ML workspace 
echo "Configuring Azure ML workspace..."
az configure --defaults group="Facade-MLRG" workspace="Facade-AML"

# Create timestamp for job name
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
JOB_NAME="test_facadeai_${TIMESTAMP}"

# Set parameters from issue
BATCH_ID="3"
MODE="auto"
WINDOW_SIZE="512"
OVERLAP="64"
CONFIDENCE="30"
MODEL1_CONFIDENCE="50"
MODEL2_CONFIDENCE="50"
MODEL3_CONFIDENCE="50"
MODEL4_CONFIDENCE="50"
MODEL5_CONFIDENCE="50"
MODEL6_CONFIDENCE="50"
MODEL7_CONFIDENCE="50"
MODEL8_CONFIDENCE="50"
COSMOS_DB="https://facadestudio.documents.azure.com:443/;AccountKey=******;"
KEY_VAULT_URL="https://facade-keyvault.vault.azure.net/"
COSMOS_DB_NAME="FacadeDB"
COSMOS_CONTAINER_NAME="Images"
LOCAL="False"
TRACE="False"

echo "Job Name:          $JOB_NAME"
echo "Batch ID:          $BATCH_ID"
echo "Mode:              $MODE"
echo "Window Size:       $WINDOW_SIZE"
echo "Overlap:           $OVERLAP"
echo "Confidence:        $CONFIDENCE"
echo "Model1 Confidence: $MODEL1_CONFIDENCE"
echo "Compute Target:    cpu-cluster2"
echo "========================================"

# Submit the job with all parameters from the issue
echo "Submitting job to Azure ML..."
echo ""

JOB_OUTPUT=$(az ml job create \
  --file pipeline.yml \
  --name "$JOB_NAME" \
  --set inputs.batch_id=$BATCH_ID \
  --set inputs.input_dataset="azureml:facade-inference-image-batches:1" \
  --set inputs.main_model="azureml:Glazing-Defects:1" \
  --set inputs.window_size=$WINDOW_SIZE \
  --set inputs.overlap=$OVERLAP \
  --set inputs.confidence_threshold=$CONFIDENCE \
  --set inputs.processing_mode=$MODE \
  --set jobs.facade_inference.compute="azureml:cpu-cluster2" \
  --set jobs.facade_inference.inputs.batch_id=$BATCH_ID \
  --set jobs.facade_inference.inputs.input_ds="azureml:facade-inference-image-batches:1" \
  --set jobs.facade_inference.inputs.model1="azureml:Glazing-Defects:1" \
  --set jobs.facade_inference.inputs.mode=$MODE \
  --set jobs.facade_inference.inputs.window_size=$WINDOW_SIZE \
  --set jobs.facade_inference.inputs.overlap=$OVERLAP \
  --set jobs.facade_inference.inputs.confidence=$CONFIDENCE \
  --set jobs.facade_inference.inputs.model1_confidence=$MODEL1_CONFIDENCE \
  --set jobs.facade_inference.inputs.model2_confidence=$MODEL2_CONFIDENCE \
  --set jobs.facade_inference.inputs.model3_confidence=$MODEL3_CONFIDENCE \
  --set jobs.facade_inference.inputs.model4_confidence=$MODEL4_CONFIDENCE \
  --set jobs.facade_inference.inputs.model5_confidence=$MODEL5_CONFIDENCE \
  --set jobs.facade_inference.inputs.model6_confidence=$MODEL6_CONFIDENCE \
  --set jobs.facade_inference.inputs.model7_confidence=$MODEL7_CONFIDENCE \
  --set jobs.facade_inference.inputs.model8_confidence=$MODEL8_CONFIDENCE \
  --set jobs.facade_inference.inputs.cosmos_db="$COSMOS_DB" \
  --set jobs.facade_inference.inputs.key_vault_url="$KEY_VAULT_URL" \
  --set jobs.facade_inference.inputs.cosmos_db_name="$COSMOS_DB_NAME" \
  --set jobs.facade_inference.inputs.cosmos_container_name="$COSMOS_CONTAINER_NAME" \
  --set jobs.facade_inference.inputs.local=$LOCAL \
  --set jobs.facade_inference.inputs.trace=$TRACE \
  --query name -o tsv)

# Display completion message
echo ""
echo "Job submitted successfully!"
echo "Job Name: $JOB_OUTPUT"
echo ""
echo "As requested, not waiting for job completion."
echo ""