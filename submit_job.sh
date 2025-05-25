#!/bin/bash
# =============================================================================
# submit_job.sh - Submit FacadeAI pipeline job to Azure ML
# =============================================================================
# This script submits a pipeline job to Azure ML using the pipeline.yml
# definition. It handles configuration, parameters, and provides job tracking.
# =============================================================================
set -e

# Default values
BATCH_ID=""
MODEL_NAME="Glazing-Defects"
MODEL_VERSION="1"
SECOND_MODEL=""
SECOND_MODEL_VERSION=""
WINDOW_SIZE=512
OVERLAP=64
CONFIDENCE=50
MODE="auto"
EXPERIMENT_NAME="FacadeAI"

# Function to display usage information
show_usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -b, --batch-id BATCH_ID       Batch ID to process (required)"
  echo "  -m, --model MODEL_NAME        Primary model name (default: Glazing-Defects)"
  echo "  -v, --version MODEL_VERSION   Primary model version (default: 1)"
  echo "  -s, --second-model MODEL      Second model name (optional)"
  echo "  -w, --window-size SIZE        Window size (default: 512)"
  echo "  -o, --overlap OVERLAP         Window overlap (default: 64)"
  echo "  -c, --confidence THRESHOLD    Confidence threshold (default: 50)"
  echo "  -f, --force                   Force reprocessing of all images"
  echo "  -e, --experiment NAME         Experiment name (default: FacadeAI)"
  echo "  -h, --help                    Show this help message"
  echo ""
  echo "Example:"
  echo "  $0 -b B42 -m Glazing-Defects -v 2 -c 30"
  echo ""
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -b|--batch-id) BATCH_ID="$2"; shift ;;
    -m|--model) MODEL_NAME="$2"; shift ;;
    -v|--version) MODEL_VERSION="$2"; shift ;;
    -s|--second-model) SECOND_MODEL="$2"; shift ;;
    --second-version) SECOND_MODEL_VERSION="$2"; shift ;;
    -w|--window-size) WINDOW_SIZE="$2"; shift ;;
    -o|--overlap) OVERLAP="$2"; shift ;;
    -c|--confidence) CONFIDENCE="$2"; shift ;;
    -f|--force) MODE="force" ;;
    -e|--experiment) EXPERIMENT_NAME="$2"; shift ;;
    -h|--help) show_usage; exit 0 ;;
    *) echo "Unknown parameter: $1"; show_usage; exit 1 ;;
  esac
  shift
done

# Validate required parameters
if [ -z "$BATCH_ID" ]; then
  echo "Error: Batch ID is required"
  show_usage
  exit 1
fi

# Check if Azure CLI and ML extension are installed
if ! command -v az &> /dev/null || ! az extension list --query "[?name=='ml'].version" -o tsv | grep -q "."; then
  echo "Azure CLI with ML extension is required but not installed."
  echo "Please run ./install_azure_cli.sh first."
  exit 1
fi

# Check if logged in to Azure
if ! az account show &> /dev/null; then
  echo "You are not logged in to Azure. Please run 'az login' first."
  exit 1
fi

# Display submission details
echo "========================================"
echo "FacadeAI Pipeline Job Submission"
echo "========================================"
echo "Batch ID:          $BATCH_ID"
echo "Primary Model:     $MODEL_NAME:$MODEL_VERSION"
if [ -n "$SECOND_MODEL" ]; then
  if [ -n "$SECOND_MODEL_VERSION" ]; then
    echo "Secondary Model:   $SECOND_MODEL:$SECOND_MODEL_VERSION" 
  else
    echo "Secondary Model:   $SECOND_MODEL:1"
  fi
fi
echo "Window Size:       $WINDOW_SIZE"
echo "Overlap:           $OVERLAP"
echo "Confidence:        $CONFIDENCE"
echo "Mode:              $MODE"
echo "Experiment:        $EXPERIMENT_NAME"
echo "========================================"

# Prepare job name with timestamp
TIMESTAMP=$(date +"%Y%m%d%H%M%S")
JOB_NAME="facadeai_${BATCH_ID}_${TIMESTAMP}"
echo "Job Name:          $JOB_NAME"
echo "========================================"

# Confirm submission
read -p "Submit job to Azure ML? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
  echo "Job submission cancelled."
  exit 0
fi

# Construct model reference strings
PRIMARY_MODEL="azureml:$MODEL_NAME:$MODEL_VERSION"
SECOND_MODEL_REF=""
if [ -n "$SECOND_MODEL" ]; then
  if [ -n "$SECOND_MODEL_VERSION" ]; then
    SECOND_MODEL_REF="--set inputs.second_model=$SECOND_MODEL:$SECOND_MODEL_VERSION"
  else
    SECOND_MODEL_REF="--set inputs.second_model=$SECOND_MODEL:1"
  fi
fi

echo "Submitting job to Azure ML..."
echo ""

# Submit the job
az ml job create \
  --file pipeline.yml \
  --name "$JOB_NAME" \
  --experiment-name "$EXPERIMENT_NAME" \
  --set inputs.batch_id=$BATCH_ID \
  --set inputs.main_model="$PRIMARY_MODEL" \
  $SECOND_MODEL_REF \
  --set inputs.window_size=$WINDOW_SIZE \
  --set inputs.overlap=$OVERLAP \
  --set inputs.confidence_threshold=$CONFIDENCE \
  --set inputs.processing_mode=$MODE

# Check if job was created successfully
if [ $? -eq 0 ]; then
  echo ""
  echo "Job submitted successfully!"
  echo ""
  echo "To monitor job status:"
  echo "  az ml job show -n $JOB_NAME"
  echo ""
  echo "To stream job logs:"
  echo "  az ml job stream -n $JOB_NAME"
  echo ""
  echo "To download job outputs when completed:"
  echo "  ./collect_outputs.sh -j $JOB_NAME -o ./outputs"
  echo ""
else
  echo ""
  echo "Job submission failed!"
  echo ""
  echo "Please check your parameters and Azure ML setup."
  echo "See SETUP.md for more information on configuring your environment."
  exit 1
fi