#!/bin/bash
# =============================================================================
# collect_outputs.sh - Download outputs from Azure ML jobs
# =============================================================================
# This script retrieves output data from completed Azure ML jobs. It can be
# used to download results for analysis or testing.
# =============================================================================
set -e

# Default values
JOB_NAME=""
EXPERIMENT_NAME=""
OUTPUT_DIR="./job_outputs"
DOWNLOAD_MODS=false
OUTPUT_NAME="pipeline_results"
LATEST_JOB=false

# Function to display usage information
show_usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -j, --job JOB_NAME          Job name to download from"
  echo "  -e, --experiment NAME       Experiment name to search for jobs"
  echo "  -l, --latest                Download from the latest job in the experiment"
  echo "  -o, --output-dir DIR        Output directory (default: ./job_outputs)"
  echo "  -n, --output-name NAME      Output name to download (default: pipeline_results)"
  echo "  -m, --download-models       Also download models used in the job"
  echo "  -h, --help                  Show this help message"
  echo ""
  echo "Example:"
  echo "  $0 -j facadeai_B42_20230615123045 -o ./my_results"
  echo "  $0 -e FacadeAI -l -o ./latest_results"
  echo ""
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -j|--job) JOB_NAME="$2"; shift ;;
    -e|--experiment) EXPERIMENT_NAME="$2"; shift ;;
    -l|--latest) LATEST_JOB=true ;;
    -o|--output-dir) OUTPUT_DIR="$2"; shift ;;
    -n|--output-name) OUTPUT_NAME="$2"; shift ;;
    -m|--download-models) DOWNLOAD_MODELS=true ;;
    -h|--help) show_usage; exit 0 ;;
    *) echo "Unknown parameter: $1"; show_usage; exit 1 ;;
  esac
  shift
done

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

# If latest flag is set, we need experiment name
if [ "$LATEST_JOB" = true ] && [ -z "$EXPERIMENT_NAME" ]; then
  echo "Error: Experiment name (-e) is required when using --latest flag"
  show_usage
  exit 1
fi

# If latest flag is set, find the latest job
if [ "$LATEST_JOB" = true ]; then
  echo "Finding latest job in experiment: $EXPERIMENT_NAME..."
  LATEST_JOB_DATA=$(az ml job list --experiment-name "$EXPERIMENT_NAME" --query "sort_by([?status=='Completed'], &properties.creationContext.createdTime)[-1]" -o json)
  
  if [ "$LATEST_JOB_DATA" = "null" ] || [ -z "$LATEST_JOB_DATA" ]; then
    echo "No completed jobs found in experiment: $EXPERIMENT_NAME"
    exit 1
  fi
  
  JOB_NAME=$(echo $LATEST_JOB_DATA | jq -r '.name')
  echo "Found latest job: $JOB_NAME"
fi

# Validate required parameters
if [ -z "$JOB_NAME" ]; then
  echo "Error: Job name is required"
  show_usage
  exit 1
fi

# Get job details
echo "Retrieving job details for: $JOB_NAME..."
JOB_DATA=$(az ml job show -n "$JOB_NAME" -o json)

if [ $? -ne 0 ]; then
  echo "Error: Failed to retrieve job data. Check if the job exists."
  exit 1
fi

# Check job status
JOB_STATUS=$(echo $JOB_DATA | jq -r '.status')
if [ "$JOB_STATUS" != "Completed" ]; then
  echo "Warning: Job is not in 'Completed' state. Current status: $JOB_STATUS"
  read -p "Continue with download anyway? (y/n): " confirm
  if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Download cancelled."
    exit 0
  fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Output will be saved to: $OUTPUT_DIR"

# Download outputs
echo "========================================"
echo "Downloading job outputs..."
echo "========================================"

# Get output paths
echo "Retrieving output paths..."
OUTPUT_PATH=$(echo $JOB_DATA | jq -r ".outputs.$OUTPUT_NAME.assetId")

if [ -z "$OUTPUT_PATH" ] || [ "$OUTPUT_PATH" = "null" ]; then
  echo "Error: Output '$OUTPUT_NAME' not found in job outputs"
  echo "Available outputs:"
  echo $JOB_DATA | jq -r '.outputs | keys[]'
  exit 1
fi

# Download the output data
echo "Downloading output data from: $OUTPUT_PATH"
az ml job download -n "$JOB_NAME" --output-name "$OUTPUT_NAME" --download-path "$OUTPUT_DIR"

# Check download success
if [ $? -eq 0 ]; then
  echo "Successfully downloaded output data to: $OUTPUT_DIR"
else
  echo "Error downloading output data"
  exit 1
fi

# Download models if requested
if [ "$DOWNLOAD_MODELS" = true ]; then
  echo "========================================"
  echo "Downloading models used in the job..."
  echo "========================================"
  
  # Create models directory
  MODELS_DIR="$OUTPUT_DIR/models"
  mkdir -p "$MODELS_DIR"
  
  # Extract model information from job data
  MODEL_INPUTS=$(echo $JOB_DATA | jq -r '.inputs | keys[] | select(contains("model"))')
  
  for MODEL_INPUT in $MODEL_INPUTS; do
    MODEL_REF=$(echo $JOB_DATA | jq -r ".inputs.$MODEL_INPUT.assetId")
    if [ -n "$MODEL_REF" ] && [ "$MODEL_REF" != "null" ]; then
      MODEL_NAME=$(echo $MODEL_REF | cut -d':' -f2)
      MODEL_VERSION=$(echo $MODEL_REF | cut -d':' -f3)
      
      if [ -n "$MODEL_NAME" ] && [ -n "$MODEL_VERSION" ]; then
        echo "Downloading model: $MODEL_NAME:$MODEL_VERSION"
        MODEL_DIR="$MODELS_DIR/${MODEL_NAME}_v${MODEL_VERSION}"
        mkdir -p "$MODEL_DIR"
        
        az ml model download --name "$MODEL_NAME" --version "$MODEL_VERSION" --download-path "$MODEL_DIR"
        
        if [ $? -eq 0 ]; then
          echo "Successfully downloaded model to: $MODEL_DIR"
        else
          echo "Error downloading model: $MODEL_NAME:$MODEL_VERSION"
        fi
      fi
    fi
  done
fi

echo "========================================"
echo "Download Summary"
echo "========================================"
echo "Job:             $JOB_NAME"
echo "Status:          $JOB_STATUS"
echo "Output Location: $OUTPUT_DIR"

# Count files downloaded
NUM_FILES=$(find "$OUTPUT_DIR" -type f | wc -l)
echo "Files Downloaded: $NUM_FILES"

# Look for annotation files specifically
NUM_ANNOTATIONS=$(find "$OUTPUT_DIR" -name "*annotations.json" | wc -l)
echo "Annotation Files: $NUM_ANNOTATIONS"

echo ""
echo "To analyze the results, check the files in: $OUTPUT_DIR"
echo "You can use these results for testing the component locally."
echo "========================================"