#!/bin/bash
# =============================================================================
# dev_runlocal.sh - Script to run FacadeAI in development mode
# =============================================================================
# This script runs the FacadeAI inference component with settings appropriate
# for local development, using the Dev-Model by default and enabling tracing.
# =============================================================================
set -e

echo "Running FacadeAI Inference Component locally in development mode..."

# Default parameters
INPUT_DS="data/images"
OUTPUT_DIR="data/output"
BATCH_ID="B3"
MODE="force"
WINDOW_SIZE=512
OVERLAP=64
CONFIDENCE=50
MODEL1="models/Dev-Model"  # Use the development model by default
MODEL2=""
LOCAL=true
TRACE=true

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Display parameters
echo "Input dataset: $INPUT_DS"
echo "Output directory: $OUTPUT_DIR"
echo "Batch ID: $BATCH_ID"
echo "Mode: $MODE"
echo "Window size: $WINDOW_SIZE"
echo "Overlap: $OVERLAP"
echo "Confidence threshold: $CONFIDENCE"
echo "Local mode: $LOCAL"
echo "Trace enabled: $TRACE"
echo "Using model: $MODEL1"

# Run the component
python component.py \
  --input_ds $INPUT_DS \
  --output_data $OUTPUT_DIR \
  --batch_id $BATCH_ID \
  --mode $MODE \
  --window_size $WINDOW_SIZE \
  --overlap $OVERLAP \
  --confidence $CONFIDENCE \
  --model1 $MODEL1 \
  --local $LOCAL \
  --trace $TRACE

echo "Process completed successfully."