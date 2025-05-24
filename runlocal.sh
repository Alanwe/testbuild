#!/bin/bash
set -e

echo "Running FacadeAI Inference Component locally..."

# Default parameters
INPUT_DS="data/images"
OUTPUT_DIR="data/output"
BATCH_ID="B3"
MODE="force"
WINDOW_SIZE=512
OVERLAP=64
CONFIDENCE=50
MODEL1="models/Glazing-Defects"
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
