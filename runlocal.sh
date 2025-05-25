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
mkdir -p "$OUTPUT_DIR/trace"

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

# Check if model exists
if [ ! -d "$MODEL1" ]; then
  echo "ERROR: Model directory not found at $MODEL1"
  exit 1
fi

# Verify required model files
if [ ! -f "$MODEL1/MLmodel" ] || [ ! -f "$MODEL1/python_model.pkl" ]; then
  echo "ERROR: Required model files not found in $MODEL1"
  echo "Make sure MLmodel and python_model.pkl exist."
  exit 1
fi

echo "Model directory validated successfully"

# Check for sample images
IMAGES_COUNT=$(find $INPUT_DS -type f -name "*.jpg" | wc -l)
if [ "$IMAGES_COUNT" -eq 0 ]; then
  echo "WARNING: No images found in $INPUT_DS"
  echo "Creating sample test directory structure..."
  mkdir -p $INPUT_DS/$BATCH_ID/cam
  
  # Create a simple test image if none exists
  python -c "
import numpy as np
from PIL import Image
import os
# Create a simple test image
img = np.ones((512, 512, 3), dtype=np.uint8) * 200
img[100:400, 100:400] = [150, 150, 150]
img[150:200, 250:350] = [100, 100, 100]
os.makedirs('$INPUT_DS/$BATCH_ID/cam', exist_ok=True)
Image.fromarray(img).save('$INPUT_DS/$BATCH_ID/cam/test_image.jpg')
" || (
  echo "Error creating test image with Python. Using alternative method."
  # Create a blank file if Python fails
  mkdir -p $INPUT_DS/$BATCH_ID/cam
  touch $INPUT_DS/$BATCH_ID/cam/test_image.jpg
)
  echo "Created sample test image at $INPUT_DS/$BATCH_ID/cam/test_image.jpg"
fi

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

# Check if any results were generated
if [ -f "$OUTPUT_DIR/results.json" ]; then
  echo "Results generated successfully in $OUTPUT_DIR/results.json"
else
  echo "Warning: No results.json file generated"
fi

# Check for trace files
TRACE_FILES_COUNT=$(find $OUTPUT_DIR/trace -type f | wc -l)
if [ "$TRACE_FILES_COUNT" -gt 0 ]; then
  echo "Trace files generated: $TRACE_FILES_COUNT"
else
  echo "Warning: No trace files generated"
fi

echo "Process completed successfully."
