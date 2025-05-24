# Running the FacadeAI Inference Component

This document provides detailed instructions for running the FacadeAI Inference component with various configurations and parameters.

## Basic Usage

The component can be run locally using the provided `runlocal.sh` script, which sets up default parameters:

```bash
./runlocal.sh
```

## Understanding Parameters

The component accepts various parameters to control its behavior. Here's a detailed explanation of each parameter:

### Input/Output Parameters

| Parameter    | Description                          | Default Value | Notes |
|--------------|--------------------------------------|---------------|-------|
| `INPUT_DS`   | Input dataset directory              | data/images   | Should contain batch folders with camera images |
| `OUTPUT_DIR` | Directory for output results         | data/output   | Will be created if it doesn't exist |
| `BATCH_ID`   | Batch identifier                     | B3            | Used to organize images and for CosmosDB lookups |

### Processing Configuration

| Parameter      | Description                          | Default Value | Notes |
|----------------|--------------------------------------|---------------|-------|
| `MODE`         | Processing mode                      | force         | 'force' processes all images, 'auto' skips already processed ones |
| `WINDOW_SIZE`  | Size of sliding window               | 512           | In pixels, should match model's expected input size |
| `OVERLAP`      | Overlap between windows              | 64            | In pixels, higher values provide better boundary detection |
| `CONFIDENCE`   | Minimum confidence threshold         | 50            | Percentage (0-100), detections below this are filtered out |

### Model Parameters

| Parameter | Description                 | Default Value        | Notes |
|-----------|-----------------------------|----------------------|-------|
| `MODEL1`  | Path to the primary model   | models/Glazing-Defects | MLflow model directory |
| `MODEL2`  | Path to the secondary model | (empty)              | Optional additional model |

### Execution Mode

| Parameter | Description                    | Default Value | Notes |
|-----------|--------------------------------|---------------|-------|
| `LOCAL`   | Run in local mode              | true          | 'true' skips CosmosDB operations |
| `TRACE`   | Enable trace image generation  | true          | 'true' saves visualization images |

## Advanced Configuration

### Custom Parameter Values

To run with custom parameters, you can modify the runlocal.sh script or override parameters when running:

```bash
# Edit runlocal.sh first to change parameters
nano runlocal.sh

# Or override parameters directly (Linux/macOS)
INPUT_DS=/path/to/custom/images OUTPUT_DIR=/path/to/results ./runlocal.sh

# For Windows PowerShell
$env:INPUT_DS = "C:\path\to\custom\images"
$env:OUTPUT_DIR = "C:\path\to\results"
.\runlocal.sh
```

### Multiple Models

The component supports running multiple models on the same images. To use multiple models:

1. Edit runlocal.sh to specify additional models:

```bash
MODEL1="models/Glazing-Defects"
MODEL2="models/Stone-Fractures"
MODEL3="models/Window-Defects"
```

2. Each model can have its own confidence threshold:

```bash
# Add to runlocal.sh
MODEL1_CONFIDENCE=50
MODEL2_CONFIDENCE=40
MODEL3_CONFIDENCE=60
```

3. Update the component.py call to include these parameters:

```bash
python component.py \
  --model1 $MODEL1 \
  --model1_confidence $MODEL1_CONFIDENCE \
  --model2 $MODEL2 \
  --model2_confidence $MODEL2_CONFIDENCE \
  --model3 $MODEL3 \
  --model3_confidence $MODEL3_CONFIDENCE \
  # ...other parameters
```

### Expected Data Structure

The component expects input data in a specific structure:

```
data/
  images/
    B3/           # Batch directory (matches BATCH_ID)
      cam/        # Camera images folder
        image1.jpg
        image2.jpg
        ...
```

Output is generated in the following structure:

```
data/
  output/
    image1_annotations.json  # Per-image annotation files
    image2_annotations.json
    ...
    results.json             # Summary of all processed images
    trace/                   # Optional trace directory when TRACE=true
      image1_visualization.jpg
      ...
```

## Running on Different Image Sets

To run the component on different sets of images:

1. Organize your images in the expected directory structure
2. Update the BATCH_ID parameter to match your batch folder name
3. Run the script with modified parameters:

```bash
# Example for a new batch of images in data/images/B5/cam
BATCH_ID=B5 ./runlocal.sh
```

## Output Files and Formats

### Annotation JSON Format

Each image produces an annotation JSON file with the following structure:

```json
[
  {
    "id": 1,
    "image_id": "image_filename",
    "category_id": 3,
    "segmentation": [[x1, y1, x2, y2, ...]], 
    "area": 12345.6,
    "bbox": [x, y, width, height],
    "iscrowd": 0,
    "objectId": "1621234567890"
  },
  // More detections...
]
```

### Results JSON Format

The overall results.json file contains a summary:

```json
[
  "image1: 5 annotations",
  "image2: 3 annotations",
  // More results...
]
```

## Performance Considerations

### Memory Usage

The sliding window technique allows processing large images with limited memory, but you may still need to adjust parameters:

- For high-resolution images, increase WINDOW_SIZE to process larger areas at once
- For memory-constrained environments, decrease WINDOW_SIZE
- Balance OVERLAP value - higher values provide better detection at boundaries but increase processing time

### Processing Time

To optimize processing time:

- Use GPU acceleration when available (enable in Docker or use GPU-enabled compute in Azure ML)
- Adjust WINDOW_SIZE and OVERLAP for a balance between accuracy and speed
- For batch processing, consider using Azure ML to parallelize across multiple compute nodes

## Common Execution Issues

### Script Not Executable

If you see "Permission denied" when running runlocal.sh:

```bash
chmod +x runlocal.sh
```

### Input Directory Not Found

If the script can't find your images:

1. Check the directory structure matches what the component expects
2. Verify the INPUT_DS parameter points to the correct parent directory
3. Ensure BATCH_ID matches your batch folder name

### Model Loading Failures

If models fail to load:

1. Verify the model directory exists and has the correct MLflow structure
2. Check Python and package versions match what the model requires
3. Try running with a simplified or dummy model for testing

## Monitoring and Logging

The component logs detailed information about its operation:

- Standard output shows progress and summary information
- Detailed logs show model loading, image processing, and detection information
- Set TRACE=true to save visualization images that show detected defects

To capture logs for analysis:

```bash
./runlocal.sh > run_log.txt 2>&1
```

## Running on a Subset of Images

To test on a smaller set of images:

1. Create a test directory with a few images
2. Update the INPUT_DS parameter to point to this directory
3. Run with the adjusted parameters:

```bash
INPUT_DS=data/test_images ./runlocal.sh
```