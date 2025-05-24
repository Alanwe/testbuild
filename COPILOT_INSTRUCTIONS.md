# GitHub Copilot Instructions for FacadeAI Development

This document provides specific guidance for GitHub Copilot when working with the FacadeAI codebase.

## Setting Up the Environment

To work with FacadeAI in your development environment, follow these steps:

1. Set up the development environment using one of the provided scripts:

```bash
# Using conda (recommended)
./setup_dev.sh

# Or using Python venv (no conda)
./setup_venv.sh
```

2. Activate the environment:

```bash
# If using conda
conda activate facadeai_dev

# If using venv
source .venv/bin/activate
```

3. Verify the environment is correctly set up:

```bash
./verify_env.py
```

4. Run the component in development mode:

```bash
./dev_runlocal.sh
```

## Code Structure

- `component.py`: Main inference code
- `utils.py`: Utility functions used by the main component
- `dev_runlocal.sh`: Script to run the component
- `good.py`: Reference implementation for model loading and prediction

## Working with Models

The development environment uses a simple dummy model for testing. When fixing issues or making changes:

1. Reference the correct model path: `models/Dev-Model`
2. Use `mlflow.pyfunc.load_model()` to load the model
3. The model should return a dictionary with at least the keys:
   - `segmentation`: List of coordinates defining a polygon
   - `confidence`: Confidence score for the detection

## Debugging Tips

For common errors:

1. **Module import errors**: Check if you've activated the correct environment.

2. **Model loading errors**: Verify the model structure with:
   ```
   ls -la models/Dev-Model/
   ```

3. **Image processing errors**: Debug with:
   ```python
   # Add code to save intermediate images for inspection
   import cv2
   cv2.imwrite("debug_image.jpg", image)
   ```

4. **Azure connection errors**: These are expected in local mode. The code should fall back to local mode automatically.

## Testing Your Changes

After making changes to the code:

1. Run the verification script first:
   ```
   ./verify_env.py
   ```

2. Run the inference script:
   ```
   ./dev_runlocal.sh
   ```

3. Check the output in the `data/output` directory.

## Best Practices

1. Maintain CPU compatibility by avoiding GPU-specific code.
2. Use appropriate error handling and logging.
3. Keep dependencies minimal when possible.
4. Use the conditional import pattern for optional dependencies:
   ```python
   try:
       import some_optional_package
       HAS_OPTIONAL = True
   except ImportError:
       HAS_OPTIONAL = False
   ```