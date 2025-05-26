# Azure ML Job Submission Fixes

This document explains the changes made to fix issues with Azure ML job submission via GitHub Actions.

## Problem

The GitHub Action for submitting AzureML jobs was failing with the following error:

```
The yaml file you provided does not match the prescribed schema for CommandJob yaml files and/or has the following issues:

Error: 
1) A least one unrecognized parameter is specified
2) At least one required parameter is missing

Details: Validation for CommandSchema failed

(x) command:
- Missing data for required field.

(x) environment:
- Missing data for required field.

(x) jobs:
- Unknown field.

(x) settings:
- Unknown field.
```

## Root Cause

The issue stemmed from a mismatch between the YAML file structure and how the Azure ML CLI was interpreting it:

1. The `azureml-job.yml` file was structured as a Pipeline Job but did not explicitly declare its type
2. The Azure ML CLI was attempting to validate it as a Command Job due to missing the explicit type
3. There was an inconsistency in job configuration between `azureml-job.yml` and the working `pipeline.yml`
4. Parameter naming was inconsistent with what the GitHub workflow was providing

## Changes Made

The following changes were implemented to fix these issues:

1. Added `type: pipeline` at the top level of `azureml-job.yml` to explicitly declare it as a Pipeline Job
2. Changed the job type from `parallel` to `command` to match the working structure in `pipeline.yml`
3. Renamed the input parameter from `model` to `main_model` for consistency with `pipeline.yml`
4. Updated GitHub workflow to reference the renamed parameter (`inputs.main_model` instead of `inputs.model`)

## Why These Changes Work

1. By explicitly specifying `type: pipeline`, we inform the Azure ML CLI that it should validate the file against the Pipeline Job schema, not the Command Job schema
2. Using the same job type as in the working `pipeline.yml` ensures consistent behavior
3. Consistent parameter naming prevents mismatches between what the Azure ML job expects and what the workflow provides

## Additional Component Type Fix

A related issue was encountered when submitting jobs using the `azureml-job.yml` file, with the following error:

```
Validation for PipelineJobSchema failed:

{
 "result": "Failed",
 "errors": [
   {
     "message": "Value 'parallel' passed is not in set ['command']",
     "path": "jobs.facade_inference.component.type",
     "value": "parallel",
     "location": "/home/runner/work/testbuild/testbuild/component.yml#line 3"
   }
 ]
} 
```

### Root Cause

The component.yml file was defined as a parallel component (`type: parallel`) but both pipeline.yml and azureml-job.yml were trying to use it as a command component (`type: command`). Azure ML now requires these types to match explicitly.

### Changes Made

The following changes were implemented to fix this issue:

1. Updated component.yml schema to use commandComponent schema instead of parallelComponent schema
2. Changed component type from `parallel` to `command`
3. Removed parallel-specific parameters (mini_batch_size, mini_batch_error_threshold, etc.)
4. Updated the structure to match the command component format

### Why These Changes Work

By making component.yml a proper command component, it now matches the job type expected in both pipeline.yml and azureml-job.yml. This ensures validation passes and the job can be submitted successfully.

### Future Considerations

When working with Azure ML components:

1. Ensure that the component type matches how it's used in job definitions
2. Consider maintaining separate component definitions if different job types are needed
3. Be aware that Azure ML validation is becoming stricter about schema compliance

## Future Considerations

When making changes to Azure ML job definitions:

1. Always specify the job type explicitly
2. Maintain consistent parameter naming across all files
3. Test job submissions using the Azure ML CLI validation feature before committing changes
4. Compare any new job definitions with existing working examples in the repository