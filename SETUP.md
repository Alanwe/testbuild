# Azure ML Setup for FacadeAI

This guide explains how to set up the FacadeAI component to work with Azure Machine Learning (AzureML). Follow these steps to connect your development environment to an AzureML workspace and prepare for submitting pipeline jobs.

## Prerequisites

- Azure account with an active subscription
- Access to an AzureML workspace
- Docker (required for local testing before deployment)
- Python 3.9+ with pip
- PowerShell or Bash shell

## Installing Azure CLI

1. Run the provided installation script:

```bash
chmod +x install_azure_cli.sh
./install_azure_cli.sh
```

2. Verify the installation:

```bash
az --version
az ml --help
```

3. Log in to Azure:

```bash
az login
```

For non-interactive environments, use a service principal:

```bash
az login --service-principal -u <client-id> -p <client-secret> --tenant <tenant-id>
```

## Connecting to Azure ML Workspace

1. Set your Azure ML workspace details:

```bash
# Set default subscription
az account set -s "<SUBSCRIPTION_ID>"

# Set default group and workspace
az configure --defaults group="<RESOURCE_GROUP>" workspace="<WORKSPACE_NAME>"
```

2. Verify connection to the workspace:

```bash
az ml workspace show
```

3. Create a config file for programmatic access:

```bash
mkdir -p .azureml
cat > .azureml/config.json << EOF
{
    "subscription_id": "<SUBSCRIPTION_ID>",
    "resource_group": "<RESOURCE_GROUP>",
    "workspace_name": "<WORKSPACE_NAME>"
}
EOF
```

## Data Preparation

1. Create a dataset in Azure ML:

```bash
# Create dataset directory
mkdir -p data/aml_upload

# Copy your images to the upload directory
cp -r data/images/<BATCH_ID>/cam/* data/aml_upload/

# Create Azure ML data asset
az ml data create --name facade-images --version 1 --path data/aml_upload --type uri_folder
```

2. Verify the dataset was created:

```bash
az ml data show --name facade-images --version 1
```

## Model Registration

1. Prepare your MLflow model:

Ensure your model directory has the required MLflow structure:
- MLmodel file
- conda.yaml file
- model.py or python_model.pkl file

2. Register the model with Azure ML:

```bash
az ml model create --name Glazing-Defects --version 1 --path models/Glazing-Defects --type mlflow_model
```

3. Verify the model was registered:

```bash
az ml model show --name Glazing-Defects --version 1
```

## Setting Up Credentials

### Azure Key Vault Integration

If your component needs to access secrets like database connection strings:

1. Create a Key Vault if you don't have one:

```bash
az keyvault create --name "<KEYVAULT_NAME>" --resource-group "<RESOURCE_GROUP>"
```

2. Add your secrets:

```bash
az keyvault secret set --vault-name "<KEYVAULT_NAME>" --name "cosmos-connection-string" --value "<CONNECTION_STRING>"
```

3. Grant the AzureML workspace's managed identity access to Key Vault:

```bash
# Get workspace managed identity object ID
WORKSPACE_MSI=$(az ml workspace show --query identity.principalId -o tsv)

# Grant permissions
az keyvault set-policy --name "<KEYVAULT_NAME>" --object-id $WORKSPACE_MSI --secret-permissions get list
```

## Testing Your Setup

1. Run a quick verification job to ensure everything is set up correctly:

```bash
# Test access to the workspace, data, and models
az ml job create -f tests/verify_setup.yml
```

2. Check the job status:

```bash
az ml job show -n <JOB_NAME>
```

## Troubleshooting

### Common Issues

1. **Authentication failures**:
   - Ensure you're logged in: `az account show`
   - Check your account has proper permissions to the workspace

2. **Model registration failures**:
   - Verify your model follows MLflow structure
   - Run `mlflow.pyfunc.load_model("models/Your-Model")` locally to validate

3. **Data access issues**:
   - Make sure datasets are properly registered and accessible
   - Check that your job has permissions to access the data

4. **Azure Key Vault access denied**:
   - Verify the managed identity has the right permissions
   - Check if your secrets exist: `az keyvault secret list --vault-name "<KEYVAULT_NAME>"`

### Getting Help

Run `az ml -h` for general help with Azure ML CLI commands.

For more information, see the [Azure ML documentation](https://docs.microsoft.com/en-us/azure/machine-learning/).

## Next Steps

After setting up your environment, you can:

1. Submit a pipeline job using `submit_job.sh`
2. Collect outputs from previous jobs using `collect_outputs.sh`
3. View pipeline.yml for examples of how to customize your own pipelines