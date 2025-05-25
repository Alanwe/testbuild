# GitHub Actions Workflows

This document provides information about the GitHub Actions workflows used in this repository.

## Authentication with Azure

The workflows in this repository use Azure service principal authentication to connect to Azure services.

### Required GitHub Secrets

The following secrets need to be set up in your GitHub repository:

- `AZURE_CLIENT_ID`: The client/app ID of your Azure service principal
- `AZURE_CLIENT_SECRET`: The client secret of your Azure service principal
- `AZURE_TENANT_ID`: Your Azure tenant ID
- `AZURE_SUBSCRIPTION_ID`: Your Azure subscription ID

### Authentication Methods

The repository uses two different methods for authentication:

1. **azure/login Action**: Used in the `submit-azureml-job.yml` workflow. This workflow authenticates using the azure/login@v1 action with service principal credentials:

   ```yaml
   - name: Login to Azure
     uses: azure/login@v1
     with:
       creds: '{"clientId":"${{ secrets.AZURE_CLIENT_ID }}", "clientSecret":"${{ secrets.AZURE_CLIENT_SECRET }}", "subscriptionId":"${{ secrets.AZURE_SUBSCRIPTION_ID }}", "tenantId":"${{ secrets.AZURE_TENANT_ID }}"}'
       enable-AzPSSession: false
       environment: azurecloud
       allow-no-subscriptions: false
       audience: api://AzureADTokenExchange
   ```

2. **Environment Variables**: Used in the `run-aml-pipeline.yml` workflow. This workflow passes the credentials as environment variables to the Python script:

   ```yaml
   env:
     AZURE_SUBSCRIPTION_ID: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
     AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
     AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
     AZURE_CLIENT_SECRET: ${{ secrets.AZURE_CLIENT_SECRET }}
   ```

### Setting Up Service Principal

To create a service principal that can be used with these workflows:

1. Create a service principal:
   ```bash
   az ad sp create-for-rbac --name "FacadeAI-GitHub-SP" --role contributor \
     --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group> \
     --sdk-auth
   ```

2. The output will include the clientId, clientSecret, tenantId, and subscriptionId needed for the GitHub secrets.

3. Add these values as secrets in your GitHub repository:
   - Go to repository Settings > Secrets > Actions
   - Add each secret separately