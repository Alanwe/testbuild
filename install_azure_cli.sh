#!/bin/bash
# =============================================================================
# install_azure_cli.sh - Installs Azure CLI and ML extension
# =============================================================================
# This script installs the Azure CLI and ML extension required for
# interfacing with Azure ML workspaces and deploying components.
# =============================================================================
set -e

echo "Installing Azure CLI and ML extension..."

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if Azure CLI is already installed
if command_exists az; then
  echo "✅ Azure CLI is already installed."
  echo "Current version: $(az --version | grep "azure-cli" | head -n 1)"
else
  echo "Installing Azure CLI..."
  
  # Check if curl is installed
  if ! command_exists curl; then
    echo "Installing curl..."
    sudo apt-get update
    sudo apt-get install -y curl
  fi
  
  # Install Azure CLI
  curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
  
  # Verify installation
  if ! command_exists az; then
    echo "❌ Azure CLI installation failed."
    exit 1
  fi
  
  echo "✅ Azure CLI installed successfully."
  echo "Version: $(az --version | grep "azure-cli" | head -n 1)"
fi

# Check if ML extension is already installed
if az extension list --query "[?name=='ml'].version" -o tsv | grep -q "."; then
  echo "✅ Azure ML extension is already installed."
  echo "Version: $(az extension list --query "[?name=='ml'].version" -o tsv)"
else
  echo "Installing Azure ML extension..."
  
  # Install ML extension
  az extension add --name ml
  
  # Verify installation
  if ! az extension list --query "[?name=='ml'].version" -o tsv | grep -q "."; then
    echo "❌ Azure ML extension installation failed."
    exit 1
  fi
  
  echo "✅ Azure ML extension installed successfully."
  echo "Version: $(az extension list --query "[?name=='ml'].version" -o tsv)"
fi

echo "Installation complete! You can now use Azure CLI with ML extension."
echo ""
echo "To verify your installation, run:"
echo "  az ml --help"
echo ""
echo "To log in to Azure:"
echo "  az login"
echo ""
echo "For more information, see SETUP.md"