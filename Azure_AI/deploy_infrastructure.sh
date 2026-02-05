#!/bin/bash

##############################################################################
# Azure Infrastructure Deployment Script
# This script creates all required Azure resources for the workflow
##############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

##############################################################################
# CONFIGURATION - Update these values
##############################################################################

RESOURCE_GROUP="databricks-error-recovery-rg"
LOCATION="eastus"
SUBSCRIPTION_ID="your-subscription-id"  # Update this

# Resource names
OPENAI_NAME="error-recovery-openai"
SEARCH_NAME="error-recovery-search"
DOC_INTEL_NAME="error-recovery-docai"
COMM_SERVICE_NAME="error-recovery-comm"
APP_INSIGHTS_NAME="error-recovery-insights"
STORAGE_ACCOUNT_NAME="errorrecoverystorage"  # Must be lowercase, no hyphens
CONTAINER_REGISTRY_NAME="errorrecoveryacr"  # Must be lowercase, no hyphens
CONTAINER_ENV_NAME="error-recovery-env"
CONTAINER_APP_NAME="databricks-error-recovery"
FUNCTION_APP_NAME="approval-webhook-func"

##############################################################################
# Step 1: Login and Set Subscription
##############################################################################

print_message "Step 1: Azure Login and Subscription Setup"
az login
az account set --subscription "$SUBSCRIPTION_ID"
print_message "✓ Logged in and subscription set"

##############################################################################
# Step 2: Create Resource Group
##############################################################################

print_message "Step 2: Creating Resource Group"
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION"
print_message "✓ Resource group created: $RESOURCE_GROUP"

##############################################################################
# Step 3: Deploy Azure OpenAI
##############################################################################

print_message "Step 3: Deploying Azure OpenAI"
az cognitiveservices account create \
  --name "$OPENAI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --kind OpenAI \
  --sku S0 \
  --location "$LOCATION" \
  --yes

print_message "Creating GPT-4o deployment..."
az cognitiveservices account deployment create \
  --name "$OPENAI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --deployment-name "gpt-4o" \
  --model-name "gpt-4" \
  --model-version "turbo-2024-04-09" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name "Standard"

print_message "Creating text-embedding-ada-002 deployment..."
az cognitiveservices account deployment create \
  --name "$OPENAI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --deployment-name "text-embedding-ada-002" \
  --model-name "text-embedding-ada-002" \
  --model-version "2" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name "Standard"

print_message "✓ Azure OpenAI deployed"

##############################################################################
# Step 4: Deploy Azure AI Search
##############################################################################

print_message "Step 4: Deploying Azure AI Search"
az search service create \
  --name "$SEARCH_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --sku standard \
  --partition-count 1 \
  --replica-count 1 \
  --location "$LOCATION"

print_message "✓ Azure AI Search deployed"

##############################################################################
# Step 5: Deploy Azure Document Intelligence
##############################################################################

print_message "Step 5: Deploying Azure Document Intelligence"
az cognitiveservices account create \
  --name "$DOC_INTEL_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --kind FormRecognizer \
  --sku S0 \
  --location "$LOCATION" \
  --yes

print_message "✓ Azure Document Intelligence deployed"

##############################################################################
# Step 6: Deploy Azure Communication Services
##############################################################################

print_message "Step 6: Deploying Azure Communication Services"
az communication create \
  --name "$COMM_SERVICE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location global \
  --data-location UnitedStates

print_warning "⚠ You need to manually configure email domain in Azure Portal"
print_warning "   Navigate to: Communication Services → Email → Domains"

print_message "✓ Azure Communication Services deployed"

##############################################################################
# Step 7: Deploy Application Insights
##############################################################################

print_message "Step 7: Deploying Application Insights"
az monitor app-insights component create \
  --app "$APP_INSIGHTS_NAME" \
  --location "$LOCATION" \
  --resource-group "$RESOURCE_GROUP" \
  --application-type web

print_message "✓ Application Insights deployed"

##############################################################################
# Step 8: Create Storage Account
##############################################################################

print_message "Step 8: Creating Storage Account"
az storage account create \
  --name "$STORAGE_ACCOUNT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS

# Get storage connection string
STORAGE_CONN_STRING=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString -o tsv)

# Create table for approval state
az storage table create \
  --name ApprovalRequests \
  --connection-string "$STORAGE_CONN_STRING"

print_message "✓ Storage Account and Table created"

##############################################################################
# Step 9: Create Container Registry
##############################################################################

print_message "Step 9: Creating Azure Container Registry"
az acr create \
  --name "$CONTAINER_REGISTRY_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --sku Standard \
  --location "$LOCATION" \
  --admin-enabled true

print_message "✓ Azure Container Registry created"

##############################################################################
# Step 10: Create Container Apps Environment
##############################################################################

print_message "Step 10: Creating Container Apps Environment"
az containerapp env create \
  --name "$CONTAINER_ENV_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

print_message "✓ Container Apps Environment created"

##############################################################################
# Step 11: Create Function App for Webhook
##############################################################################

print_message "Step 11: Creating Function App for Approval Webhook"
az functionapp create \
  --resource-group "$RESOURCE_GROUP" \
  --consumption-plan-location "$LOCATION" \
  --runtime python \
  --runtime-version 3.10 \
  --functions-version 4 \
  --name "$FUNCTION_APP_NAME" \
  --storage-account "$STORAGE_ACCOUNT_NAME" \
  --os-type Linux

# Configure Function App settings
az functionapp config appsettings set \
  --name "$FUNCTION_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings "AZURE_STORAGE_CONNECTION_STRING=$STORAGE_CONN_STRING"

print_message "✓ Function App created"

##############################################################################
# Step 12: Retrieve Connection Strings and Keys
##############################################################################

print_message "Step 12: Retrieving connection strings and keys..."

# Azure OpenAI
OPENAI_ENDPOINT=$(az cognitiveservices account show \
  --name "$OPENAI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.endpoint -o tsv)

OPENAI_KEY=$(az cognitiveservices account keys list \
  --name "$OPENAI_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query key1 -o tsv)

# Azure AI Search
SEARCH_ENDPOINT="https://${SEARCH_NAME}.search.windows.net"
SEARCH_KEY=$(az search admin-key show \
  --service-name "$SEARCH_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query primaryKey -o tsv)

# Document Intelligence
DOC_INTEL_ENDPOINT=$(az cognitiveservices account show \
  --name "$DOC_INTEL_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.endpoint -o tsv)

DOC_INTEL_KEY=$(az cognitiveservices account keys list \
  --name "$DOC_INTEL_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query key1 -o tsv)

# Communication Services
COMM_CONN_STRING=$(az communication list-key \
  --name "$COMM_SERVICE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query primaryConnectionString -o tsv)

# Application Insights
APP_INSIGHTS_CONN_STRING=$(az monitor app-insights component show \
  --app "$APP_INSIGHTS_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString -o tsv)

# Function App URL
FUNCTION_APP_URL=$(az functionapp show \
  --name "$FUNCTION_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query defaultHostName -o tsv)

##############################################################################
# Step 13: Generate .env file
##############################################################################

print_message "Step 13: Generating .env file with all credentials..."

cat > .env << EOF
# Azure AI Foundry Configuration
# NOTE: You need to create this manually in Azure AI Foundry Portal
AZURE_AI_PROJECT_CONNECTION_STRING=your-project-connection-string-here

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=${OPENAI_ENDPOINT}
AZURE_OPENAI_KEY=${OPENAI_KEY}

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT=${SEARCH_ENDPOINT}
AZURE_SEARCH_KEY=${SEARCH_KEY}

# Azure Document Intelligence Configuration
AZURE_DOC_INTELLIGENCE_ENDPOINT=${DOC_INTEL_ENDPOINT}
AZURE_DOC_INTELLIGENCE_KEY=${DOC_INTEL_KEY}

# Databricks Configuration (UPDATE THESE MANUALLY)
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-databricks-personal-access-token

# LangSmith Configuration (Optional - for advanced tracing)
LANGSMITH_API_KEY=your-langsmith-api-key-here

# Azure Communication Services (for email notifications)
AZURE_COMMUNICATION_CONNECTION_STRING=${COMM_CONN_STRING}
SENDER_EMAIL=noreply@your-domain.com  # UPDATE THIS
ENGINEERING_TEAM_EMAIL=engineering-team@your-domain.com  # UPDATE THIS

# Azure Application Insights (for observability)
APPLICATIONINSIGHTS_CONNECTION_STRING=${APP_INSIGHTS_CONN_STRING}

# Azure Storage (for approval state)
AZURE_STORAGE_CONNECTION_STRING=${STORAGE_CONN_STRING}

# Function App Webhook URL
APPROVAL_WEBHOOK_URL=https://${FUNCTION_APP_URL}/api/approval
EOF

print_message "✓ .env file generated"

##############################################################################
# Summary
##############################################################################

echo ""
echo "=========================================="
echo "  ✓ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "Resources created in resource group: $RESOURCE_GROUP"
echo ""
echo "Next steps:"
echo "1. Create Azure AI Foundry Project manually:"
echo "   → Visit: https://ai.azure.com"
echo "   → Create project and copy connection string to .env"
echo ""
echo "2. Configure email domain in Communication Services:"
echo "   → Azure Portal → Communication Services → Email → Domains"
echo ""
echo "3. Update Databricks credentials in .env file"
echo ""
echo "4. (Optional) Sign up for LangSmith and add API key to .env"
echo ""
echo "5. Review and update sender/recipient emails in .env"
echo ""
echo "6. Build and deploy the container:"
echo "   → docker build -t databricks-error-recovery:latest ."
echo "   → az acr login --name $CONTAINER_REGISTRY_NAME"
echo "   → docker tag databricks-error-recovery:latest $CONTAINER_REGISTRY_NAME.azurecr.io/databricks-error-recovery:latest"
echo "   → docker push $CONTAINER_REGISTRY_NAME.azurecr.io/databricks-error-recovery:latest"
echo ""
echo "7. Deploy container to Container Apps (see DEPLOYMENT_GUIDE.md)"
echo ""
echo "All credentials have been saved to .env file"
echo "=========================================="
