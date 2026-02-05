# Deployment Guide: Databricks Error Recovery Agentic Workflow

## Overview
This guide explains **WHERE** and **HOW** to deploy each component of the agentic workflow.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT TOPOLOGY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐          ┌──────────────────┐             │
│  │ Databricks      │──Trigger→│ Event Grid       │             │
│  │ (Job Failed)    │          │ (Event Hub)      │             │
│  └─────────────────┘          └────────┬─────────┘             │
│                                         │                        │
│                                         ▼                        │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Azure Container Apps / Azure Functions              │      │
│  │  (Main Workflow: databricks_error_recovery_workflow.py) │   │
│  │                                                        │      │
│  │  Components:                                          │      │
│  │  • Log Retrieval Agent                                │      │
│  │  • RAG Knowledge Base (Azure AI Search)               │      │
│  │  • IDP Visual Agent (Document Intelligence)           │      │
│  │  • Azure AI Foundry Agent Service                     │      │
│  │  • TensorFlow Reliability Classifier                  │      │
│  │  • Human-in-the-Loop Orchestrator                     │      │
│  │  • Action Executor                                     │      │
│  └──────────────────────────────────────────────────────┘      │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Azure Functions (HTTP Trigger)                       │      │
│  │  (Approval Webhook: function_app.py)                  │      │
│  │                                                        │      │
│  │  • Handles email approval/rejection                   │      │
│  │  • Updates approval status in Table Storage           │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  Supporting Azure Services                            │      │
│  │                                                        │      │
│  │  • Azure OpenAI (GPT-4o)                              │      │
│  │  • Azure AI Search (Vector Store)                     │      │
│  │  • Azure Document Intelligence (IDP)                  │      │
│  │  • Azure Communication Services (Email)               │      │
│  │  • Azure Monitor + Application Insights               │      │
│  │  • Azure Table Storage (Approval State)               │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deployment Options

You have **TWO PRIMARY OPTIONS** for deploying the main workflow:

### **Option 1: Azure Container Apps (RECOMMENDED)**
✅ Best for: Long-running workflows, complex dependencies, full control
- Supports async workflows with long wait times
- Better for TensorFlow model hosting
- Full container isolation
- Easier debugging and logging

### **Option 2: Azure Functions (Consumption/Premium Plan)**
✅ Best for: Event-driven, serverless, cost-optimized scenarios
- Pay-per-execution model
- Auto-scaling
- May require Durable Functions for long-running approval waits
- Premium plan needed for longer execution times (60+ minutes)

---

## Step-by-Step Deployment

### **STEP 1: Prerequisites - Create Azure Resources**

#### 1.1 Create Azure AI Foundry Project
```bash
# Via Azure Portal
1. Navigate to Azure AI Foundry (preview.ai.azure.com)
2. Create a new Project
3. Note the Connection String (Settings → Connection String)
```

#### 1.2 Deploy Azure OpenAI
```bash
az cognitiveservices account create \
  --name your-openai-instance \
  --resource-group your-rg \
  --kind OpenAI \
  --sku S0 \
  --location eastus

# Deploy GPT-4o model
az cognitiveservices account deployment create \
  --name your-openai-instance \
  --resource-group your-rg \
  --deployment-name gpt-4o \
  --model-name gpt-4 \
  --model-version "turbo-2024-04-09" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name "Standard"

# Deploy text-embedding-ada-002
az cognitiveservices account deployment create \
  --name your-openai-instance \
  --resource-group your-rg \
  --deployment-name text-embedding-ada-002 \
  --model-name text-embedding-ada-002 \
  --model-version "2" \
  --model-format OpenAI \
  --sku-capacity 10 \
  --sku-name "Standard"
```

#### 1.3 Create Azure AI Search (with vector support)
```bash
az search service create \
  --name your-search-service \
  --resource-group your-rg \
  --sku standard \
  --partition-count 1 \
  --replica-count 1
```

#### 1.4 Deploy Azure Document Intelligence
```bash
az cognitiveservices account create \
  --name your-doc-intelligence \
  --resource-group your-rg \
  --kind FormRecognizer \
  --sku S0 \
  --location eastus
```

#### 1.5 Create Azure Communication Services (for email)
```bash
az communication create \
  --name your-communication-service \
  --resource-group your-rg \
  --location global \
  --data-location UnitedStates

# Create email domain
az communication email domain create \
  --name your-domain.com \
  --resource-group your-rg \
  --communication-service-name your-communication-service
```

#### 1.6 Create Azure Application Insights
```bash
az monitor app-insights component create \
  --app your-app-insights \
  --location eastus \
  --resource-group your-rg \
  --application-type web
```

#### 1.7 Create Azure Table Storage (for approval state)
```bash
az storage account create \
  --name yourstorageaccount \
  --resource-group your-rg \
  --location eastus \
  --sku Standard_LRS

# Create table
az storage table create \
  --name ApprovalRequests \
  --account-name yourstorageaccount
```

---

### **STEP 2: Configure Environment Variables**

1. Copy `.env.template` to `.env`
2. Fill in all the values from Step 1:

```bash
cp .env.template .env
# Edit .env with your actual values
```

---

### **STEP 3A: Deploy Main Workflow to Azure Container Apps**

#### 3A.1 Create Dockerfile
```dockerfile
# Create this as 'Dockerfile' in your project root
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY databricks_error_recovery_workflow.py .
COPY .env .

# Create models directory
RUN mkdir -p /app/models

# Run the workflow
CMD ["python", "databricks_error_recovery_workflow.py"]
```

#### 3A.2 Build and Push Docker Image
```bash
# Build image
docker build -t databricks-error-recovery:latest .

# Tag for Azure Container Registry
docker tag databricks-error-recovery:latest \
  yourregistry.azurecr.io/databricks-error-recovery:latest

# Login to ACR
az acr login --name yourregistry

# Push image
docker push yourregistry.azurecr.io/databricks-error-recovery:latest
```

#### 3A.3 Deploy to Azure Container Apps
```bash
# Create Container Apps environment
az containerapp env create \
  --name error-recovery-env \
  --resource-group your-rg \
  --location eastus

# Deploy container app
az containerapp create \
  --name databricks-error-recovery \
  --resource-group your-rg \
  --environment error-recovery-env \
  --image yourregistry.azurecr.io/databricks-error-recovery:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --env-vars \
    AZURE_AI_PROJECT_CONNECTION_STRING=secretref:ai-project-conn \
    AZURE_OPENAI_ENDPOINT=secretref:openai-endpoint \
    AZURE_OPENAI_KEY=secretref:openai-key \
    # ... add all other env vars as secrets
```

---

### **STEP 3B: Alternative - Deploy to Azure Functions**

#### 3B.1 Create Azure Functions Project
```bash
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Create Functions project
func init ErrorRecoveryFunctions --python
cd ErrorRecoveryFunctions

# Copy your code
cp ../databricks_error_recovery_workflow.py .
cp ../requirements.txt .
```

#### 3B.2 Modify for Durable Functions (for long-running approval)
```python
# Create: ErrorRecoveryFunctions/orchestrator/__init__.py
import azure.functions as func
import azure.durable_functions as df

def orchestrator_function(context: df.DurableOrchestrationContext):
    job_id = context.get_input()
    
    # Step 1: Retrieve logs
    log_data = yield context.call_activity("RetrieveLogs", job_id)
    
    # Step 2: Analyze error
    analysis = yield context.call_activity("AnalyzeError", log_data)
    
    # Step 3: Send approval request
    approval_id = yield context.call_activity("SendApproval", analysis)
    
    # Step 4: Wait for approval (can wait hours)
    approval_event = yield context.wait_for_external_event("ApprovalReceived")
    
    # Step 5: Execute action
    if approval_event == "approved":
        result = yield context.call_activity("RestartJob", job_id)
    else:
        result = yield context.call_activity("TerminateJob", job_id)
    
    return result

main = df.Orchestrator.create(orchestrator_function)
```

#### 3B.3 Deploy to Azure Functions
```bash
# Login to Azure
az login

# Create function app
az functionapp create \
  --resource-group your-rg \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.10 \
  --functions-version 4 \
  --name databricks-error-recovery-func \
  --storage-account yourstorageaccount

# Deploy
func azure functionapp publish databricks-error-recovery-func
```

---

### **STEP 4: Deploy Approval Webhook Function**

```bash
# Create separate Function App for webhook
az functionapp create \
  --resource-group your-rg \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.10 \
  --functions-version 4 \
  --name approval-webhook-func \
  --storage-account yourstorageaccount

# Deploy function_app.py
func azure functionapp publish approval-webhook-func
```

**Get the webhook URL:**
```bash
az functionapp function show \
  --name approval-webhook-func \
  --resource-group your-rg \
  --function-name approval \
  --query invokeUrlTemplate -o tsv
```

Update the email template in the main workflow with this URL.

---

### **STEP 5: Set Up Event Grid for Databricks Triggers**

```bash
# Create Event Grid topic
az eventgrid topic create \
  --name databricks-job-events \
  --resource-group your-rg \
  --location eastus

# Create subscription to trigger your workflow
az eventgrid event-subscription create \
  --name job-failure-subscription \
  --source-resource-id /subscriptions/{sub-id}/resourceGroups/{rg}/providers/Microsoft.EventGrid/topics/databricks-job-events \
  --endpoint https://databricks-error-recovery.{region}.azurecontainerapps.io/api/trigger \
  --endpoint-type webhook
```

---

### **STEP 6: Configure Databricks to Send Events**

In your Databricks workspace, set up a webhook to send job failure events:

```python
# Databricks Notebook: Configure Job Webhook
import requests
import json

# Event Grid endpoint
EVENT_GRID_ENDPOINT = "https://databricks-job-events.{region}.eventgrid.azure.net/api/events"
EVENT_GRID_KEY = "your-event-grid-key"

def send_job_failure_event(job_id, run_id, error_message):
    """Send job failure event to Event Grid"""
    event = [{
        "id": f"{job_id}_{run_id}",
        "eventType": "Databricks.Job.Failed",
        "subject": f"jobs/{job_id}",
        "eventTime": datetime.utcnow().isoformat(),
        "data": {
            "job_id": job_id,
            "run_id": run_id,
            "error_message": error_message
        },
        "dataVersion": "1.0"
    }]
    
    headers = {
        "aeg-sas-key": EVENT_GRID_KEY,
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        EVENT_GRID_ENDPOINT,
        headers=headers,
        json=event
    )
    
    return response.status_code

# Add this to your Databricks job failure handler
```

---

## Where to Run What

### **Summary Table**

| Component | Deployment Location | Purpose |
|-----------|-------------------|---------|
| `databricks_error_recovery_workflow.py` | **Azure Container Apps** or Azure Functions | Main orchestration workflow |
| `function_app.py` | **Azure Functions (HTTP Trigger)** | Approval webhook handler |
| Azure AI Foundry Project | **Azure AI Foundry** | Agent service orchestration |
| Azure OpenAI | **Azure Cognitive Services** | GPT-4o model hosting |
| Azure AI Search | **Azure AI Search** | Vector store for RAG |
| Azure Document Intelligence | **Azure Cognitive Services** | IDP for visual processing |
| Event Grid | **Azure Event Grid** | Event routing from Databricks |
| Azure Monitor | **Azure Monitor** | Observability and tracing |

---

## Testing the Workflow

### **Local Testing (Development)**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export $(cat .env | xargs)

# 3. Run locally
python databricks_error_recovery_workflow.py
```

### **Production Testing**

1. Trigger a Databricks job failure (intentionally)
2. Check Event Grid delivery status
3. Monitor Container App logs:
   ```bash
   az containerapp logs show \
     --name databricks-error-recovery \
     --resource-group your-rg \
     --follow
   ```
4. Check email delivery
5. Click approval link
6. Verify job restart in Databricks

---

## Monitoring & Observability

### **Azure Monitor Queries**

```kusto
// View all workflow executions
traces
| where cloud_RoleName == "databricks-error-recovery"
| where message contains "execute_workflow"
| project timestamp, message, severityLevel

// Check RAG evaluation scores
customMetrics
| where name == "rag_evaluation"
| extend scores = parse_json(value)
| project timestamp, 
    faithfulness = scores.faithfulness,
    answer_relevance = scores.answer_relevance,
    context_precision = scores.context_precision
```

### **LangSmith Dashboard**

1. Go to langsmith.com
2. Navigate to your project: "databricks-error-recovery"
3. View traces, evaluation scores, and feedback

---

## Troubleshooting

### **Issue: Workflow times out**
- **Solution**: Increase Container App timeout or use Azure Functions with Durable Functions extension

### **Issue: Email not received**
- **Solution**: Verify Azure Communication Services email domain is verified and not in spam

### **Issue: RAG returns no results**
- **Solution**: Ensure Azure AI Search index is created and populated with historical error data

### **Issue: TensorFlow model fails to load**
- **Solution**: Pre-train and upload model to Azure Blob Storage, mount in Container App

---

## Cost Optimization

| Service | Estimated Monthly Cost | Optimization Tips |
|---------|----------------------|------------------|
| Azure Container Apps | $50-200 | Use consumption plan, scale to zero |
| Azure OpenAI | $100-500 | Cache embeddings, use batch processing |
| Azure AI Search | $100-300 | Use standard tier, optimize index size |
| Azure Functions | $0-50 | Consumption plan for webhook |
| **Total** | **$250-1050** | |

---

## Next Steps

1. ✅ Deploy all Azure resources (Step 1)
2. ✅ Configure environment variables (Step 2)
3. ✅ Choose deployment option: Container Apps or Functions (Step 3)
4. ✅ Deploy approval webhook (Step 4)
5. ✅ Set up Event Grid (Step 5)
6. ✅ Configure Databricks webhooks (Step 6)
7. ✅ Test end-to-end workflow
8. ✅ Monitor and optimize

---

## Support

For issues or questions:
- Azure AI Foundry: https://docs.microsoft.com/azure/ai-foundry
- Databricks: https://docs.databricks.com
- LangChain: https://docs.langchain.com
