# Databricks Error Recovery - Agentic Workflow

An intelligent, autonomous system for detecting, analyzing, and recovering from Databricks job failures using Azure AI Foundry, RAG, and Human-in-the-Loop orchestration.

## 🎯 Features

- **Automated Error Detection**: Retrieves logs from failed Databricks jobs
- **RAG-based Analysis**: Compares errors against historical knowledge base using Azure AI Search
- **Visual Intelligence**: Processes screenshots and visual telemetry with Azure Document Intelligence
- **AI-Powered Root Cause Analysis**: Uses GPT-4o via Azure AI Foundry Agent Service
- **Reliability Prediction**: TensorFlow model predicts pipeline stability trends
- **Human-in-the-Loop**: Email-based approval system for critical actions
- **End-to-End Observability**: Azure Monitor + LangSmith tracing
- **Automated Recovery**: Restarts or terminates jobs based on human approval

## 🏗️ Architecture

```
Databricks Job Failure → Event Grid → Container App (Main Workflow)
                                           ↓
                            ┌──────────────┴──────────────┐
                            ↓                             ↓
                    Azure AI Foundry              Azure Functions
                    (Agent Service)            (Approval Webhook)
                            ↓                             ↓
                    Action Execution ←────────────────────┘
                    (Restart/Terminate)
```

## 📁 Project Structure

```
databricks-error-recovery/
├── databricks_error_recovery_workflow.py  # Main workflow orchestrator
├── function_app.py                        # Approval webhook handler
├── requirements.txt                       # Python dependencies
├── Dockerfile                             # Container image definition
├── .env.template                          # Environment variables template
├── DEPLOYMENT_GUIDE.md                    # Detailed deployment instructions
└── README.md                              # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Azure Subscription** with the following services:
   - Azure AI Foundry Project
   - Azure OpenAI (with GPT-4o deployment)
   - Azure AI Search
   - Azure Document Intelligence
   - Azure Communication Services
   - Azure Application Insights
   - Azure Table Storage

2. **Databricks Workspace** with:
   - Personal Access Token
   - Job configured with failure notifications

3. **Development Tools**:
   - Python 3.10+
   - Docker (for containerization)
   - Azure CLI
   - Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd databricks-error-recovery
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env with your Azure service credentials
   ```

4. **Run locally** (for testing):
   ```bash
   python databricks_error_recovery_workflow.py
   ```

## 🔧 Configuration

### Environment Variables

See `.env.template` for all required environment variables. Key configurations:

```bash
# Azure AI Foundry
AZURE_AI_PROJECT_CONNECTION_STRING=your-connection-string

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_KEY=your-key

# Databricks
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=your-token

# ... (see .env.template for complete list)
```

## 📦 Deployment

### Option 1: Azure Container Apps (Recommended)

```bash
# Build Docker image
docker build -t databricks-error-recovery:latest .

# Push to Azure Container Registry
az acr login --name yourregistry
docker tag databricks-error-recovery:latest yourregistry.azurecr.io/databricks-error-recovery:latest
docker push yourregistry.azurecr.io/databricks-error-recovery:latest

# Deploy to Container Apps
az containerapp create \
  --name databricks-error-recovery \
  --resource-group your-rg \
  --environment your-env \
  --image yourregistry.azurecr.io/databricks-error-recovery:latest \
  --min-replicas 1 \
  --max-replicas 3
```

### Option 2: Azure Functions

```bash
# Initialize Functions project
func init ErrorRecoveryFunctions --python
cd ErrorRecoveryFunctions

# Deploy
func azure functionapp publish databricks-error-recovery-func
```

**👉 For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**

## 🎮 Usage

### Trigger the Workflow

1. **Automatic Trigger** (Production):
   - Configure Event Grid to listen to Databricks job failures
   - Workflow automatically triggers on job failure

2. **Manual Trigger** (Testing):
   ```python
   from databricks_error_recovery_workflow import ErrorRecoveryWorkflow
   
   workflow = ErrorRecoveryWorkflow()
   await workflow.execute(job_id=12345)
   ```

### Approval Process

1. Engineering team receives email with error analysis
2. Email contains:
   - Error category and root cause
   - Suggested fix with confidence score
   - RAG evaluation metrics
   - Reliability prediction
3. Team clicks "Approve" or "Reject"
4. Workflow automatically restarts or terminates job

## 📊 Monitoring

### Azure Monitor

```kusto
// View workflow executions
traces
| where cloud_RoleName == "databricks-error-recovery"
| project timestamp, message, severityLevel

// Check RAG evaluation scores
customMetrics
| where name == "rag_evaluation"
| extend scores = parse_json(value)
```

### LangSmith Dashboard

1. Navigate to https://smith.langchain.com
2. Select project: "databricks-error-recovery"
3. View traces and evaluation metrics

## 🧪 Testing

### Unit Tests

```bash
pytest tests/
```

### Integration Tests

```bash
# Test with a known failed job
python -m pytest tests/integration/test_workflow.py --job-id=12345
```

### End-to-End Test

1. Trigger a Databricks job failure (intentionally)
2. Monitor Container App logs
3. Check email delivery
4. Test approval workflow
5. Verify job action in Databricks

## 🔍 RAG Triad Evaluation

The system evaluates AI responses using three metrics:

1. **Faithfulness**: Is the answer grounded in retrieved context?
2. **Answer Relevance**: Does the answer address the specific error?
3. **Context Precision**: Are retrieved documents relevant?

Target scores: All metrics > 0.75 for production use.

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Workflow times out | Increase container timeout or use Durable Functions |
| Email not received | Verify Communication Services email domain |
| RAG returns no results | Populate Azure AI Search index with historical data |
| TensorFlow model fails | Pre-train and upload to Azure Blob Storage |

## 💰 Cost Estimation

| Service | Monthly Cost (USD) |
|---------|-------------------|
| Azure Container Apps | $50-200 |
| Azure OpenAI | $100-500 |
| Azure AI Search | $100-300 |
| Azure Functions | $0-50 |
| Other services | $50-100 |
| **Total** | **$300-1150** |

*Costs vary based on usage and scale*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 📞 Support

For issues or questions:
- **Documentation**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Azure Support**: https://azure.microsoft.com/support
- **Issues**: Open an issue in this repository

## 🙏 Acknowledgments

- Built with Azure AI Foundry
- Powered by LangChain and LangSmith
- Observability by Azure Monitor and OpenTelemetry

---

**Ready to deploy?** Start with the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)! 🚀
