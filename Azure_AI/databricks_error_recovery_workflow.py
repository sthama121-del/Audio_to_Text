"""
Agentic Workflow: Databricks Job Error Recovery
Azure AI Foundry + RAG + IDP + Human-in-the-Loop

DEPLOYMENT LOCATION: Azure Function App or Azure Container Apps
RUNTIME: Python 3.10+
"""

import os
import json
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

# Azure SDKs
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import Agent, MessageTextContent, RunStatus
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.communication.email import EmailClient

# LangChain & LangSmith
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.chains import RetrievalQA
from langsmith import Client as LangSmithClient
from langsmith.evaluation import evaluate

# Document Intelligence for IDP
from azure.ai.documentintelligence import DocumentIntelligenceClient

# Databricks SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState

# TensorFlow for reliability prediction
import tensorflow as tf
import numpy as np

# Observability
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# ==================== Configuration ====================
class Config:
    # Azure AI Foundry
    PROJECT_CONNECTION_STRING = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING")
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"
    
    # Azure AI Search
    SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
    SEARCH_INDEX_NAME = "databricks-error-fixes"
    
    # Document Intelligence
    DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
    DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
    
    # Databricks
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
    
    # LangSmith
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = "databricks-error-recovery"
    
    # Azure Communication Services
    COMMUNICATION_CONNECTION_STRING = os.getenv("AZURE_COMMUNICATION_CONNECTION_STRING")
    SENDER_EMAIL = os.getenv("SENDER_EMAIL")
    ENGINEERING_TEAM_EMAIL = os.getenv("ENGINEERING_TEAM_EMAIL")
    
    # TensorFlow Model Path
    RELIABILITY_MODEL_PATH = "./models/pipeline_reliability_classifier.h5"

# ==================== Observability Setup ====================
configure_azure_monitor(
    connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
)
tracer = trace.get_tracer(__name__)

langsmith_client = LangSmithClient(api_key=Config.LANGSMITH_API_KEY)

# ==================== 1. Log Retrieval Agent ====================
class DatabricksLogRetriever:
    def __init__(self):
        self.workspace_client = WorkspaceClient(
            host=Config.DATABRICKS_HOST,
            token=Config.DATABRICKS_TOKEN
        )
    
    @tracer.start_as_current_span("retrieve_job_logs")
    def retrieve_failed_job_logs(self, job_id: int) -> Dict:
        """Retrieve logs and metadata for a failed Databricks job"""
        span = trace.get_current_span()
        
        try:
            # Get job run details
            runs = self.workspace_client.jobs.list_runs(job_id=job_id, limit=1)
            latest_run = next(iter(runs))
            
            if latest_run.state.life_cycle_state != RunLifeCycleState.TERMINATED:
                span.set_status(Status(StatusCode.ERROR, "Job not in terminated state"))
                return None
            
            # Retrieve error logs
            run_output = self.workspace_client.jobs.get_run_output(latest_run.run_id)
            
            # Get visual telemetry (screenshots if available)
            screenshots = self._get_job_screenshots(latest_run.run_id)
            
            log_data = {
                "job_id": job_id,
                "run_id": latest_run.run_id,
                "error_message": run_output.error or "Unknown error",
                "error_trace": run_output.error_trace or "",
                "logs": run_output.logs or "",
                "screenshots": screenshots,
                "cluster_id": latest_run.cluster_instance.cluster_id,
                "start_time": latest_run.start_time,
                "end_time": latest_run.end_time,
                "task_key": latest_run.tasks[0].task_key if latest_run.tasks else None
            }
            
            span.add_event("logs_retrieved", {"run_id": latest_run.run_id})
            return log_data
            
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
    
    def _get_job_screenshots(self, run_id: int) -> List[str]:
        """Retrieve visual telemetry from Databricks job run"""
        # Implementation would use Databricks API to fetch notebook output images
        # Returns list of base64-encoded images or blob storage URLs
        return []

# ==================== 2. RAG Implementation with Azure AI Search ====================
class ErrorKnowledgeBase:
    def __init__(self):
        credential = DefaultAzureCredential()
        
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_KEY,
            deployment=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )
        
        self.vector_store = AzureSearch(
            azure_search_endpoint=Config.SEARCH_ENDPOINT,
            azure_search_key=Config.SEARCH_KEY,
            index_name=Config.SEARCH_INDEX_NAME,
            embedding_function=self.embeddings.embed_query
        )
    
    @tracer.start_as_current_span("vectorize_and_index")
    def index_error_log(self, log_data: Dict):
        """Vectorize error log and index in Azure AI Search"""
        document = {
            "content": f"{log_data['error_message']}\n{log_data['error_trace']}\n{log_data['logs']}",
            "metadata": {
                "job_id": log_data["job_id"],
                "run_id": log_data["run_id"],
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        self.vector_store.add_texts(
            texts=[document["content"]],
            metadatas=[document["metadata"]]
        )
    
    @tracer.start_as_current_span("retrieve_similar_fixes")
    def retrieve_similar_fixes(self, error_description: str, k: int = 5) -> List[Dict]:
        """Retrieve similar historical error fixes using RAG"""
        results = self.vector_store.similarity_search_with_relevance_scores(
            error_description,
            k=k
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            }
            for doc, score in results
        ]

# ==================== 3. IDP Visual Underwriting Agent ====================
class VisualUnderwritingAgent:
    def __init__(self):
        self.doc_intelligence_client = DocumentIntelligenceClient(
            endpoint=Config.DOC_INTELLIGENCE_ENDPOINT,
            credential=DefaultAzureCredential()
        )
    
    @tracer.start_as_current_span("analyze_visual_telemetry")
    async def analyze_screenshots(self, screenshots: List[str]) -> Dict:
        """Process screenshots using Azure Document Intelligence"""
        if not screenshots:
            return {"visual_insights": "No visual data available"}
        
        visual_insights = []
        
        for screenshot_url in screenshots:
            # Analyze layout and extract text from error screenshots
            poller = await self.doc_intelligence_client.begin_analyze_document(
                "prebuilt-layout",
                screenshot_url
            )
            result = await poller.result()
            
            # Extract error messages, stack traces from images
            extracted_text = " ".join([
                line.content for page in result.pages 
                for line in page.lines
            ])
            
            visual_insights.append({
                "screenshot": screenshot_url,
                "extracted_text": extracted_text,
                "confidence": result.confidence if hasattr(result, 'confidence') else 0.0
            })
        
        return {
            "visual_insights": visual_insights,
            "summary": self._summarize_visual_errors(visual_insights)
        }
    
    def _summarize_visual_errors(self, insights: List[Dict]) -> str:
        """Summarize extracted visual error information"""
        all_text = " ".join([i["extracted_text"] for i in insights])
        return all_text[:500]  # Truncate for context

# ==================== 4. Azure AI Foundry Agent Orchestrator ====================
class ErrorRecoveryAgent:
    def __init__(self):
        self.project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=Config.PROJECT_CONNECTION_STRING
        )
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_KEY,
            deployment_name=Config.AZURE_OPENAI_DEPLOYMENT,
            temperature=0.1
        )
        
        self.knowledge_base = ErrorKnowledgeBase()
        self.visual_agent = VisualUnderwritingAgent()
        
        # Create agent in Azure AI Foundry
        self.agent = self.project_client.agents.create_agent(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            name="Databricks Error Recovery Agent",
            instructions="""You are an expert in analyzing Databricks job failures.
            Your role is to:
            1. Analyze error logs and visual telemetry
            2. Compare against historical fixes
            3. Categorize the error type
            4. Suggest a specific fix with confidence score
            5. Provide reasoning for your recommendation
            
            Be precise and actionable in your suggestions."""
        )
    
    @tracer.start_as_current_span("analyze_error")
    async def analyze_error(self, log_data: Dict) -> Dict:
        """Orchestrate error analysis using GPT-4o and RAG"""
        span = trace.get_current_span()
        
        # Step 1: Process visual telemetry if available
        visual_analysis = {}
        if log_data.get("screenshots"):
            visual_analysis = await self.visual_agent.analyze_screenshots(
                log_data["screenshots"]
            )
        
        # Step 2: Retrieve similar historical fixes
        error_context = f"{log_data['error_message']}\n{log_data['error_trace']}"
        if visual_analysis:
            error_context += f"\nVisual Insights: {visual_analysis.get('summary', '')}"
        
        similar_fixes = self.knowledge_base.retrieve_similar_fixes(error_context)
        
        # Step 3: Create thread and analyze with agent
        thread = self.project_client.agents.create_thread()
        
        context_prompt = self._build_analysis_prompt(
            log_data, 
            similar_fixes, 
            visual_analysis
        )
        
        message = self.project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=context_prompt
        )
        
        # Run agent analysis
        run = self.project_client.agents.create_run(
            thread_id=thread.id,
            agent_id=self.agent.id
        )
        
        # Wait for completion
        while run.status in [RunStatus.QUEUED, RunStatus.IN_PROGRESS]:
            await asyncio.sleep(1)
            run = self.project_client.agents.get_run(
                thread_id=thread.id,
                run_id=run.id
            )
        
        # Get response
        messages = self.project_client.agents.list_messages(thread_id=thread.id)
        agent_response = next(
            (msg for msg in messages if msg.role == "assistant"),
            None
        )
        
        if not agent_response:
            raise Exception("Agent failed to produce analysis")
        
        analysis_result = self._parse_agent_response(agent_response.content[0].text.value)
        
        # Step 4: Evaluate with RAG Triad (via LangSmith)
        evaluation_scores = await self._evaluate_rag_quality(
            error_context,
            similar_fixes,
            analysis_result
        )
        
        span.add_event("analysis_complete", {
            "error_category": analysis_result["category"],
            "confidence": analysis_result["confidence"]
        })
        
        return {
            **analysis_result,
            "visual_analysis": visual_analysis,
            "evaluation_scores": evaluation_scores,
            "thread_id": thread.id
        }
    
    def _build_analysis_prompt(
        self, 
        log_data: Dict, 
        similar_fixes: List[Dict],
        visual_analysis: Dict
    ) -> str:
        """Build comprehensive analysis prompt"""
        prompt = f"""Analyze this Databricks job failure:

**Job Details:**
- Job ID: {log_data['job_id']}
- Run ID: {log_data['run_id']}
- Cluster ID: {log_data['cluster_id']}

**Error Information:**
{log_data['error_message']}

**Stack Trace:**
{log_data['error_trace'][:1000]}

**Logs:**
{log_data['logs'][:1000]}
"""
        
        if visual_analysis.get('summary'):
            prompt += f"\n**Visual Telemetry:**\n{visual_analysis['summary']}\n"
        
        if similar_fixes:
            prompt += "\n**Historical Similar Fixes:**\n"
            for i, fix in enumerate(similar_fixes[:3], 1):
                prompt += f"{i}. (Relevance: {fix['relevance_score']:.2f})\n{fix['content'][:300]}\n\n"
        
        prompt += """
**Provide analysis in JSON format:**
{
  "category": "error category (e.g., OOM, Dependency, Configuration, Data Quality)",
  "root_cause": "technical root cause",
  "suggested_fix": "specific actionable fix",
  "confidence": 0.0-1.0,
  "reasoning": "explanation of analysis"
}
"""
        return prompt
    
    def _parse_agent_response(self, response_text: str) -> Dict:
        """Parse structured response from agent"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return {
                    "category": "Unknown",
                    "root_cause": response_text,
                    "suggested_fix": "Manual investigation required",
                    "confidence": 0.5,
                    "reasoning": response_text
                }
        except Exception as e:
            return {
                "category": "Parse Error",
                "root_cause": str(e),
                "suggested_fix": "Review agent output manually",
                "confidence": 0.0,
                "reasoning": response_text
            }
    
    async def _evaluate_rag_quality(
        self,
        query: str,
        retrieved_docs: List[Dict],
        answer: Dict
    ) -> Dict:
        """Evaluate RAG Triad: Faithfulness, Answer Relevance, Context Precision"""
        
        # LangSmith evaluation
        def faithfulness_evaluator(run, example):
            """Check if answer is grounded in retrieved context"""
            contexts = [doc['content'] for doc in retrieved_docs]
            answer_text = answer.get('suggested_fix', '')
            
            # Simple heuristic: check if key terms appear in context
            context_text = " ".join(contexts)
            answer_terms = set(answer_text.lower().split())
            context_terms = set(context_text.lower().split())
            overlap = len(answer_terms & context_terms) / max(len(answer_terms), 1)
            
            return {"score": overlap}
        
        def answer_relevance_evaluator(run, example):
            """Check if answer addresses the query"""
            # Use LLM to judge relevance
            relevance_prompt = f"""
            Query: {query}
            Answer: {answer.get('suggested_fix', '')}
            
            Is this answer relevant to the query? Score 0.0-1.0
            Respond with just a number.
            """
            # Simplified - in production use actual LLM call
            return {"score": answer.get('confidence', 0.5)}
        
        def context_precision_evaluator(run, example):
            """Check if retrieved contexts are relevant"""
            relevant_count = sum(1 for doc in retrieved_docs if doc['relevance_score'] > 0.7)
            precision = relevant_count / max(len(retrieved_docs), 1)
            return {"score": precision}
        
        scores = {
            "faithfulness": faithfulness_evaluator(None, None)["score"],
            "answer_relevance": answer_relevance_evaluator(None, None)["score"],
            "context_precision": context_precision_evaluator(None, None)["score"]
        }
        
        # Log to LangSmith
        langsmith_client.create_feedback(
            run_id=None,  # Would be actual run ID
            key="rag_evaluation",
            score=sum(scores.values()) / 3,
            value=scores
        )
        
        return scores

# ==================== 5. TensorFlow Reliability Classifier ====================
class PipelineReliabilityClassifier:
    def __init__(self):
        self.model = self._load_or_create_model()
    
    def _load_or_create_model(self) -> tf.keras.Model:
        """Load pretrained model or create new one"""
        if os.path.exists(Config.RELIABILITY_MODEL_PATH):
            return tf.keras.models.load_model(Config.RELIABILITY_MODEL_PATH)
        
        # Create simple classification model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(50,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(3, activation='softmax')  # High/Medium/Low reliability
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @tracer.start_as_current_span("predict_reliability")
    def predict_reliability(self, log_data: Dict, analysis: Dict) -> Dict:
        """Predict pipeline reliability trend"""
        
        # Feature engineering from log data and analysis
        features = self._extract_features(log_data, analysis)
        features_array = np.array([features])
        
        # Predict
        prediction = self.model.predict(features_array, verbose=0)
        reliability_class = np.argmax(prediction[0])
        confidence = float(prediction[0][reliability_class])
        
        reliability_labels = ["High Risk", "Medium Risk", "Low Risk"]
        
        return {
            "reliability_class": reliability_labels[reliability_class],
            "confidence": confidence,
            "prediction_probabilities": {
                label: float(prob)
                for label, prob in zip(reliability_labels, prediction[0])
            }
        }
    
    def _extract_features(self, log_data: Dict, analysis: Dict) -> List[float]:
        """Extract numerical features for classification"""
        # Simplified feature extraction - in production, use more sophisticated features
        features = [
            len(log_data.get('error_trace', '')),
            len(log_data.get('logs', '')),
            analysis.get('confidence', 0.5),
            1 if log_data.get('screenshots') else 0,
            # ... add more features up to 50 dimensions
        ]
        
        # Pad to 50 features
        features.extend([0.0] * (50 - len(features)))
        return features[:50]

# ==================== 6. Human-in-the-Loop Orchestration ====================
class HumanInTheLoopOrchestrator:
    def __init__(self):
        self.email_client = EmailClient.from_connection_string(
            Config.COMMUNICATION_CONNECTION_STRING
        )
        self.approval_store = {}  # In production, use Azure Table Storage or Cosmos DB
    
    @tracer.start_as_current_span("send_approval_request")
    async def send_approval_request(
        self,
        job_id: int,
        run_id: int,
        analysis: Dict,
        reliability_prediction: Dict
    ) -> str:
        """Send email to engineering team for approval"""
        
        approval_id = f"{job_id}_{run_id}_{datetime.utcnow().timestamp()}"
        
        # Build email content
        email_body = f"""
        <h2>Databricks Job Failure - Action Required</h2>
        
        <p><strong>Job ID:</strong> {job_id}</p>
        <p><strong>Run ID:</strong> {run_id}</p>
        
        <h3>AI Analysis Results</h3>
        <ul>
            <li><strong>Error Category:</strong> {analysis['category']}</li>
            <li><strong>Root Cause:</strong> {analysis['root_cause']}</li>
            <li><strong>Confidence:</strong> {analysis['confidence']:.2%}</li>
            <li><strong>Reliability Prediction:</strong> {reliability_prediction['reliability_class']} 
                ({reliability_prediction['confidence']:.2%})</li>
        </ul>
        
        <h3>Suggested Fix</h3>
        <p>{analysis['suggested_fix']}</p>
        
        <h3>RAG Evaluation Scores</h3>
        <ul>
            <li>Faithfulness: {analysis['evaluation_scores']['faithfulness']:.2%}</li>
            <li>Answer Relevance: {analysis['evaluation_scores']['answer_relevance']:.2%}</li>
            <li>Context Precision: {analysis['evaluation_scores']['context_precision']:.2%}</li>
        </ul>
        
        <h3>Action Required</h3>
        <p>Please approve or reject the recommended action:</p>
        
        <p>
            <a href="https://your-function-app.azurewebsites.net/api/approval?id={approval_id}&action=approve"
               style="background-color: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                ✓ Approve - Restart Job
            </a>
            
            <a href="https://your-function-app.azurewebsites.net/api/approval?id={approval_id}&action=reject"
               style="background-color: #dc3545; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-left: 10px;">
                ✗ Reject - Terminate Job
            </a>
        </p>
        
        <p><em>This is an automated message from the Azure AI Foundry Agent Service.</em></p>
        """
        
        message = {
            "senderAddress": Config.SENDER_EMAIL,
            "recipients": {
                "to": [{"address": Config.ENGINEERING_TEAM_EMAIL}]
            },
            "content": {
                "subject": f"[Action Required] Databricks Job {job_id} Failure Analysis",
                "html": email_body
            }
        }
        
        poller = self.email_client.begin_send(message)
        result = poller.result()
        
        # Store approval request
        self.approval_store[approval_id] = {
            "job_id": job_id,
            "run_id": run_id,
            "status": "pending",
            "analysis": analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return approval_id
    
    async def wait_for_approval(
        self,
        approval_id: str,
        timeout_seconds: int = 3600
    ) -> Optional[str]:
        """Wait for human approval signal (webhook callback)"""
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check approval status
            approval_data = self.approval_store.get(approval_id)
            
            if not approval_data:
                return None
            
            if approval_data["status"] in ["approved", "rejected"]:
                return approval_data["status"]
            
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout_seconds:
                approval_data["status"] = "timeout"
                return "timeout"
            
            await asyncio.sleep(5)  # Poll every 5 seconds
    
    def handle_approval_webhook(self, approval_id: str, action: str):
        """Webhook handler for approval/rejection (called by Azure Function)"""
        if approval_id in self.approval_store:
            self.approval_store[approval_id]["status"] = action
            self.approval_store[approval_id]["decision_time"] = datetime.utcnow().isoformat()

# ==================== 7. Action Execution ====================
class DatabricksActionExecutor:
    def __init__(self):
        self.workspace_client = WorkspaceClient(
            host=Config.DATABRICKS_HOST,
            token=Config.DATABRICKS_TOKEN
        )
    
    @tracer.start_as_current_span("restart_job")
    def restart_job(self, job_id: int) -> Dict:
        """Restart failed Databricks job"""
        span = trace.get_current_span()
        
        try:
            run = self.workspace_client.jobs.run_now(job_id=job_id)
            
            span.add_event("job_restarted", {
                "job_id": job_id,
                "new_run_id": run.run_id
            })
            
            return {
                "action": "restart",
                "job_id": job_id,
                "run_id": run.run_id,
                "status": "submitted",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
    
    @tracer.start_as_current_span("terminate_job")
    def terminate_job(self, job_id: int, run_id: int) -> Dict:
        """Terminate failed Databricks job"""
        span = trace.get_current_span()
        
        try:
            self.workspace_client.jobs.cancel_run(run_id=run_id)
            
            span.add_event("job_terminated", {
                "job_id": job_id,
                "run_id": run_id
            })
            
            return {
                "action": "terminate",
                "job_id": job_id,
                "run_id": run_id,
                "status": "cancelled",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

# ==================== 8. End-to-End Workflow Orchestrator ====================
class ErrorRecoveryWorkflow:
    def __init__(self):
        self.log_retriever = DatabricksLogRetriever()
        self.error_agent = ErrorRecoveryAgent()
        self.reliability_classifier = PipelineReliabilityClassifier()
        self.hitl_orchestrator = HumanInTheLoopOrchestrator()
        self.action_executor = DatabricksActionExecutor()
    
    @tracer.start_as_current_span("execute_workflow")
    async def execute(self, job_id: int):
        """Execute end-to-end error recovery workflow"""
        span = trace.get_current_span()
        span.set_attribute("job_id", job_id)
        
        try:
            # Step 1: Retrieve logs
            print(f"[1/6] Retrieving logs for Job {job_id}...")
            log_data = self.log_retriever.retrieve_failed_job_logs(job_id)
            
            if not log_data:
                print("No failed job found.")
                return
            
            # Step 2: Analyze error with AI agent
            print(f"[2/6] Analyzing error with Azure AI Foundry Agent...")
            analysis = await self.error_agent.analyze_error(log_data)
            
            # Step 3: Predict reliability
            print(f"[3/6] Predicting pipeline reliability with TensorFlow...")
            reliability_prediction = self.reliability_classifier.predict_reliability(
                log_data,
                analysis
            )
            
            # Step 4: Human-in-the-loop
            print(f"[4/6] Sending approval request to engineering team...")
            approval_id = await self.hitl_orchestrator.send_approval_request(
                job_id,
                log_data["run_id"],
                analysis,
                reliability_prediction
            )
            
            print(f"[5/6] Waiting for human approval (ID: {approval_id})...")
            approval_status = await self.hitl_orchestrator.wait_for_approval(
                approval_id,
                timeout_seconds=3600
            )
            
            # Step 5: Execute action based on approval
            print(f"[6/6] Executing action based on approval: {approval_status}")
            
            if approval_status == "approved":
                result = self.action_executor.restart_job(job_id)
                print(f"✓ Job restarted: {result}")
            elif approval_status == "rejected":
                result = self.action_executor.terminate_job(
                    job_id,
                    log_data["run_id"]
                )
                print(f"✗ Job terminated: {result}")
            else:
                print(f"⚠ Approval timeout or invalid status: {approval_status}")
                result = {"action": "no_action", "reason": approval_status}
            
            # Step 6: Log final result to Azure Monitor
            span.add_event("workflow_complete", {
                "approval_status": approval_status,
                "final_action": result.get("action")
            })
            
            print("\n=== Workflow Complete ===")
            print(f"Analysis: {analysis['category']} - {analysis['suggested_fix']}")
            print(f"Confidence: {analysis['confidence']:.2%}")
            print(f"Reliability: {reliability_prediction['reliability_class']}")
            print(f"Action: {result.get('action')}")
            
            return result
            
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            print(f"Error in workflow: {e}")
            raise

# ==================== Main Entry Point ====================
async def main():
    """Main entry point for the workflow"""
    
    # Example: Trigger workflow for a failed job
    FAILED_JOB_ID = 12345  # Replace with actual job ID
    
    workflow = ErrorRecoveryWorkflow()
    await workflow.execute(FAILED_JOB_ID)

if __name__ == "__main__":
    asyncio.run(main())
