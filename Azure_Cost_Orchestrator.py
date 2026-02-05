#Azure_Cost_Orchestrator.py
import os
import logging
from azure.identity import DefaultAzureCredential
from azure.openai import OpenAIClient
from azure.search.documents import SearchClient

# --- TEACHING: MEASUREMENT & OBSERVABILITY ---
# We use Azure Monitor-ready logging to track every request's cost.
# In production, these logs are routed to a 'Log Analytics Workspace' 
# for Cost Dashboards.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CostOptimizer")

class AICostGateway:
    def __init__(self):
        self.credential = DefaultAzureCredential()
        # Deployment IDs for different tiers
        self.cheap_model = "gpt-35-turbo" # 10x-20x cheaper
        self.expensive_model = "gpt-4o"
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    def route_request(self, user_query):
        """
        TEACHING: MODEL CASCADING
        Simple queries (like greetings or basic facts) are routed to smaller models.
        Complex reasoning goes to GPT-4.
        """
        # Logic: If query is < 10 words, assume it's 'Simple'
        # In a real app, use a classifier model here.
        is_simple = len(user_query.split()) < 10
        model = self.cheap_model if is_simple else self.expensive_model
        
        logger.info(f"Routing query to: {model} (Complexity: {'Low' if is_simple else 'High'})")
        return self.execute_safe_call(user_query, model)

    def execute_safe_call(self, prompt, model_deployment):
        client = OpenAIClient(endpoint=self.openai_endpoint, credential=self.credential)

        # --- TEACHING: TOKEN CAPPING ---
        # Setting 'max_tokens' prevents "runaway" costs where a model 
        # generates thousands of unnecessary lines.
        response = client.get_chat_completions(
            deployment_id=model_deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150, # Capping output length
            user="BU-Marketing-01" # TEACHING: TAGGING for chargeback
        )

        # --- TEACHING: REAL-TIME MEASUREMENT ---
        # The 'usage' field returns the EXACT tokens consumed. 
        # Caching status can also be monitored here.
        usage = response.usage
        logger.info(f"MEASUREMENT: [Prompt: {usage.prompt_tokens}] [Completion: {usage.completion_tokens}] [Total: {usage.total_tokens}]")
        
        return response.choices[0].message.content

    def secure_retrieval(self, query, user_id):
        """
        TEACHING: RBAC SCOPING
        We restrict the search index based on the user's role to prevent 
        searching across the entire enterprise unnecessarily.
        """
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="sop-index",
            credential=self.credential
        )
        
        # Using OData filters to restrict retrieval to the user's Department
        # This reduces data volume and improves relevance.
        results = search_client.search(
            search_text=query,
            filter=f"department eq 'Finance'" # Mock RBAC Filter
        )
        return results