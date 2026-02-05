#enterprise_rag_ops_demo.py
"""
enterprise_rag_ops_demo.py

This script is a *teaching* / interview-demo example that shows patterns for:
1) Correlation IDs for tracing requests end-to-end
2) Structured logging (similar to how you'd feed App Insights / Azure Monitor)
3) An MCP-like "tool registry" pattern (agent/tool abstraction)
4) Calling external systems via REST (ServiceNow example)
5) Calling data platforms (Databricks example placeholder)
6) A simple "RAG fallback" gate using a similarity threshold
7) An Azure Function-style `main(req)` entrypoint

NOTE: This is NOT production-ready code. Production would add:
- Managed Identity / Key Vault (instead of username/password)
- Retries + backoff + circuit breakers
- Input validation + schema checks
- Proper error handling and idempotency for ServiceNow ticket creation
"""

# -----------------------------
# Imports (bring in libraries)
# -----------------------------

import os          # Lets us read environment variables like SERVICENOW_USER
import uuid        # Generates unique IDs (UUIDs) used as correlation IDs
import requests    # Popular HTTP client library to call REST APIs

# typing: lets us add type hints (helps readability, IDE hints, safer code)
from typing import Dict, Any, Callable

# dataclasses: lets us define simple "data containers" without boilerplate
from dataclasses import dataclass


# -----------------------------
# Observability / Logging
# -----------------------------
def log(event: str, cid: str, **k: Any) -> None:
    """
    A tiny structured logger.

    event: name of what happened (e.g., 'request.start')
    cid: correlation_id (same ID flows across systems for tracing)
    **k: extra key/value details (any additional context you want in logs)

    In production:
    - you'd use Python logging + JSON formatter
    - ship logs to App Insights / Azure Monitor
    """
    # We print a Python dict. Many log collectors can parse JSON-like payloads.
    # In production you would `json.dumps(...)` for strict JSON.
    print({"event": event, "cid": cid, **k})


# -----------------------------
# MCP-style tools abstraction
# -----------------------------

# ToolFn is a "type alias": it describes what a tool function looks like.
# Callable[..., Any] means:
# - Callable = a function
# - ...      = it can accept any arguments
# - Any      = it can return anything
ToolFn = Callable[..., Any]


@dataclass
class Tool:
    """
    Tool is a simple object holding:
    - name: tool name string
    - fn:   the callable function to run

    This resembles how agent tool registries work:
    - The agent picks a tool by name
    - The system calls the underlying function with structured arguments
    """
    name: str         # tool name (example: "servicenow")
    fn: ToolFn        # the function that performs the tool action


# -----------------------------
# Tool 1: ServiceNow ticket creation via REST
# -----------------------------
def servicenow_ticket(cid: str, text: str) -> Dict[str, Any]:
    """
    Create a ServiceNow incident ticket.

    cid  = correlation id for tracing
    text = short description for the ticket

    Returns:
      Parsed JSON dict from ServiceNow response

    Production notes:
    - Prefer OAuth / token auth or Managed Identity integration if supported
    - Add retries for transient network errors
    - Add idempotency (don't create duplicate tickets on retry)
    """

    # Read the ServiceNow instance from environment variables.
    # Example value: "acme.service-now.com"
    instance = os.getenv("SERVICENOW_INSTANCE")

    # Read credentials from environment variables.
    # In production: you'd pull secrets from Azure Key Vault, not env vars.
    user = os.getenv("SERVICENOW_USER")
    pwd = os.getenv("SERVICENOW_PASS")

    # Build the full ServiceNow Table API endpoint for the "incident" table.
    # This is a standard ServiceNow REST endpoint pattern.
    url = f"https://{instance}/api/now/table/incident"

    # Log that we're about to call ServiceNow.
    # This is where you'd see requests in your monitoring tools.
    log("servicenow.request", cid, url=url)

    # Make an HTTP POST request to create an incident.
    # auth=(user, pwd) uses HTTP Basic Auth.
    # json=... makes requests send a JSON request body.
    r = requests.post(
        url,
        auth=(user, pwd),
        json={"short_description": text},  # minimal ticket payload
        timeout=20                          # always set timeout in production
    )

    # Log the HTTP status code (200/201 success, 4xx client error, 5xx server error).
    log("servicenow.response", cid, status=r.status_code)

    # If ServiceNow returned an error (4xx/5xx), raise an exception.
    # This makes failures obvious instead of silently passing.
    r.raise_for_status()

    # Convert JSON response text into a Python dict and return it.
    return r.json()


# -----------------------------
# Tool 2: Databricks query placeholder
# -----------------------------
def databricks_query(cid: str, sql: str) -> Dict[str, Any]:
    """
    Example tool for querying Databricks.

    This is a placeholder because real Databricks calls require:
    - workspace host
    - PAT token / Azure auth
    - SQL warehouse id or cluster
    - statement execution API call

    We still demonstrate the pattern:
    - log request
    - return a structured response
    """

    # Log the query request (never log sensitive data in real systems).
    log("databricks.query", cid, sql=sql)

    # Return a fake response to demonstrate shape.
    # In real code, you'd call Databricks REST APIs here.
    return {"rows": [], "note": "placeholder response"}


# -----------------------------
# Tool registry (like MCP tool server registry)
# -----------------------------

# Build a dictionary mapping tool name -> Tool object
TOOLS: Dict[str, Tool] = {
    "servicenow": Tool("servicenow", servicenow_ticket),  # register ServiceNow tool
    "databricks": Tool("databricks", databricks_query),   # register Databricks tool
}


def call_tool(name: str, cid: str, **kw: Any) -> Any:
    """
    A generic tool invoker.
    This mimics MCP-like behavior:
      - name: tool name
      - cid: correlation id
      - **kw: keyword arguments passed into the tool function

    Example:
      call_tool("servicenow", cid, text="please migrate bot X")
    """

    # Safety: ensure tool exists
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")

    # Log tool call
    log("tool.call", cid, tool=name, args=list(kw.keys()))

    # Execute the tool function and return its output
    return TOOLS[name].fn(cid, **kw)


# -----------------------------
# Azure Function-style entrypoint
# -----------------------------
def main(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    This resembles an Azure Function handler.
    In Azure Functions, you'd typically receive an HTTP request,
    parse JSON, then process it.

    req: dict representing request JSON body

    Expected input keys (for demo):
      - "question": user query / task description
      - "similarity": float (0..1) from retrieval similarity score
    """

    # Create a brand-new correlation ID for this request.
    # In production, you might propagate an existing one from headers.
    cid = str(uuid.uuid4())

    # Log the request start.
    log("request.start", cid, keys=list(req.keys()))

    # -------------------------
    # RAG fallback gate
    # -------------------------
    # similarity is a stand-in for retrieval confidence.
    # In a real RAG pipeline:
    # - retrieve top-k chunks
    # - score them (cosine similarity)
    # - possibly re-rank
    # - decide if context is "good enough"
    similarity = float(req.get("similarity", 1.0))

    # If similarity is too low, we do NOT call the LLM.
    # Instead we return a safe response to avoid hallucination.
    if similarity < 0.7:
        log("rag.fallback", cid, similarity=similarity)
        return {
            "cid": cid,
            "answer": "No confident answer. Please refine your question or check access.",
            "used_llm": False,
        }

    # If similarity is acceptable, proceed.
    question = req.get("question", "No question provided")
    log("rag.ok", cid, similarity=similarity, question=question)

    # Create a ServiceNow ticket (example: request-driven automation).
    sn_result = call_tool("servicenow", cid, text=question)

    # Call Databricks (placeholder) to show multi-system workflow.
    dbx_result = call_tool("databricks", cid, sql="SELECT 1")

    # Return final response payload.
    # In real RAG, you'd now:
    # - assemble context
    # - call GPT model
    # - return answer + citations
    return {
        "cid": cid,
        "answer": "Workflow executed (ServiceNow + Databricks). LLM step omitted in this demo.",
        "used_llm": False,
        "servicenow": sn_result,
        "databricks": dbx_result,
    }


# -----------------------------
# Local run (only if executed directly)
# -----------------------------
if __name__ == "__main__":
    # A sample request to test locally.
    # similarity=0.85 means retrieval confidence is good (so workflow proceeds).
    sample_req = {
        "question": "Migrate Bot-42 from WebEx to Teams for BU=Finance",
        "similarity": 0.85,
    }

    # Run the main function and print the result.
    result = main(sample_req)
    print(result)
