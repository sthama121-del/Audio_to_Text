#enterprise_ai_workflow_demo.py
"""
Single-file "interview demo" that shows patterns for:
- Blue/Green routing flag
- Azure Function-style entrypoint
- ServiceNow ticket creation via REST
- App Insights / Azure Monitor style structured logging
- Microsoft Graph call example (send Teams channel message placeholder)
- Databricks + Snowflake query placeholders
- MCP-style tool registry (conceptual)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import requests


# -----------------------------
# Config (Blue/Green switch)
# -----------------------------
ACTIVE_COLOR = os.getenv("ACTIVE_COLOR", "blue")  # "blue" or "green"
SERVICENOW_INSTANCE = os.getenv("SERVICENOW_INSTANCE", "example.service-now.com")
SERVICENOW_USER = os.getenv("SERVICENOW_USER", "")
SERVICENOW_PASS = os.getenv("SERVICENOW_PASS", "")

# In real Azure: use Managed Identity to acquire Graph token.
GRAPH_TENANT_ID = os.getenv("GRAPH_TENANT_ID", "")
GRAPH_CLIENT_ID = os.getenv("GRAPH_CLIENT_ID", "")
GRAPH_CLIENT_SECRET = os.getenv("GRAPH_CLIENT_SECRET", "")

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_TOKEN = os.getenv("SNOWFLAKE_TOKEN", "")


# -----------------------------
# Observability helpers
# (App Insights / Azure Monitor pattern: structured logs + correlation id)
# -----------------------------
def log_event(event_name: str, correlation_id: str, **props: Any) -> None:
    payload = {
        "ts": time.time(),
        "event": event_name,
        "correlation_id": correlation_id,
        "active_color": ACTIVE_COLOR,
        **props,
    }
    print(json.dumps(payload, ensure_ascii=False))


# -----------------------------
# ServiceNow integration
# -----------------------------
def create_servicenow_ticket(
    correlation_id: str,
    short_description: str,
    description: str,
    category: str = "inquiry",
    priority: str = "3",
) -> Dict[str, Any]:
    """
    Creates a ServiceNow incident via Table API.
    Interview notes:
      - Use correlation_id for tracing.
      - In production add idempotency: store a mapping (correlation_id -> incident sys_id)
        to avoid duplicates on retries.
    """
    url = f"https://{SERVICENOW_INSTANCE}/api/now/table/incident"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    body = {
        "short_description": short_description,
        "description": f"[corr_id={correlation_id}] {description}",
        "category": category,
        "priority": priority,
    }

    log_event("servicenow.create_ticket.request", correlation_id, url=url)

    resp = requests.post(url, auth=(SERVICENOW_USER, SERVICENOW_PASS), headers=headers, json=body, timeout=20)
    resp.raise_for_status()

    data = resp.json()
    log_event("servicenow.create_ticket.success", correlation_id, incident=data.get("result", {}).get("number"))
    return data


# -----------------------------
# Microsoft Graph (example)
# -----------------------------
def get_graph_token_client_credentials() -> str:
    """
    Minimal client-credentials OAuth2 token flow.
    In Azure: prefer Managed Identity when available to avoid secrets.
    """
    token_url = f"https://login.microsoftonline.com/{GRAPH_TENANT_ID}/oauth2/v2.0/token"
    form = {
        "client_id": GRAPH_CLIENT_ID,
        "client_secret": GRAPH_CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": "https://graph.microsoft.com/.default",
    }
    resp = requests.post(token_url, data=form, timeout=20)
    resp.raise_for_status()
    return resp.json()["access_token"]


def graph_send_teams_channel_message(
    correlation_id: str,
    team_id: str,
    channel_id: str,
    message_html: str,
) -> None:
    """
    Example Graph call.
    In real world you'd store team_id/channel_id in config and handle permissions carefully.
    """
    token = get_graph_token_client_credentials()
    url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    body = {"body": {"contentType": "html", "content": message_html}}

    log_event("graph.teams_message.request", correlation_id, team_id=team_id, channel_id=channel_id)
    resp = requests.post(url, headers=headers, json=body, timeout=20)
    resp.raise_for_status()
    log_event("graph.teams_message.success", correlation_id)


# -----------------------------
# Databricks / Snowflake placeholders
# -----------------------------
def databricks_sql_query(correlation_id: str, sql: str) -> Dict[str, Any]:
    """
    Placeholder pattern (Databricks SQL Statement Execution API exists).
    For interview: show you know it should be REST + token + workspace host.
    """
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        # In production you'd validate at runtime and raise a clear error.
        log_event("databricks.config_missing", correlation_id)
        return {"error": "Databricks not configured"}

    url = f"{DATABRICKS_HOST}/api/2.0/sql/statements"
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
    body = {"statement": sql, "warehouse_id": os.getenv("DATABRICKS_WAREHOUSE_ID", "")}

    log_event("databricks.sql.request", correlation_id, url=url)
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def snowflake_query(correlation_id: str, sql: str) -> Dict[str, Any]:
    """
    Placeholder: Snowflake has REST and Python connector options.
    Here we just demonstrate pattern + traceability.
    """
    if not SNOWFLAKE_ACCOUNT or not SNOWFLAKE_TOKEN:
        log_event("snowflake.config_missing", correlation_id)
        return {"error": "Snowflake not configured"}

    # Example only — real endpoints vary by auth method and org setup.
    log_event("snowflake.sql.request", correlation_id, account=SNOWFLAKE_ACCOUNT)
    return {"ok": True, "rows": []}


# -----------------------------
# MCP-style tool registry (conceptual)
# -----------------------------
ToolFn = Callable[..., Any]

@dataclass
class Tool:
    name: str
    fn: ToolFn
    description: str

TOOLS: Dict[str, Tool] = {
    "servicenow_create_ticket": Tool(
        name="servicenow_create_ticket",
        fn=create_servicenow_ticket,
        description="Create ServiceNow incident (Table API).",
    ),
    "databricks_sql_query": Tool(
        name="databricks_sql_query",
        fn=databricks_sql_query,
        description="Run SQL on Databricks via REST API (placeholder).",
    ),
    "snowflake_query": Tool(
        name="snowflake_query",
        fn=snowflake_query,
        description="Run SQL on Snowflake (placeholder).",
    ),
}

def call_tool(tool_name: str, correlation_id: str, **kwargs: Any) -> Any:
    """
    MCP-like idea: agent chooses a tool by name, sends structured args,
    platform executes with governance + audit logs.
    """
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")

    log_event("tool.call", correlation_id, tool=tool_name, args=list(kwargs.keys()))
    return TOOLS[tool_name].fn(correlation_id=correlation_id, **kwargs)


# -----------------------------
# Azure Function-style entrypoint
# -----------------------------
def main(req_body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Imagine this is triggered by HTTP request (Azure Function).
    We:
      - create correlation_id
      - run a blue/green aware workflow
      - create ServiceNow ticket
      - optionally notify Teams (Graph)
    """
    correlation_id = req_body.get("correlation_id") or str(uuid.uuid4())
    log_event("request.received", correlation_id, body_keys=list(req_body.keys()))

    # Example: blue/green cutover gate
    if req_body.get("target_color") and req_body["target_color"] != ACTIVE_COLOR:
        # In real deployment, traffic manager/gateway controls routing; this is only a demo flag.
        log_event("route.mismatch", correlation_id, target=req_body["target_color"])
        return {"status": 409, "message": f"Routed to {ACTIVE_COLOR}, expected {req_body['target_color']}"}

    # Example: create incident
    incident = call_tool(
        "servicenow_create_ticket",
        correlation_id,
        short_description=req_body.get("short_description", "AI workflow request"),
        description=req_body.get("description", "No description provided"),
    )

    # Example: query Databricks/Snowflake if needed
    if req_body.get("databricks_sql"):
        dbx = call_tool("databricks_sql_query", correlation_id, sql=req_body["databricks_sql"])
        log_event("databricks.sql.result", correlation_id, has_error=("error" in dbx))

    if req_body.get("snowflake_sql"):
        sn = call_tool("snowflake_query", correlation_id, sql=req_body["snowflake_sql"])
        log_event("snowflake.sql.result", correlation_id, rows=len(sn.get("rows", [])))

    return {"status": 200, "correlation_id": correlation_id, "incident": incident}


if __name__ == "__main__":
    # Local test run
    sample = {
        "short_description": "Test request from agent workflow",
        "description": "Please provision access for Business Unit X",
        "target_color": "blue",
        "databricks_sql": "SELECT 1",
    }
    print(main(sample))
