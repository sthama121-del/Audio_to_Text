#mcp_databricks_server.py
"""
MCP Server: Databricks Jobs/Run Logs Access (Azure Databricks)

INTERVIEW EXPLAINER:
- MCP lets an agent call tools via a standard interface (tools/resources/prompts).
- Here we expose Databricks run metadata and run output (logs-ish) as MCP tools.
- The agent (CrewAI/LangChain) doesn't need to know REST details.

Refs:
- MCP server pattern: FastMCP / python-sdk docs. :contentReference[oaicite:5]{index=5}
- Databricks Jobs API: runs/get and getRunOutput. :contentReference[oaicite:6]{index=6}
"""

import os
import time
from typing import Any, Dict, Optional, List
import requests

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("azure-databricks-logs")

# ---- Required env vars (keep these out of code in real life) ----
# DATABRICKS_HOST = "https://adb-<workspace-id>.<region>.azuredatabricks.net"
# DATABRICKS_TOKEN = "dapi...."  (PAT) or AAD token depending on setup
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    # Don't crash at import time in some runtimes; tools will validate.
    pass


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}


def _request_with_retries(method: str, url: str, *, params=None, json=None, timeout=20, max_tries=3) -> Dict[str, Any]:
    """
    Basic retry with exponential backoff.
    INTERVIEW NOTE:
    - Logs APIs can transiently fail (network/429/5xx).
    - Retrying with backoff is standard in production monitoring.
    """
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            resp = requests.request(method, url, headers=_headers(), params=params, json=json, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Transient error {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json() if resp.text else {}
        except Exception as e:
            last_err = str(e)
            if attempt < max_tries:
                time.sleep(2 ** (attempt - 1))  # 1s,2s,4s
            else:
                break
    raise RuntimeError(f"Databricks API failed after {max_tries} tries. Last error: {last_err}")


@mcp.tool()
def list_recent_runs(job_id: int, limit: int = 10, active_only: bool = False) -> Dict[str, Any]:
    """
    List recent runs for a Databricks Job.

    Args:
      job_id: Databricks Job ID
      limit: max number of runs
      active_only: if True, filter for active/running runs

    Returns:
      A dict containing recent runs metadata.
    """
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        raise RuntimeError("Set DATABRICKS_HOST and DATABRICKS_TOKEN env vars.")

    # Databricks Jobs API 2.2: /api/2.2/jobs/runs/list (or listruns)
    # Some workspaces use /api/2.1 - adjust if needed.
    url = f"{DATABRICKS_HOST}/api/2.2/jobs/runs/list"
    params = {"job_id": job_id, "limit": limit}
    if active_only:
        params["active_only"] = "true"
    return _request_with_retries("GET", url, params=params)


@mcp.tool()
def get_run_metadata(run_id: int) -> Dict[str, Any]:
    """
    Get metadata for a single run.
    Uses: GET /api/2.2/jobs/runs/get  :contentReference[oaicite:7]{index=7}
    """
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        raise RuntimeError("Set DATABRICKS_HOST and DATABRICKS_TOKEN env vars.")

    url = f"{DATABRICKS_HOST}/api/2.2/jobs/runs/get"
    params = {"run_id": run_id}
    return _request_with_retries("GET", url, params=params)


@mcp.tool()
def get_run_output(run_id: int) -> Dict[str, Any]:
    """
    Get output for a single run (often includes notebook output / error text).
    Databricks docs: jobs/getRunOutput endpoint. :contentReference[oaicite:8]{index=8}
    """
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        raise RuntimeError("Set DATABRICKS_HOST and DATABRICKS_TOKEN env vars.")

    url = f"{DATABRICKS_HOST}/api/2.2/jobs/runs/get-output"
    params = {"run_id": run_id}
    return _request_with_retries("GET", url, params=params)


@mcp.tool()
def extract_error_text(run_id: int) -> Dict[str, Any]:
    """
    Convenience tool: returns a normalized error string from run output/metadata.
    INTERVIEW NOTE:
    - 'Normalization' makes downstream LLM prompts consistent.
    """
    meta = get_run_metadata(run_id)
    out = get_run_output(run_id)

    # Common fields
    state = (meta.get("state") or {})
    life_cycle = state.get("life_cycle_state")
    result_state = state.get("result_state")
    state_msg = state.get("state_message", "")

    # get-output can include error or notebook_output, depending on task type
    err = out.get("error") or ""
    err_trace = out.get("error_trace") or ""
    notebook_output = (out.get("notebook_output") or {}).get("result", "")

    combined = "\n".join([str(x) for x in [life_cycle, result_state, state_msg, err, err_trace, notebook_output] if x])
    return {"run_id": run_id, "error_text": combined[:8000]}  # cap to keep prompts safe


if __name__ == "__main__":
    # stdio transport is the most common for local/agent usage
    mcp.run()
