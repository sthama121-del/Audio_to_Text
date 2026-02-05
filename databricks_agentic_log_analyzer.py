#databricks_agentic_log_analyzer.py
import os
import uuid
import requests
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# -------------------------
# 1) Structured logging
# -------------------------
def log(event: str, cid: str, **k: Any) -> None:
    print({"event": event, "cid": cid, **k})


# -------------------------
# 2) Databricks API client (minimal)
# -------------------------
class DatabricksClient:
    """
    Minimal Databricks REST wrapper.
    In real usage you'd add retries + token refresh.
    """
    def __init__(self, host: str, token: str):
        self.host = host.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    def list_runs(self, job_id: int, limit: int = 5) -> Dict[str, Any]:
        url = f"{self.host}/api/2.1/jobs/runs/list"
        r = requests.get(url, headers=self.headers, params={"job_id": job_id, "limit": limit}, timeout=20)
        r.raise_for_status()
        return r.json()

    def get_run(self, run_id: int) -> Dict[str, Any]:
        url = f"{self.host}/api/2.1/jobs/runs/get"
        r = requests.get(url, headers=self.headers, params={"run_id": run_id}, timeout=20)
        r.raise_for_status()
        return r.json()

    def get_run_output(self, run_id: int) -> Dict[str, Any]:
        url = f"{self.host}/api/2.1/jobs/runs/get-output"
        r = requests.get(url, headers=self.headers, params={"run_id": run_id}, timeout=20)
        r.raise_for_status()
        return r.json()


# -------------------------
# 3) Agent payload (what moves between agents)
# -------------------------
@dataclass
class JobEvent:
    """
    This is the shared message/payload passed across agents.
    """
    correlation_id: str
    job_id: int
    run_id: int
    status: str
    error_text: str = ""
    classification: str = ""
    recommendations: List[str] = None


# -------------------------
# 4) Agent A: Collector Agent
# -------------------------
def collector_agent(dbx: DatabricksClient, job_id: int) -> JobEvent:
    """
    Pull latest run info + output logs from Databricks.
    """
    cid = str(uuid.uuid4())
    log("collector.start", cid, job_id=job_id)

    runs = dbx.list_runs(job_id=job_id, limit=1)
    latest = runs["runs"][0]
    run_id = latest["run_id"]

    run_detail = dbx.get_run(run_id)
    life_cycle = run_detail.get("state", {}).get("life_cycle_state", "UNKNOWN")
    result_state = run_detail.get("state", {}).get("result_state", "UNKNOWN")

    output = dbx.get_run_output(run_id)
    # Databricks returns different fields; this is a best-effort extraction
    error_text = (output.get("error") or "") + "\n" + (output.get("error_trace") or "")

    status = f"{life_cycle}/{result_state}"
    log("collector.fetched", cid, run_id=run_id, status=status)

    return JobEvent(
        correlation_id=cid,
        job_id=job_id,
        run_id=run_id,
        status=status,
        error_text=error_text.strip(),
        recommendations=[]
    )


# -------------------------
# 5) Agent B: Analyzer Agent
# -------------------------
def analyzer_agent(evt: JobEvent) -> JobEvent:
    """
    Classify failure based on known patterns (fast + deterministic).
    """
    log("analyzer.start", evt.correlation_id, run_id=evt.run_id)

    t = evt.error_text.lower()

    if "outofmemory" in t or "java.lang.outofmemoryerror" in t:
        evt.classification = "OOM"
    elif "permission" in t or "unauthorized" in t or "forbidden" in t:
        evt.classification = "AUTH"
    elif "timeout" in t:
        evt.classification = "TIMEOUT"
    elif "sparkexception" in t or "executor" in t:
        evt.classification = "SPARK_RUNTIME"
    elif evt.error_text.strip() == "":
        evt.classification = "NO_ERROR_TEXT"
    else:
        evt.classification = "UNKNOWN"

    log("analyzer.classified", evt.correlation_id, classification=evt.classification)
    return evt


# -------------------------
# 6) Agent C: Recommender Agent
# -------------------------
def recommender_agent(evt: JobEvent) -> JobEvent:
    """
    Add remediation steps based on classification.
    """
    log("recommender.start", evt.correlation_id, classification=evt.classification)

    recs = []
    if evt.classification == "OOM":
        recs = [
            "Increase driver/executor memory or use autoscaling.",
            "Check for data skew (salting, repartitioning).",
            "Persist intermediate DataFrames carefully; avoid large collect()."
        ]
    elif evt.classification == "AUTH":
        recs = [
            "Validate Databricks PAT / Azure AD permissions.",
            "Check workspace/cluster policy permissions and secret scopes."
        ]
    elif evt.classification == "TIMEOUT":
        recs = [
            "Increase job timeout; verify network/private endpoints.",
            "Check external dependency latency (storage, APIs)."
        ]
    elif evt.classification == "SPARK_RUNTIME":
        recs = [
            "Inspect executor logs and Spark UI stage failures.",
            "Tune partitions; verify library dependency versions."
        ]
    else:
        recs = [
            "Collect full run output and cluster logs.",
            "Check recent changes in data volume, schema, and dependencies."
        ]

    evt.recommendations = recs
    log("recommender.done", evt.correlation_id, rec_count=len(recs))
    return evt


# -------------------------
# 7) Agent D: Ticketing Agent (ServiceNow placeholder)
# -------------------------
def ticketing_agent(evt: JobEvent) -> Dict[str, Any]:
    """
    Create a ServiceNow ticket (here we simulate).
    In production, call ServiceNow REST Table API.
    """
    log("ticketing.start", evt.correlation_id, run_id=evt.run_id)

    # Simulated ticket id
    ticket = f"INC{evt.run_id}"

    summary = {
        "ticket": ticket,
        "job_id": evt.job_id,
        "run_id": evt.run_id,
        "status": evt.status,
        "classification": evt.classification,
        "recommendations": evt.recommendations[:3],  # keep it short
    }

    log("ticketing.created", evt.correlation_id, ticket=ticket)
    return summary


# -------------------------
# 8) Orchestrator: ties agents together
# -------------------------
def orchestrate(job_id: int) -> Dict[str, Any]:
    """
    This function is your 'Coordinator Agent'.
    It calls the other agents in sequence (direct-call pattern).
    """
    host = os.getenv("DATABRICKS_HOST", "")
    token = os.getenv("DATABRICKS_TOKEN", "")
    if not host or not token:
        raise RuntimeError("Missing DATABRICKS_HOST or DATABRICKS_TOKEN")

    dbx = DatabricksClient(host, token)

    evt = collector_agent(dbx, job_id)
    evt = analyzer_agent(evt)
    evt = recommender_agent(evt)

    ticket = ticketing_agent(evt)
    return ticket


if __name__ == "__main__":
    # Example: analyze job_id=123
    print(orchestrate(job_id=123))
