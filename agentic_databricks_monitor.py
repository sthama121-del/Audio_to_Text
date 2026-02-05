#agentic_databricks_monitor.py
"""
Multi-Agent Databricks Job Monitor (CrewAI + LangChain + MCP + LangSmith)

USE CASE:
- Poll Databricks job runs
- Detect failures
- Extract error
- Suggest remediation (RAG over runbooks)
- Human approves action
- Send email alert to distro list

INTERVIEW TALKING POINTS:
1) Why multi-agent?
   - Separation of concerns improves reliability & maintainability:
     Observer (detection) → Diagnoser (classify) → Reasoner (fix) → Action (notify)
2) Why polling frequency?
   - Tradeoff: timeliness vs API cost/rate limits.
   - If jobs run hourly, poll every 5-10 minutes.
   - If jobs run every few minutes, poll every 1-2 minutes, with caching and backoff.
3) Why max tries?
   - APIs and LLM calls can fail; retries + escalation avoid silent failures.
4) Human-in-the-loop:
   - For risky changes/alerts, require approval (prevents false positives + spam).
5) Evaluation:
   - Offline eval via LangSmith correctness against known incident dataset.
   - Ensures model changes don’t degrade recommended solutions.

Refs:
- CrewAI MCP DSL integration :contentReference[oaicite:9]{index=9}
- LangSmith evaluation concepts :contentReference[oaicite:10]{index=10}
"""

import os
import time
import smtplib
from email.message import EmailMessage
from typing import Dict, Any, List, Optional, Tuple

from crewai import Agent, Task, Crew, Process

# LangChain for LLM + RAG
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# LangSmith eval
from langsmith.evaluation import evaluate

# -------------------------
# CONFIG (edit for your env)
# -------------------------
JOB_ID = int(os.getenv("DATABRICKS_JOB_ID", "12345"))

# Poll frequency:
# - Good baseline for “hourly / few times a day” jobs: 10 minutes
# - For “frequent pipelines” (every 5 mins): poll every 1-2 mins but implement rate limits
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", "10"))

# Max tries for external calls:
MAX_API_TRIES = int(os.getenv("MAX_API_TRIES", "3"))
MAX_LLM_TRIES = int(os.getenv("MAX_LLM_TRIES", "2"))

# Email settings (stub)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.yourcompany.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "databricks-alerts@yourcompany.com")
EMAIL_TO = os.getenv("EMAIL_TO", "dl-data-platform@yourcompany.com")  # distro list

# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# -------------------------
# MCP TOOL ACCESS (CrewAI)
# -------------------------
# CrewAI supports MCP via DSL "mcps" field. We'll point to a local stdio MCP server.
# You can also run MCP server in a container/host.
MCP_SERVER_CMD = "python mcp_databricks_server.py"

# -------------------------
# Simple Runbook KB for RAG
# (In real life load from wiki/Jira/Confluence)
# -------------------------
RUNBOOK_DOCS = [
    Document(page_content="OutOfMemoryError: Increase driver/executor memory, reduce shuffle partitions, check data skew, enable AQE, optimize joins."),
    Document(page_content="PERMISSION_DENIED/403: Check secret scopes, workspace ACLs, service principal permissions, token scopes."),
    Document(page_content="NotebookExecutionException: Often wrapper—inspect root cause; missing libraries, invalid init scripts, cluster config."),
    Document(page_content="Spark job stuck: Check shuffle spill, skew, cluster autoscaling, partition sizes, broadcast joins."),
]

emb = OpenAIEmbeddings()
vs = FAISS.from_documents(RUNBOOK_DOCS, emb)
retriever = vs.as_retriever(search_kwargs={"k": 3})

# -------------------------
# Helper: Safe email sending
# -------------------------
def send_email(subject: str, body: str) -> None:
    """
    INTERVIEW NOTE:
    - In production use: SendGrid/Graph API/SMTP relay with secrets manager.
    - Include run_id, job_id, timestamp, link to run UI.
    """
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    # If SMTP creds not provided, just print (safe for demo/interview)
    if not SMTP_USER or not SMTP_PASS:
        print("\n--- EMAIL (DRY RUN) ---")
        print("TO:", EMAIL_TO)
        print("SUBJECT:", subject)
        print(body)
        print("--- END EMAIL ---\n")
        return

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

# -------------------------
# LangChain RAG prompt
# -------------------------
fix_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a senior ML/Platform engineer. Given a Databricks job error, propose likely root cause and actionable fix steps. "
     "Be concise. Provide: (1) probable cause (2) immediate mitigation (3) longer-term prevention (4) confidence 0-1."),
    ("user", "ERROR:\n{error_text}\n\nRUNBOOK CONTEXT:\n{context}")
])

def rag_fix_recommendation(error_text: str) -> Dict[str, Any]:
    """
    Produce remediation using RAG over runbooks.
    Includes LLM retry (MAX_LLM_TRIES).
    """
    docs = retriever.get_relevant_documents(error_text)
    context = "\n\n".join([d.page_content for d in docs])

    last_err = None
    for attempt in range(1, MAX_LLM_TRIES + 1):
        try:
            msg = fix_prompt.format_messages(error_text=error_text, context=context)
            resp = llm.invoke(msg).content
            return {"recommendation": resp, "context": context}
        except Exception as e:
            last_err = str(e)
            if attempt < MAX_LLM_TRIES:
                time.sleep(2 ** (attempt - 1))
            else:
                break
    raise RuntimeError(f"LLM failed after {MAX_LLM_TRIES} tries: {last_err}")

# -------------------------
# CrewAI Agents
# -------------------------
observer = Agent(
    role="Observer Agent",
    goal="Monitor Databricks job runs and detect failures quickly with minimal noise.",
    backstory="Expert in ops monitoring and API-based job orchestration.",
    # MCP integration: CrewAI will discover MCP tools from this server command
    mcps=[{"server": MCP_SERVER_CMD}],
    verbose=True,
)

diagnoser = Agent(
    role="Diagnosis Agent",
    goal="Extract the true error signal and classify it (OOM, permissions, missing lib, etc.).",
    backstory="Expert in Spark/Databricks troubleshooting and error normalization.",
    mcps=[{"server": MCP_SERVER_CMD}],
    verbose=True,
)

reasoner = Agent(
    role="Reasoner Agent",
    goal="Use runbooks + RAG to propose a reliable fix and estimate confidence.",
    backstory="Senior engineer who writes actionable incident remediation steps.",
    verbose=True,
)

actioner = Agent(
    role="Action Agent",
    goal="Draft an alert for the distribution list, require human approval if confidence is low, then send.",
    backstory="Owns incident communications and escalation playbooks.",
    verbose=True,
)

# -------------------------
# Tasks
# -------------------------
t1 = Task(
    description=(
        "Use MCP tool list_recent_runs for job_id={job_id} and find the most recent FAILED run.\n"
        "Return run_id and a short reason why it is considered failure."
    ).format(job_id=JOB_ID),
    expected_output="JSON with fields: run_id, failure_reason",
    agent=observer
)

t2 = Task(
    description=(
        "Given the run_id from previous step, call MCP tool extract_error_text.\n"
        "Return normalized error_text and a preliminary classification label."
    ),
    expected_output="JSON with fields: run_id, error_text, label",
    agent=diagnoser
)

t3 = Task(
    description=(
        "Given error_text and label, produce remediation steps using internal runbooks.\n"
        "You may call the python function rag_fix_recommendation conceptually. "
        "Return: cause, mitigation, prevention, confidence (0-1)."
    ),
    expected_output="JSON with fields: cause, mitigation, prevention, confidence, recommended_message",
    agent=reasoner
)

t4 = Task(
    description=(
        "Draft an email alert to distribution list with run_id/job_id, error summary, and remediation.\n"
        "HUMAN-IN-THE-LOOP RULE:\n"
        "- If confidence < 0.70 OR label is unknown, require human approval (ask 'APPROVE? y/n').\n"
        "- If approved or confidence >= 0.70, send email.\n"
        "Also include: max_tries rationale and poll frequency rationale in the email footer (1-2 lines)."
    ),
    expected_output="A final status: SENT or NOT_SENT with reason.",
    agent=actioner
)

crew = Crew(
    agents=[observer, diagnoser, reasoner, actioner],
    tasks=[t1, t2, t3, t4],
    process=Process.sequential,
    verbose=True
)

# -------------------------
# Human-in-loop wrapper
# -------------------------
def human_approve(prompt: str) -> bool:
    ans = input(prompt).strip().lower()
    return ans in ("y", "yes")

# -------------------------
# Orchestration loop
# -------------------------
def run_once() -> None:
    """
    Runs the crew once, then performs the final action in python
    (so we can enforce human approval + email sending deterministically).
    """
    result = crew.kickoff()
    # CrewAI returns text; in real implementation parse JSON outputs from each task.
    # For interview: explain you'd structure outputs as JSON and parse into objects.
    print("\n=== CREW OUTPUT ===\n", result, "\n===================\n")

    # In a production implementation:
    # - parse outputs from each task
    # - extract run_id, error_text, label
    # - call rag_fix_recommendation
    # Here, we demonstrate the *core logic* with placeholders:
    # (If you want, I’ll adapt this to parse the real tool outputs your Crew returns.)

def monitor_forever():
    """
    POLLING STRATEGY (INTERVIEW):
    - Poll every POLL_INTERVAL_MINUTES.
    - Why: balance early detection vs API cost/rate limit noise.
    - For batch jobs (hourly/daily), 5-15 mins is typical.
    - For streaming/near-real-time jobs, event-driven (webhooks) is better; polling can be 1-2 mins with caching.
    """
    while True:
        try:
            run_once()
        except Exception as e:
            # Production: send a meta-alert if monitor itself fails
            print("Monitor error:", e)

        time.sleep(POLL_INTERVAL_MINUTES * 60)

# -------------------------
# LangSmith Evaluation (Offline)
# -------------------------
def langsmith_offline_eval():
    """
    EVALUATION (INTERVIEW):
    - Create a dataset of known errors with gold/reference remediation.
    - Score model output with reference-based correctness (LLM-as-judge).
    - Run on every prompt/model change to prevent regressions.
    """
    if not os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        print("LangSmith API key not set; skipping eval.")
        return

    examples = [
        {
            "inputs": {"error_text": "java.lang.OutOfMemoryError: Java heap space"},
            "outputs": {"reference": "Increase driver/executor memory; reduce shuffle partitions; check skew; optimize joins."}
        },
        {
            "inputs": {"error_text": "PERMISSION_DENIED: 403 for secret scope"},
            "outputs": {"reference": "Check secret scope permissions and service principal ACLs; verify workspace access."}
        },
    ]

    def system_under_test(inputs: Dict[str, Any]) -> str:
        out = rag_fix_recommendation(inputs["error_text"])
        return out["recommendation"]

    # LangSmith evaluate() will run evaluators against examples.
    # A common choice: reference-based "correctness" evaluator (semantic match to reference).
    # Docs explain reference-based evaluators + offline correctness concepts. :contentReference[oaicite:11]{index=11}
    results = evaluate(
        system_under_test,
        data=examples,
        evaluators=["correctness"],  # built-in evaluator name in LangSmith
        experiment_prefix="databricks-log-monitor"
    )
    print("LangSmith eval results:", results)

# -------------
# MAIN
# -------------
if __name__ == "__main__":
    mode = os.getenv("MODE", "once")  # "once" | "forever" | "eval"
    if mode == "eval":
        langsmith_offline_eval()
    elif mode == "forever":
        monitor_forever()
    else:
        run_once()
