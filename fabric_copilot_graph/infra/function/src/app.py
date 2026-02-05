from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from typing import TypedDict, Literal, Any, Dict, List

from langgraph.graph import StateGraph, END

# ----------------------------
# Telemetry (prints JSON; later swap to AppInsights)
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("telemetry")

def telemetry(node_name: str):
    def deco(fn):
        def wrapper(state: "AgentState"):
            t0 = time.time()
            out = fn(state)
            ms = int((time.time() - t0) * 1000)
            log.info(json.dumps({
                "node": node_name,
                "ms": ms,
                "route": state.get("route"),
                "attempts": state.get("attempts", 0),
                "guardrail_pass": state.get("guardrail_pass"),
                "guardrail_reason": state.get("guardrail_reason"),
            }, ensure_ascii=False))
            return out
        return wrapper
    return deco

# ----------------------------
# State
# ----------------------------
Route = Literal["semantic", "sql", "rag", "synthesize", "guardrail", "fallback"]

class AgentState(TypedDict, total=False):
    user_query: str
    route: Route
    attempts: int

    semantic_plan: Dict[str, Any]
    sql_query: str
    sql_rows: List[Dict[str, Any]]
    rag_hits: List[Dict[str, str]]

    draft_answer: str
    final_answer: str

    guardrail_pass: bool
    guardrail_reason: str

# ----------------------------
# Mock Fabric Semantic Model
# ----------------------------
SEMANTIC_MODEL = {
    "datasets": {
        "learning_adoption": {
            "table": "learning_adoption",
            "columns": ["month", "program_name", "audience", "adoption_pct"],
            "measures": {"avg_adoption_pct": "AVG(adoption_pct)"},
            "synonyms": ["adoption", "percent", "%", "participation", "usage"]
        },
        "course_completions": {
            "table": "course_completions",
            "columns": ["employee_id", "course_name", "completed_on", "hours", "org"],
            "measures": {"completion_count": "COUNT(*)", "total_hours": "SUM(hours)"},
            "synonyms": ["completion", "completed", "how many", "count", "hours", "total"]
        }
    }
}

# ----------------------------
# Data Layer: SQLite simulating Fabric Warehouse
# ----------------------------
DB_PATH = "tlx_learning.db"

def setup_sample_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS learning_adoption(
      month TEXT,
      program_name TEXT,
      audience TEXT,
      adoption_pct REAL
    )
    """)
    cur.execute("DELETE FROM learning_adoption")
    cur.executemany("INSERT INTO learning_adoption VALUES (?,?,?,?)", [
        ("2026-01", "Manager Excellence", "Managers", 30.0),
        ("2026-01", "Employee Upskilling", "Employees", 50.0),
        ("2026-02", "Manager Excellence", "Managers", 34.0),
        ("2026-02", "Employee Upskilling", "Employees", 52.0),
        ("2026-03", "Manager Excellence", "Managers", 36.0),
        ("2026-03", "Employee Upskilling", "Employees", 55.0),
    ])

    cur.execute("""
    CREATE TABLE IF NOT EXISTS course_completions(
      employee_id TEXT,
      course_name TEXT,
      completed_on TEXT,
      hours REAL,
      org TEXT
    )
    """)
    cur.execute("DELETE FROM course_completions")
    cur.executemany("INSERT INTO course_completions VALUES (?,?,?,?,?)", [
        ("E1001", "Responsible AI 101", "2026-01-03", 2.0, "HR"),
        ("E1002", "Responsible AI 101", "2026-01-20", 2.0, "Engineering"),
        ("E1003", "Fabric Fundamentals", "2026-01-15", 3.0, "Engineering"),
        ("E1004", "Copilot for Analysts", "2026-02-01", 1.0, "Product"),
        ("E1005", "Security Basics", "2026-02-04", 1.5, "Engineering"),
    ])

    conn.commit()
    conn.close()

def run_sql(query: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

# ----------------------------
# Retrieval Layer: simulates Azure AI Search / OneLake vector
# ----------------------------
RAG_DOCS = [
    {"id": "pol1", "title": "Leave Policy", "content": "PTO requests are submitted via HR portal. Avoid sharing medical details."},
    {"id": "pol2", "title": "External Staff HR Support", "content": "External staff use staffing agency portal for payroll questions. Do not reveal personal employee info."},
    {"id": "pol3", "title": "TLX Learning Overview", "content": "TLX tracks learning adoption monthly by audience (Managers/Employees) and program."},
]

def keyword_search(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    q = query.lower()
    scored = []
    for d in RAG_DOCS:
        text = (d["title"] + " " + d["content"]).lower()
        score = sum(1 for w in re.findall(r"[a-zA-Z]+", q) if w in text)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k] if s > 0]

# ----------------------------
# Router (Parent)
# ----------------------------
def classify_route(user_query: str) -> Route:
    q = user_query.lower()
    if any(k in q for k in ["adoption", "percent", "%", "count", "total", "sum", "avg", "average", "trend", "how many"]):
        return "semantic"
    if any(k in q for k in ["policy", "pto", "leave", "compliance", "external staff", "faq", "rule"]):
        return "rag"
    return "semantic"

# ----------------------------
# Nodes
# ----------------------------
@telemetry("parent_router")
def parent_router(state: AgentState) -> AgentState:
    state["attempts"] = state.get("attempts", 0) + 1
    state["route"] = classify_route(state["user_query"])
    return state

@telemetry("semantic_agent")
def semantic_agent(state: AgentState) -> AgentState:
    q = state["user_query"].lower()
    best_name = "learning_adoption"
    best_score = -1
    for name, meta in SEMANTIC_MODEL["datasets"].items():
        score = sum(1 for s in meta["synonyms"] if s in q)
        if score > best_score:
            best_name, best_score = name, score

    audience = None
    if "manager" in q:
        audience = "Managers"
    if "employee" in q:
        audience = None if audience else "Employees"

    measure = "avg_adoption_pct" if best_name == "learning_adoption" else "completion_count"

    state["semantic_plan"] = {
        "dataset": best_name,
        "table": SEMANTIC_MODEL["datasets"][best_name]["table"],
        "measure": measure,
        "audience": audience,
    }
    state["route"] = "sql"
    return state

@telemetry("sql_data_agent")
def sql_data_agent(state: AgentState) -> AgentState:
    plan = state.get("semantic_plan", {})
    table = plan.get("table")
    audience = plan.get("audience")
    q = state["user_query"].lower()

    if table == "learning_adoption":
        if ("manager" in q or "employee" in q) and audience is None:
            state["route"] = "fallback"
            return state

        where = f"WHERE audience = '{audience}'" if audience else ""
        sql = f"""
        SELECT program_name, audience, ROUND(AVG(adoption_pct), 2) AS avg_adoption_pct
        FROM learning_adoption
        {where}
        GROUP BY program_name, audience
        ORDER BY avg_adoption_pct DESC
        """.strip()

    elif table == "course_completions":
        sql = """
        SELECT course_name, COUNT(*) AS completion_count, ROUND(SUM(hours), 2) AS total_hours
        FROM course_completions
        GROUP BY course_name
        ORDER BY completion_count DESC
        """.strip()
    else:
        state["route"] = "fallback"
        return state

    state["sql_query"] = sql
    state["sql_rows"] = run_sql(sql)
    state["route"] = "synthesize"
    return state

@telemetry("retrieval_agent")
def retrieval_agent(state: AgentState) -> AgentState:
    state["rag_hits"] = keyword_search(state["user_query"], top_k=3)
    state["route"] = "synthesize"
    return state

@telemetry("response_synthesizer")
def response_synthesizer(state: AgentState) -> AgentState:
    parts = [f"Question: {state['user_query']}"]

    if state.get("sql_rows"):
        parts.append("Structured result (Warehouse/Semantic):")
        parts.append(json.dumps(state["sql_rows"], indent=2))

    if state.get("rag_hits"):
        parts.append("Retrieved policy context (Search/Vector):")
        for d in state["rag_hits"]:
            parts.append(f"- {d['title']}: {d['content']}")

    state["draft_answer"] = "\n".join(parts)
    state["route"] = "guardrail"
    return state

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
]

def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)

@telemetry("rai_guardrail")
def rai_guardrail(state: AgentState) -> AgentState:
    draft = state.get("draft_answer", "")

    for pat in PII_PATTERNS:
        if re.search(pat, draft):
            state["guardrail_pass"] = False
            state["guardrail_reason"] = "PII detected"
            state["route"] = "fallback"
            return state

    source_blob = ""
    if state.get("sql_rows"):
        source_blob += json.dumps(state["sql_rows"])
    if state.get("rag_hits"):
        source_blob += " ".join(d["content"] for d in state["rag_hits"])

    nums_draft = set(extract_numbers(draft))
    nums_src = set(extract_numbers(source_blob))
    if not nums_draft.issubset(nums_src):
        state["guardrail_pass"] = False
        state["guardrail_reason"] = "Ungrounded numeric claim"
        state["route"] = "fallback"
        return state

    state["guardrail_pass"] = True
    state["guardrail_reason"] = "Pass"
    state["final_answer"] = draft
    return state

@telemetry("fallback_loop")
def fallback_loop(state: AgentState) -> AgentState:
    q = state["user_query"].lower()
    if "adoption" in q and ("manager" in q or "employee" in q):
        state["final_answer"] = (
            "Clarification needed to answer precisely from structured data:\n"
            "Do you want adoption for **Managers**, **Employees**, or **both**?"
        )
        return state

    if any(k in q for k in ["policy", "pto", "leave"]) and not state.get("rag_hits"):
        state["final_answer"] = (
            "I don’t have enough grounded policy text available in the knowledge base to answer safely.\n"
            "Provide the policy doc to index, and I’ll answer with citations."
        )
        return state

    state["final_answer"] = "I’m missing enough grounded information to answer safely. Please specify dataset/time range."
    return state

# ----------------------------
# Build Graph (matches Gemini flow)
# ----------------------------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("parent_router", parent_router)
    g.add_node("semantic_agent", semantic_agent)
    g.add_node("sql_data_agent", sql_data_agent)
    g.add_node("retrieval_agent", retrieval_agent)
    g.add_node("response_synthesizer", response_synthesizer)
    g.add_node("rai_guardrail", rai_guardrail)
    g.add_node("fallback_loop", fallback_loop)

    g.set_entry_point("parent_router")

    def route_from_parent(state: AgentState):
        r = state["route"]
        if r == "rag":
            return "retrieval_agent"
        return "semantic_agent"

    g.add_conditional_edges("parent_router", route_from_parent)

    g.add_edge("semantic_agent", "sql_data_agent")
    g.add_edge("sql_data_agent", "response_synthesizer")
    g.add_edge("retrieval_agent", "response_synthesizer")
    g.add_edge("response_synthesizer", "rai_guardrail")

    def guardrail_branch(state: AgentState):
        return END if state.get("guardrail_pass") else "fallback_loop"

    g.add_conditional_edges("rai_guardrail", guardrail_branch)
    g.add_edge("fallback_loop", END)

    return g.compile()

def build_app():
    setup_sample_sqlite()
    return build_graph()
