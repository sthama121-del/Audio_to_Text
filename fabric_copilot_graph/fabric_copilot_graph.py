from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import TypedDict, Literal, Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

# ----------------------------
# Telemetry (simulate Azure Monitor / Fabric telemetry)
# Later: replace logger.info(...) with AppInsights exporter
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("telemetry")


def telemetry(node_name: str):
    def deco(fn):
        def wrapper(state: "AgentState"):
            t0 = time.time()
            out = fn(state)
            ms = int((time.time() - t0) * 1000)
            log.info(json.dumps(
                {
                    "node": node_name,
                    "ms": ms,
                    "route": state.get("route"),
                    "attempts": state.get("attempts", 0),
                    "user_query": state.get("user_query"),
                    "guardrail_pass": state.get("guardrail_pass"),
                    "guardrail_reason": state.get("guardrail_reason"),
                    "notes": out.get("notes"),
                    "error": out.get("error"),
                },
                ensure_ascii=False
            ))
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

    # outputs
    semantic_plan: Dict[str, Any]
    sql_query: str
    sql_rows: List[Dict[str, Any]]
    rag_hits: List[Dict[str, str]]  # [{id,title,content}]
    draft_answer: str
    final_answer: str

    guardrail_pass: bool
    guardrail_reason: str


# ----------------------------
# "Fabric Semantic Model" (mock)
# - maps business terms -> tables/columns/measures
# ----------------------------
SEMANTIC_MODEL = {
    "datasets": {
        "learning_adoption": {
            "description": "Program adoption by audience (Managers/Employees), by month",
            "table": "learning_adoption",
            "grain": "month, program_name, audience",
            "columns": ["month", "program_name", "audience", "adoption_pct"],
            "measures": {
                "avg_adoption_pct": "AVG(adoption_pct)",
                "max_adoption_pct": "MAX(adoption_pct)",
                "min_adoption_pct": "MIN(adoption_pct)",
            },
            "synonyms": ["adoption", "participation", "usage"]
        },
        "course_completions": {
            "description": "Course completions facts",
            "table": "course_completions",
            "grain": "employee_id, course_name, completed_on",
            "columns": ["employee_id", "course_name", "completed_on", "hours", "org"],
            "measures": {
                "completion_count": "COUNT(*)",
                "total_hours": "SUM(hours)",
            },
            "synonyms": ["completion", "completed", "hours", "training hours"]
        }
    },
    "global_synonyms": {
        "mgr": "Managers",
        "manager": "Managers",
        "managers": "Managers",
        "employee": "Employees",
        "employees": "Employees",
        "tlx": "TLX",
    }
}


# ----------------------------
# Data Layer: SQLite simulating Fabric Warehouse/Lakehouse SQL
# ----------------------------
DB_PATH = "tlx_learning.db"


def setup_sample_sqlite():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # adoption table (fits their “30% managers, 50% employees” discussion)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS learning_adoption (
      month TEXT,
      program_name TEXT,
      audience TEXT,
      adoption_pct REAL
    )
    """)
    cur.execute("DELETE FROM learning_adoption")

    adoption_rows = [
        ("2026-01", "Manager Excellence", "Managers", 30.0),
        ("2026-01", "Employee Upskilling", "Employees", 50.0),
        ("2026-02", "Manager Excellence", "Managers", 34.0),
        ("2026-02", "Employee Upskilling", "Employees", 52.0),
        ("2026-03", "Manager Excellence", "Managers", 36.0),
        ("2026-03", "Employee Upskilling", "Employees", 55.0),
    ]
    cur.executemany("INSERT INTO learning_adoption VALUES (?,?,?,?)", adoption_rows)

    # completions table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS course_completions (
      employee_id TEXT,
      course_name TEXT,
      completed_on TEXT,
      hours REAL,
      org TEXT
    )
    """)
    cur.execute("DELETE FROM course_completions")
    completions_rows = [
        ("E1001", "Responsible AI 101", "2026-01-03", 2.0, "HR"),
        ("E1002", "Responsible AI 101", "2026-01-20", 2.0, "Engineering"),
        ("E1003", "Fabric Fundamentals", "2026-01-15", 3.0, "Engineering"),
        ("E1004", "Copilot for Analysts", "2026-02-01", 1.0, "Product"),
        ("E1005", "Security Basics", "2026-02-04", 1.5, "Engineering"),
    ]
    cur.executemany("INSERT INTO course_completions VALUES (?,?,?,?,?)", completions_rows)

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
# Retrieval Layer: simulates Azure AI Search / OneLake vector store
# (keyword-based for portability; swap to Azure AI Search later)
# ----------------------------
RAG_DOCS = [
    {
        "id": "hr_policy_leave_001",
        "title": "Leave Policy - General",
        "content": "Employees may request PTO through the HR portal. Manager approval is required for PTO longer than 3 days. Do not include personal medical details in requests."
    },
    {
        "id": "hr_policy_external_staff_002",
        "title": "External Staff HR Support",
        "content": "External staff should use the staffing agency portal for payroll questions. HR Copilot can answer policy FAQs but must not reveal personal employee information."
    },
    {
        "id": "learning_programs_003",
        "title": "TLX Learning Programs Overview",
        "content": "TLX supports learning initiatives like Manager Excellence and Employee Upskilling. Adoption metrics are tracked monthly by audience."
    }
]


def keyword_search(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    q = query.lower()
    scored = []
    for d in RAG_DOCS:
        text = (d["title"] + " " + d["content"]).lower()
        score = sum(1 for w in re.findall(r"[a-zA-Z]+", q) if w in text)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for score, d in scored[:top_k] if score > 0]


# ----------------------------
# Router Logic (Parent Router Agent)
# - This is where you show the “structured vs unstructured” maturity
# ----------------------------
def classify_route(user_query: str) -> Route:
    q = user_query.lower()

    # if user asks for metric/aggregation -> SQL/Semantic
    agg_keywords = ["percent", "%", "adoption", "count", "total", "sum", "average", "avg", "how many", "trend", "by month"]
    if any(k in q for k in agg_keywords):
        # prefer semantic first, then SQL
        return "semantic"

    # policy/FAQ type -> RAG
    policy_keywords = ["policy", "pto", "leave", "compliance", "external staff", "staffing agency", "faq", "what is the rule"]
    if any(k in q for k in policy_keywords):
        return "rag"

    # default: semantic (because TLX is structured-data heavy per interview)
    return "semantic"


# ----------------------------
# Agents / Nodes
# ----------------------------
@telemetry("parent_router")
def parent_router(state: AgentState) -> AgentState:
    state["attempts"] = state.get("attempts", 0) + 1
    state["route"] = classify_route(state["user_query"])
    return state


@telemetry("semantic_agent")
def semantic_agent(state: AgentState) -> AgentState:
    """Find best dataset + propose measure/filter plan (metadata-aware)."""
    q = state["user_query"].lower()

    # pick dataset by keyword overlap with synonyms
    best = None
    best_score = -1
    for name, meta in SEMANTIC_MODEL["datasets"].items():
        score = sum(1 for s in meta.get("synonyms", []) if s in q)
        if score > best_score:
            best, best_score = (name, meta), score

    ds_name, ds_meta = best
    audience = None
    if "manager" in q or "managers" in q or "mgr" in q:
        audience = "Managers"
    if "employee" in q or "employees" in q:
        # if both appear, leave None to force clarification in fallback
        if audience and audience != "Employees":
            audience = None
        else:
            audience = "Employees"

    # choose a measure
    measure = None
    if "average" in q or "avg" in q:
        measure = list(ds_meta["measures"].keys())[0]
    elif "max" in q:
        measure = "max_adoption_pct" if "adoption_pct" in ds_meta["columns"] else None
    elif "min" in q:
        measure = "min_adoption_pct" if "adoption_pct" in ds_meta["columns"] else None
    else:
        # default measure based on dataset
        measure = "avg_adoption_pct" if ds_name == "learning_adoption" else "completion_count"

    state["semantic_plan"] = {
        "dataset": ds_name,
        "table": ds_meta["table"],
        "measure": measure,
        "audience": audience,
        "notes": f"Resolved dataset={ds_name}, measure={measure}, audience={audience}"
    }
    # after semantic plan, go to SQL agent for exact computation
    state["route"] = "sql"
    return state


@telemetry("sql_data_agent")
def sql_data_agent(state: AgentState) -> AgentState:
    plan = state.get("semantic_plan", {})
    table = plan.get("table")
    audience = plan.get("audience")
    measure = plan.get("measure")

    if not table:
        state["sql_query"] = ""
        state["sql_rows"] = []
        state["route"] = "fallback"
        return state

    # if audience is ambiguous and the query needs it, fallback
    q = state["user_query"].lower()
    if "adoption" in q and audience is None and ("manager" in q or "employee" in q):
        state["route"] = "fallback"
        return state

    # Build SQL safely (simple template)
    if table == "learning_adoption":
        agg_expr = SEMANTIC_MODEL["datasets"]["learning_adoption"]["measures"]["avg_adoption_pct"]
        where = ""
        if audience:
            where = f"WHERE audience = '{audience}'"
        sql = f"""
        SELECT program_name, audience, ROUND({agg_expr}, 2) AS avg_adoption_pct
        FROM learning_adoption
        {where}
        GROUP BY program_name, audience
        ORDER BY avg_adoption_pct DESC
        """.strip()

    elif table == "course_completions":
        # example aggregation: total hours or count
        if measure == "total_hours":
            agg_expr = "ROUND(SUM(hours), 2) AS total_hours"
        else:
            agg_expr = "COUNT(*) AS completion_count"
        sql = f"""
        SELECT course_name, {agg_expr}
        FROM course_completions
        GROUP BY course_name
        ORDER BY 2 DESC
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
    hits = keyword_search(state["user_query"], top_k=3)
    state["rag_hits"] = hits
    state["route"] = "synthesize"
    return state


@telemetry("response_synthesizer")
def response_synthesizer(state: AgentState) -> AgentState:
    q = state["user_query"]
    parts = [f"Question: {q}\n"]

    if state.get("sql_rows"):
        parts.append("Structured answer (from Warehouse/Semantic Model):")
        parts.append(json.dumps(state["sql_rows"], indent=2))
    if state.get("rag_hits"):
        parts.append("\nPolicy/knowledge context (from Search/Vector store):")
        for h in state["rag_hits"]:
            parts.append(f"- {h['title']}: {h['content']}")

    # Draft (simple deterministic synthesis)
    draft = "\n".join(parts)
    state["draft_answer"] = draft
    state["route"] = "guardrail"
    return state


# ----------------------------
# RAI Guardrail Layer
# - PII: basic patterns (email/phone/ssn)
# - Faithfulness: ensure any numeric claim is present in sources
# ----------------------------
PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",          # SSN-like
    r"\b\d{10}\b",                    # 10-digit phone-like
    r"\b[\w\.-]+@[\w\.-]+\.\w+\b",    # email
]

def extract_numbers(text: str) -> List[str]:
    return re.findall(r"\b\d+(?:\.\d+)?\b", text)


@telemetry("rai_guardrail")
def rai_guardrail(state: AgentState) -> AgentState:
    draft = state.get("draft_answer", "")

    # PII check
    for pat in PII_PATTERNS:
        if re.search(pat, draft):
            state["guardrail_pass"] = False
            state["guardrail_reason"] = "PII pattern detected"
            state["route"] = "fallback"
            return state

    # Faithfulness check (very lightweight):
    # any numbers in final answer must exist in retrieved context (sql_rows or rag_hits)
    source_blob = ""
    if state.get("sql_rows"):
        source_blob += json.dumps(state["sql_rows"])
    if state.get("rag_hits"):
        source_blob += " ".join(h["content"] for h in state["rag_hits"])

    nums_in_draft = set(extract_numbers(draft))
    nums_in_sources = set(extract_numbers(source_blob))

    # allow if draft contains numbers only from sources (or none)
    if not nums_in_draft.issubset(nums_in_sources):
        state["guardrail_pass"] = False
        state["guardrail_reason"] = "Ungrounded numeric claim detected"
        state["route"] = "fallback"
        return state

    state["guardrail_pass"] = True
    state["guardrail_reason"] = "Pass"
    state["final_answer"] = draft
    return state


@telemetry("fallback_loop")
def fallback_loop(state: AgentState) -> AgentState:
    # Keep it simple: produce a safe clarification or re-route.
    q = state["user_query"].lower()

    # Common interview scenario: manager vs employee adoption ambiguity
    if "adoption" in q and ("manager" in q or "employee" in q):
        state["final_answer"] = (
            "I can answer this precisely from the structured learning adoption tables.\n"
            "Quick clarification: do you want adoption for **Managers**, **Employees**, or both?\n"
            "Once confirmed, I’ll compute the exact numbers via SQL/semantic model (not via semantic search over rows)."
        )
        return state

    # If retrieval found nothing for a policy question, fail safely
    if any(k in q for k in ["policy", "pto", "leave", "compliance"]) and not state.get("rag_hits"):
        state["final_answer"] = (
            "I don’t have enough grounded policy text available in the current knowledge base to answer safely.\n"
            "If you point me to the policy document (or allow indexing it), I can answer with citations."
        )
        return state

    # Generic safe fallback
    state["final_answer"] = (
        "I’m missing enough grounded information to answer safely.\n"
        "Please specify which program/dataset (e.g., Manager Excellence vs Employee Upskilling) and the time range."
    )
    return state


# ----------------------------
# Build Graph (matches Gemini diagram)
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

    # Router -> (Semantic | SQL | RAG)
    def route_from_parent(state: AgentState):
        r = state["route"]
        if r == "semantic":
            return "semantic_agent"
        if r == "sql":
            return "sql_data_agent"
        if r == "rag":
            return "retrieval_agent"
        return "semantic_agent"

    g.add_conditional_edges("parent_router", route_from_parent)

    # Semantic -> SQL (already enforced)
    g.add_edge("semantic_agent", "sql_data_agent")

    # SQL -> Synth
    g.add_edge("sql_data_agent", "response_synthesizer")

    # RAG -> Synth
    g.add_edge("retrieval_agent", "response_synthesizer")

    # Synth -> Guardrail
    g.add_edge("response_synthesizer", "rai_guardrail")

    # Guardrail -> (END | fallback)
    def guardrail_branch(state: AgentState):
        return END if state.get("guardrail_pass") else "fallback_loop"

    g.add_conditional_edges("rai_guardrail", guardrail_branch)

    # Fallback -> END (or you can loop back to parent_router if user clarifies)
    g.add_edge("fallback_loop", END)

    return g.compile()


# ----------------------------
# Demo runner
# ----------------------------
if __name__ == "__main__":
    setup_sample_sqlite()
    app = build_graph()

    tests = [
        "What is the adoption percentage for managers vs employees for our learning programs?",
        "What is the average adoption for Manager Excellence?",
        "How many completions do we have per course?",
        "What is the PTO policy for external staff?"
    ]

    for t in tests:
        print("\n" + "=" * 80)
        result = app.invoke({"user_query": t})
        print(result["final_answer"])
