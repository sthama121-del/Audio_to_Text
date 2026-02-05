from typing import TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    extracted: str
    processed: str
    degraded: str
    iters: int
    max_iters: int

def agent_1(state: AgentState) -> AgentState:
    # Simulate empty extraction (so agent_2 will use fallback)
    return {"extracted": ""}

def agent_2(state: AgentState) -> AgentState:
    # Do some "processing" step
    if not state["extracted"]:
        processed = "fallback-used"
    else:
        processed = state["extracted"].upper()

    return {"processed": processed}

def agent_3(state: AgentState) -> AgentState:
    # "Finalize/validate" step, and decide degraded
    if state["processed"] == "fallback-used":
        degraded = f"someval-{state['iters']}"
    else:
        degraded = "no"

    # Increment loop counter here (or in agent_2 — just pick one place)
    return {"degraded": degraded, "iters": state["iters"] + 1}

def route_after_3(state: AgentState) -> str:
    # Stop looping after max_iters
    if state["iters"] >= state["max_iters"]:
        return "stop"
    return "loop"

# ---- Build graph ----
g = StateGraph(AgentState)

g.add_node("agent_1", agent_1)
g.add_node("agent_2", agent_2)
g.add_node("agent_3", agent_3)

g.set_entry_point("agent_1")
g.add_edge("agent_1", "agent_2")
g.add_edge("agent_2", "agent_3")

# Create the cycle: from agent_3 either go back to agent_2 or END
g.add_conditional_edges(
    "agent_3",
    route_after_3,
    {
        "loop": "agent_2",  # <-- cycle between agent_2 and agent_3
        "stop": END,
    },
)

app = g.compile()

initial_state: AgentState = {
    "extracted": "",
    "processed": "",
    "degraded": "",
    "iters": 0,
    "max_iters": 3,   # loop agent_2 <-> agent_3 three times
}

out = app.invoke(initial_state)
print(out)
