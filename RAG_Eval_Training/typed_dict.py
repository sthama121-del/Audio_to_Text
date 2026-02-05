from typing import TypedDict
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    extracted: str
    processed: str

def agent_1(state: AgentState) -> AgentState:
    # Imagine extraction failed
    return {
        "extracted": "",   # EMPTY but valid
    }

def agent_2(state: AgentState) -> AgentState:
    if not state["extracted"]:
        value = "fallback-used"
    else:
        value = state["extracted"].upper()

    return {
        "processed": value
    }

g = StateGraph(AgentState)

g.add_node("extract", agent_1)
g.add_node("process", agent_2)

g.set_entry_point("extract")
g.add_edge("extract", "process")
g.add_edge("process", END)

app = g.compile()

out = app.invoke({
    "extracted": "",
    "processed": ""
})

print(out)
