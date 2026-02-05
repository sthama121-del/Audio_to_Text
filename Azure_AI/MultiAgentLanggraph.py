# pip install -U langgraph langchain-core

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# 1) Define the "shared state" shape that every agent reads/writes
class AgentState(TypedDict):
    hops: int                 # how many transitions happened so far
    max_hops: int             # when to stop
    log: List[str]            # shared "memory"/trace of what each agent did
    payload: str              # a shared value agents can transform

# 2) Define 3 "agents" (nodes). Each reads state, appends to log, updates payload/hops.
def agent_a(state: AgentState) -> AgentState:
    new_payload = state["payload"] + " -> A"
    return {
        "hops": state["hops"] + 1,
        "payload": new_payload,
        "log": state["log"] + [f"A saw payload='{state['payload']}' and wrote '{new_payload}'"],
    }

def agent_b(state: AgentState) -> AgentState:
    new_payload = state["payload"] + " -> B"
    return {
        "hops": state["hops"] + 1,
        "payload": new_payload,
        "log": state["log"] + [f"B saw payload='{state['payload']}' and wrote '{new_payload}'"],
    }

def agent_c(state: AgentState) -> AgentState:
    new_payload = state["payload"] + " -> C"
    return {
        "hops": state["hops"] + 1,
        "payload": new_payload,
        "log": state["log"] + [f"C saw payload='{state['payload']}' and wrote '{new_payload}'"],
    }

# 3) Router decides whether to continue the loop or stop
def should_continue(state: AgentState) -> str:
    # If we've reached max_hops, stop the graph
    if state["hops"] >= state["max_hops"]:
        return "stop"
    return "go"

# 4) Build graph
g = StateGraph(AgentState)

g.add_node("A", agent_a)
g.add_node("B", agent_b)
g.add_node("C", agent_c)

# Start at A
g.set_entry_point("A")

# A -> B -> C is a fixed chain
g.add_edge("A", "B")
g.add_edge("B", "C")

# From C, decide: loop back to A or END
g.add_conditional_edges(
    "C",
    should_continue,
    {
        "go": "A",
        "stop": END,
    },
)

app = g.compile()

# 5) Run
initial_state: AgentState = {
    "hops": 0,
    "max_hops": 7,         # change this to see more/less passing
    "log": [],
    "payload": "start",
}

final_state = app.invoke(initial_state)

print("FINAL PAYLOAD:", final_state["payload"])
print("HOPS:", final_state["hops"])
print("\nTRACE LOG:")
for line in final_state["log"]:
    print("-", line)
