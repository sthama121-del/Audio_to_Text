cd ~/Azure_AI/

pip install langgraph --user

cat > langgraph_workflow.py << 'EOF'
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define state that flows between agents
class WorkflowState(TypedDict):
    detection_result: str
    analysis_result: str
    notification: str

# Define agent functions
def agent_1(state: WorkflowState):
    print("🔍 Agent 1: Detecting...")
    # Call your Azure Function
    detection = "Found 1 failed job: run_id 770449065051703"
    return {"detection_result": detection}

def agent_2(state: WorkflowState):
    print("🔬 Agent 2: Analyzing...")
    detection = state["detection_result"]  # Get from previous agent
    # Call your Azure Function with run_id
    analysis = f"Root cause analysis for: {detection}"
    return {"analysis_result": analysis}

def agent_3(state: WorkflowState):
    print("📧 Agent 3: Notifying...")
    detection = state["detection_result"]
    analysis = state["analysis_result"]
    notification = f"Alert: {detection}\n{analysis}"
    return {"notification": notification}

# Build the graph
workflow = StateGraph(WorkflowState)

# Add nodes (agents)
workflow.add_node("detector", agent_1)
workflow.add_node("analyzer", agent_2)
workflow.add_node("notifier", agent_3)

# Add edges (connections) - THIS DEFINES THE ORDER!
workflow.set_entry_point("detector")      # Start here
workflow.add_edge("detector", "analyzer")  # detector → analyzer
workflow.add_edge("analyzer", "notifier")  # analyzer → notifier
workflow.add_edge("notifier", END)         # notifier → END

# Compile
app = workflow.compile()

# Run
result = app.invoke({})
print("\n✅ Final Result:")
print(result["notification"])
EOF
```

---

### **Visual Representation of LangGraph:**
```
┌──────────────┐
│   START      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  detector    │  (Agent 1)
│  node        │
└──────┬───────┘
       │ edge
       ▼
┌──────────────┐
│  analyzer    │  (Agent 2)
│  node        │
└──────┬───────┘
       │ edge
       ▼
┌──────────────┐
│  notifier    │  (Agent 3)
│  node        │
└──────┬───────┘
       │ edge
       ▼
┌──────────────┐
│    END       │
└──────────────┘