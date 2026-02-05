#generator2.py
import time
from functools import wraps

# --- AGENT STEPS (The Generator) ---
def ai_agent_process(query):
    """
    This is a Generator. It 'streams' the agent's progress.
    """
    # Step 1: Simulated Analysis
    yield {"step": "Analysing", "data": f"Breaking down query: {query}"}
    time.sleep(1) # Simulating LLM processing

    # Step 2: Simulated Database Retrieval (Snowflake/Vector DB)
    yield {"step": "Retrieving", "data": "Fetching 50TB+ optimization logs from Snowflake..."}
    time.sleep(1.5) 

    # Step 3: Simulated Reasoning
    yield {"step": "Reasoning", "data": "Applying anomaly detection models to IoT sensor data."}
    time.sleep(1)

    # Step 4: Final Answer
    yield {"step": "Final Answer", "data": "Optimization complete. Suggested cluster shift: +2 nodes."}

# --- THE EXECUTION ---
print("--- STARTING AGENTIC STREAM ---")

# We initialize the generator (Lazy Evaluation)
agent_stream = ai_agent_process("Optimize Houston Data Center")

# We iterate through the stream, displaying thoughts in real-time
for update in agent_stream:
    # Each 'yield' brings us back here immediately
    print(f"[{update['step']}] >> {update['data']}")

print("--- AGENT FINISHED ---")