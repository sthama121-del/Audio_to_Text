# 1. Import 'wraps' to preserve the identity of 'main_task'
from functools import wraps

# --- PHASE 1: THE DEFINITION (Happens once when script starts) ---
def coordinate(extra_func):
    """
    Step 0: The 'coordinate' factory is defined. 
    It holds 'say_hello' in its closure.
    """
    def decorator(main_func):
        """Step 0.1: The decorator is defined and receives 'main_task'."""
        
        @wraps(main_func)
        def wrapper(*args, **kwargs):
            # --- START EXECUTION FLOW ---
            
            print(f"\n[CONTROL FLOW: 1] --> Entering 'wrapper'")
            print(f"[DEBUG] Received args: {args}, kwargs: {kwargs}")
            
            print("[CONTROL FLOW: 2] --> Jumping to 'extra_func' (say_hello)")
            extra_func() # This triggers the jump
            
            print("[CONTROL FLOW: 4] --> Returned to 'wrapper' from 'extra_func'")
            print("[CONTROL FLOW: 5] --> Jumping to 'main_func' (actual slow_add logic)")
            
            # The control 'dives' into the original function here
            result = main_func(*args, **kwargs) 
            
            print(f"[CONTROL FLOW: 7] --> Returned to 'wrapper' with result: {result}")
            print("[CONTROL FLOW: 8] --> Finalizing wrapper logic and returning to main script")
            
            return result
            
        return wrapper
    return decorator

# --- THE HELPERS ---
def say_hello():
    """This is the 'extra' task logic."""
    print("   [CONTROL FLOW: 3] inside 'say_hello' function!")

# --- APPLYING THE DECORATOR ---
@coordinate(say_hello)
def main_task(message):
    """This is the core business logic."""
    print(f"   [CONTROL FLOW: 6] inside 'main_task' processing: {message}")
    return "SUCCESS"

# --- PHASE 2: THE EXECUTION (The trigger) ---
print("--- SCRIPT START ---")
# When we call this, we follow steps 1 through 8 above
final_status = main_task("Optimizing Snowflake") 
print(f"--- SCRIPT END: {final_status} ---")