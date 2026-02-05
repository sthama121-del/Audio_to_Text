# -----------------------------
# GENERATOR FUNCTION DEFINITION
# -----------------------------

def squares(n):
    """
    Step 0: Definition. Python sees 'yield' and marks this as a Generator.
    """

    # [CONTROL FLOW: 2, 5, 8...]
    # The function "wakes up" here whenever next() is called.
    for i in range(n):         
        
        # [CONTROL FLOW: 3, 6, 9...] 
        # 'yield' is the PAUSE button. It sends the value out and 
        # FREEZES the function right here until the next call.
        print(f"   (Generator: Yielding {i*i} and pausing...)")
        yield i * i           


# -----------------------------
# USING THE GENERATOR
# -----------------------------

# [CONTROL FLOW: 1] - Initialization
# This prepares the object but does NOT enter the function body yet.
print("[CONTROL FLOW: 1] Initializing generator")
gen = squares(5)
print(f"\nGenerator object created: {gen}")              

# [CONTROL FLOW: 2 & 3] - First manual call
# Control jumps into 'squares', runs the loop, yields, and jumps BACK.
print("\n[CONTROL FLOW: 4] Requesting first value with next()")
val1 = next(gen)
print(f"Main Script received: {val1}")

# [CONTROL FLOW: 5 & 6] - Second manual call
# Control jumps BACK to where it paused, finishes i=0, starts i=1, and yields.
print("\n[CONTROL FLOW: 7] Requesting second value with next()")
val2 = next(gen)
print(f"Main Script received: {val2}")

# [CONTROL FLOW: 8 & 9] - The for-loop
# The 'for' loop automatically handles the 'next()' calls until exhaustion.
print("\n[CONTROL FLOW: 10] Starting for-loop for remaining values")
for value in gen:
    # Control jumps back and forth for i=2, i=3, and i=4.
    print(f"Loop received: {value}")

# [CONTROL FLOW: 11] - Completion
# When i=5, the loop inside 'squares' ends. It sends a StopIteration signal.
print("\n[CONTROL FLOW: 11] Generator exhausted. Script finished.")