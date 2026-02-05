# DECORATOR DEFINITION
import time                     # Used to measure execution time
from functools import wraps    # Preserves original function name/docs

# -----------------------------
# DECORATOR DEFINITION
# -----------------------------

def timer(func):
    """
    timer is the DECORATOR.
    It receives another function (func) as input.
    """

    print(f"\nInside timer decorator: func = {func}")

    @wraps(func)               # Keeps original function metadata
    def wrapper(*args, **kwargs):
        # *args  -> positional arguments passed to func
        # **kwargs -> keyword arguments passed to func

        print("\n 1. .........Starting timer...from wrapper")
        start = time.time()    # Record start time
        print(f"\nArguments were: args={args}, kwargs={kwargs}...from wrapper")
        print(f"start time is {start}...from wrapper")
        print("Before calling the original function...from wrapper")
        print("Calling the original function...from wrapper")

        result = func(*args, **kwargs)
        print(f"\nAfter calling the original function...from wrapper")
        print(f"Result obtained: {result}...from wrapper")
        # Call the ORIGINAL function here

        end = time.time()      # Record end time
        print("Stopping timer...from wrapper")
        print(f"end time is {end}...from wrapper")

        print(f"{func.__name__} took {end - start:.4f} seconds")
        # Print how long function ran

        return result          # Return original result

    return wrapper             # Return wrapped version of function


# -----------------------------
# APPLY DECORATOR
# -----------------------------

@timer                          # Same as: slow_add = timer(slow_add)
def slow_add(a, b):
    print("\n.................Adding numbers slowly...from actual function")
    time.sleep(1)              # Simulate slow work
    print(".............Done adding after sleeping 1 sec... printing this from actual function")
    return a + b               # Return sum


# -----------------------------
# EXECUTION
# -----------------------------
a=[3, 5, 7]
b=[2, 9, 4]
print(slow_add(a = a, b = b))          # Calls wrapper(), not slow_add() directly
