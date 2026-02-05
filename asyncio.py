#asyncio
import asyncio


# -----------------------------
# ASYNC FUNCTION
# -----------------------------

async def task(name, delay):
    print(f"Starting {name}")

    await asyncio.sleep(delay)
    # await pauses THIS task but allows others to run

    print(f"Finished {name}")
    return name


# -----------------------------
# MAIN ASYNC CONTROLLER
# -----------------------------

async def main():

    # Run tasks concurrently
    results = await asyncio.gather(
        task("A", 2),
        task("B", 2),
        task("C", 2),
    )

    print(results)


# -----------------------------
# EVENT LOOP START
# -----------------------------

asyncio.run(main())
