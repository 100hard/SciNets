
import asyncio
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

# 1. Define State
class State(TypedDict):
    val: str

# 2. Define Tool
@tool
def my_tool(msg: str):
    """Log tool."""
    return f"Logged: {msg}"

# 3. Define Node WITHOUT Config (Current State)
async def node_without_config(state: State):
    print("Executing node_without_config...")
    # This invoke is detached from the tracer?
    my_tool.invoke("Hello from detached tool")
    return {"val": "done"}

# 4. Define Node WITH Config (Proposed Fix)
from langchain_core.runnables import RunnableConfig
async def node_with_config(state: State, config: RunnableConfig):
    print("Executing node_with_config...")
    # Pass config explicitly
    my_tool.invoke("Hello from attached tool", config=config)
    return {"val": "done"}

async def test_tracing():
    # TEST 1: WITHOUT CONFIG
    print("\n--- TEST 1: Node WITHOUT Config ---")
    workflow = StateGraph(State)
    workflow.add_node("node", node_without_config)
    workflow.set_entry_point("node")
    workflow.add_edge("node", END)
    app = workflow.compile()
    
    events_found = False
    async for event in app.astream_events({"val": "init"}, version="v1"):
        if event["event"] == "on_tool_start":
            print(f"CAPTURED EVENT: {event['name']} -> {event['data'].get('input')}")
            events_found = True
            
    if not events_found:
        print("FAIL: No tool events captured.")

    # TEST 2: WITH CONFIG
    print("\n--- TEST 2: Node WITH Config ---")
    workflow2 = StateGraph(State)
    workflow2.add_node("node", node_with_config)
    workflow2.set_entry_point("node")
    workflow2.add_edge("node", END)
    app2 = workflow2.compile()
    
    events_found = False
    async for event in app2.astream_events({"val": "init"}, version="v1"):
        if event["event"] == "on_tool_start":
            print(f"CAPTURED EVENT: {event['name']} -> {event['data'].get('input')}")
            events_found = True
            
    if events_found:
        print("SUCCESS: Tool event captured!")

if __name__ == "__main__":
    asyncio.run(test_tracing())
