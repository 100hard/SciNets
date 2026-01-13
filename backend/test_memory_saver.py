
import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    count: int

def node(state):
    return {"count": state["count"] + 1}

async def main():
    print("Testing MemorySaver on Windows...")
    try:
        memory = MemorySaver()
        workflow = StateGraph(State)
        workflow.add_node("node", node)
        workflow.set_entry_point("node")
        workflow.add_edge("node", END)
        app = workflow.compile(checkpointer=memory)
        
        config = {"configurable": {"thread_id": "1"}}
        inputs = {"count": 1}
        print("Running graph...")
        async for event in app.astream(inputs, config=config):
            print(event)
        
        print("Reading state...")
        state = await app.aget_state(config)
        print(f"State: {state}")
        print("Success!")
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
