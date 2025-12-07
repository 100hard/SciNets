
import asyncio
import os
from dotenv import load_dotenv
from app.agents.literature import literature_node
from app.state import DiscoveryState
from langchain_core.runnables import RunnableConfig

load_dotenv()

async def test_node_directly():
    print("Testing literature_node direct invocation...")
    
    state = DiscoveryState(
        user_query="Test Query",
        goal="discover",
        lens="none",
        speculation="high",
        run_experiments=False
    )
    
    config = RunnableConfig(tags=["test"], callbacks=None)
    
    try:
        # Pass config as expected by the new signature
        res = await literature_node(state, config)
        print("Success!", res.keys())
    except TypeError as e:
        print(f"Signature Mismatch Error: {e}")
    except Exception as e:
        print(f"Execution Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_node_directly())
