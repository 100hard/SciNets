
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json
import asyncio
import uvicorn
import os
import sys

# Ensure backend dir is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

load_dotenv()

app = FastAPI(title="SciNets V2 API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: Dict[str, Any] = {}

def custom_serializer(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)

class RunRequest(BaseModel):
    query: str
    goal: str = "discover"
    lens: str = "none"
    speculation: str = "medium"
    run_experiments: bool = False
    documents: List[str] = []

@app.post("/run_stream")
async def run_discovery_stream(request: RunRequest):
    """
    Trigger the discovery loop and stream events.
    """
    try:
        from app.graph import create_graph
        from app.state import DiscoveryState
        
        graph = create_graph()
        initial_state = DiscoveryState(
            user_query=request.query,
            goal=request.goal,
            lens=request.lens,
            speculation=request.speculation,
            run_experiments=request.run_experiments,
            documents=request.documents
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Init failed: {e}")

        


    async def event_generator():
        # Emit initial thinking event
        yield f"data: {json.dumps({'type': 'activity', 'data': {'agent': 'orchestrator', 'action': f'Starting discovery for: {request.query}', 'status': 'thinking'}})}\n\n"
        await asyncio.sleep(0.1) # Force flush
        
        try:
            # Accumulator for final result
            accumulated_state = {}
            
            # Use astream_events to get granular updates
            async for event in graph.astream_events(initial_state, version="v1"):
                kind = event["event"]
                name = event.get("name", "")
                data = event.get("data", {})
                
                # 1. MAJOR NODE UPDATES (High Level)
                if kind == "on_chain_start" and name in ["literature", "hypothesis", "evidence", "experiment", "critique", "plan"]:
                    agent_map = {
                        "plan": "planner", "literature": "scientist", "hypothesis": "scientist",
                        "evidence": "critic", "experiment": "scientist", "critique": "critic"
                    }
                    action_map = {
                        "plan": "Structuring research plan...",
                        "literature": "Searching and reading literature...",
                        "hypothesis": "Generating and refining hypotheses...",
                        "evidence": "Verifying evidence and facts...",
                        "experiment": "Designing and running experiments...",
                        "critique": "Critiquing and validating findings..."
                    }
                    if name in agent_map:
                        activity = {
                            "agent": agent_map[name],
                            "action": action_map.get(name, f"Starting {name}..."),
                            "status": "thinking" if name in ["plan", "critique", "evidence"] else "reading" if name == "literature" else "building"
                        }
                        yield f"data: {json.dumps({'type': 'activity', 'data': activity})}\n\n"
                        await asyncio.sleep(0) # Yield control

                # 2. TOOL & LOG UPDATES (In-Depth)
                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'log', 'data': f'[TOOL START] {name}: {str(data.get("input"))[:100]}...'})}\n\n"
                    await asyncio.sleep(0)
                
                elif kind == "on_tool_end":
                    output = str(data.get("output"))
                    preview = output[:200] + "..." if len(output) > 200 else output
                    yield f"data: {json.dumps({'type': 'log', 'data': f'[TOOL END]   {name} -> {preview}'})}\n\n"
                    await asyncio.sleep(0)

                # 3. CHAT MODEL STREAMING
                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                         yield f"data: {json.dumps({'type': 'log_chunk', 'data': chunk.content})}\n\n"

                elif kind == "on_chain_end":
                    batch_output = event.get('data', {}).get('output', {})
                    if hasattr(batch_output, "dict"):
                        update_dict = batch_output.model_dump()
                    elif isinstance(batch_output, dict):
                        update_dict = batch_output
                    else:
                        update_dict = {}
                    
                    # SMART ACCUMULATION: Only pick up known state keys
                    # This filters out "messages" from sub-agents and other noise
                    relevant_keys = ["plan", "literature", "hypotheses", "evidence", "experiments", "critique", "user_query", "concept_graph", "domain_tags"]
                    
                    has_update = False
                    for key in relevant_keys:
                        if key in update_dict:
                            accumulated_state[key] = update_dict[key]
                            has_update = True
                            
                    if has_update:
                        # Ensure user_query/goal are present
                        if "user_query" not in accumulated_state:
                            accumulated_state["user_query"] = request.query
                        
                        # Yield update if we have at least the initial plan or structure
                        yield f"data: {json.dumps({'type': 'result', 'data': accumulated_state}, default=custom_serializer)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

@app.get("/sessions")
def get_sessions():
    return [{"id": k, "query": "session"} for k in sessions]

if __name__ == "__main__":
    print("Starting SciNets Server on Port 8005...")
    uvicorn.run(app, host="127.0.0.1", port=8005, log_level="info")
