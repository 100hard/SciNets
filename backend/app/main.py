from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.graph import create_graph
from app.state import DiscoveryState
from pydantic import BaseModel
import uuid
from typing import List, Dict, Any

app = FastAPI(title="SciNets V2 API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for MVP
sessions: Dict[str, DiscoveryState] = {}

class RunRequest(BaseModel):
    query: str
    goal: str = "discover"
    lens: str = "none"
    speculation: str = "medium"
    run_experiments: bool = False

@app.post("/run_stream")
async def run_discovery_stream(request: RunRequest):
    """
    Trigger the discovery loop and stream events.
    """
    graph = create_graph()
    initial_state = DiscoveryState(
        user_query=request.query,
        goal=request.goal,
        lens=request.lens,
        speculation=request.speculation,
        run_experiments=request.run_experiments
    )
    
    try:
        # Invoke the graph
        final_state = await graph.ainvoke(initial_state)
        
        # Save session
        session_id = str(uuid.uuid4())
        
        # Handle LangGraph output which might be a dict or object
        if isinstance(final_state, dict):
            # If it's a dict, we need to be careful. 
            # DiscoveryState has default values, so we can use parse_obj or similar if it was a dict.
            # But wait, create_graph uses StateGraph(DiscoveryState), so it should return a dict representing the state.
            state_obj = DiscoveryState(**final_state)
        else:
            state_obj = final_state
            
        sessions[session_id] = state_obj
        
        # Return the state with the session ID
        response = state_obj.dict()
        response["session_id"] = session_id
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import StreamingResponse
import json
import asyncio

@app.post("/run_stream")
async def run_discovery_stream(request: RunRequest):
    """
    Trigger the discovery loop and stream events.
    """
    graph = create_graph()
    initial_state = DiscoveryState(
        user_query=request.query,
        goal=request.goal,
        lens=request.lens,
        speculation=request.speculation,
        run_experiments=request.run_experiments
    )

    async def event_generator():
        # Emit initial thinking event
        yield f"data: {json.dumps({'type': 'activity', 'data': {'agent': 'orchestrator', 'action': f'Starting discovery for: {request.query}', 'status': 'thinking'}})}\n\n"
        
        try:
            # Use astream_events to get granular updates
            # 'v2' is required for valid astream_events output in newer langchain versions
            async for event in graph.astream_events(initial_state, version="v1"):
                kind = event["event"]
                name = event.get("name", "")
                
                # Map graph nodes to agents and actions
                if kind == "on_chain_start" and name in ["literature", "hypothesis", "evidence", "experiment", "critique", "plan"]:
                    agent_map = {
                        "plan": "planner",
                        "literature": "scientist",
                        "hypothesis": "scientist",
                        "evidence": "critic",
                        "experiment": "scientist",
                        "critique": "critic"
                    }
                    action_map = {
                        "plan": "Structuring research plan...",
                        "literature": "Searching and reading literature...",
                        "hypothesis": "Generating and refining hypotheses...",
                        "evidence": "Verifying evidence and facts...",
                        "experiment": "Designing and running experiments...",
                        "critique": "Critiquing and validating findings..."
                    }
                    
                    activity = {
                        "agent": agent_map.get(name, "orchestrator"),
                        "action": action_map.get(name, f"Starting {name}..."),
                        "status": "thinking" if name in ["plan", "critique", "evidence"] else "reading" if name == "literature" else "building"
                    }
                    
                    yield f"data: {json.dumps({'type': 'activity', 'data': activity})}\n\n"

                # We can also yield partial graph updates if the node finishes
                elif kind == "on_chain_end" and name == "LangGraph":
                    # The whole graph finished
                    # The output is in event['data']['output']
                    final_state = event['data']['output']
                    # Yield the final result
                    # Note: final_state might be a dict or object depending on implementation
                    if hasattr(final_state, "dict"):
                        res = final_state.dict()
                    elif isinstance(final_state, dict):
                        res = final_state
                    else:
                        res = {} # Should not happen
                        
                    yield f"data: {json.dumps({'type': 'result', 'data': res})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/sessions")
def get_sessions():
    """Returns a list of past sessions (summary)."""
    return [
        {
            "id": sid,
            "query": s.user_query,
            "hypotheses_count": len(s.hypotheses),
            "experiments_count": len(s.experiments),
            "done": s.done
        }
        for sid, s in sessions.items()
    ]

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]
