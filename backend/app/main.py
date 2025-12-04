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

@app.get("/")
async def root():
    return {"message": "Welcome to SciNets V2 API"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/run")
async def run_discovery(request: RunRequest):
    """
    Trigger the discovery loop.
    """
    graph = create_graph()
    initial_state = DiscoveryState(user_query=request.query)
    
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
