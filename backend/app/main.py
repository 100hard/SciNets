
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from app.graph import create_graph
# from app.state import DiscoveryState
from pydantic import BaseModel, field_validator
import uuid
from typing import List, Dict, Any, Literal
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json
import asyncio

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

# In-memory session store for MVP
# We can't type hint with DiscoveryState here anymore if lazy loading
sessions: Dict[str, Any] = {}

class RunRequest(BaseModel):
    query: str
    goal: Literal["discover", "survey", "write"] = "discover"
    lens: str = "none"
    speculation: Literal["low", "medium", "high"] = "medium"
    run_experiments: bool = False
    documents: List[str] = []
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if len(v) > 2000:
            raise ValueError('Query must be under 2000 characters')
        if len(v) < 10:
            raise ValueError('Query must be at least 10 characters')
        return v


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
            # Use astream_events to get granular updates
            # 'v2' is required for valid astream_events output in newer langchain versions
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
                    # e.g., name="duckduckgo_search" or "read_paper"
                    yield f"data: {json.dumps({'type': 'log', 'data': f'[TOOL START] {name}: {str(data.get("input"))[:100]}...'})}\n\n"
                    await asyncio.sleep(0)
                
                elif kind == "on_tool_end":
                    output = str(data.get("output"))
                    # Truncate long outputs for the log stream
                    preview = output[:200] + "..." if len(output) > 200 else output
                    yield f"data: {json.dumps({'type': 'log', 'data': f'[TOOL END]   {name} -> {preview}'})}\n\n"
                    await asyncio.sleep(0)

                # 3. CHAT MODEL STREAMING (Thought Process)
                elif kind == "on_chat_model_stream":
                    # This gives token-by-token updates. Too noisy for now, maybe aggregate or skip.
                    # For "Deep Dive", users might like to see the "Thinking..." chunks.
                    chunk = data.get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                         yield f"data: {json.dumps({'type': 'log_chunk', 'data': chunk.content})}\n\n"

                # 4. FINAL RESULT
                elif kind == "on_chain_end" and name == "LangGraph":
                    final_state = event['data']['output']
                    if hasattr(final_state, "dict"):
                        res = final_state.dict()
                    elif isinstance(final_state, dict):
                        res = final_state
                    else:
                        res = {} 
                    yield f"data: {json.dumps({'type': 'result', 'data': res})}\n\n"

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
