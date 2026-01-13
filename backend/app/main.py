
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from app.graph import create_graph
# from app.state import DiscoveryState
from pydantic import BaseModel, field_validator
import uuid
from typing import List, Dict, Any, Literal, Optional
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json
import asyncio
import sys
import logging

# Configure logging for consistent, unbuffered output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("scinets")

# Windows asyncio fix for subprocess compatibility
# Prevents [Errno 22] Invalid argument when using run_in_executor
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
    # REMOVED: run_experiments - experiments are now user-triggered only via /run_experiment
    documents: List[str] = []
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        if len(v) > 2000:
            raise ValueError('Query must be under 2000 characters')
        if len(v) < 10:
            raise ValueError('Query must be at least 10 characters')
        return v


class ExperimentRequest(BaseModel):
    """Request for user-triggered experiment (POST /run_experiment)."""
    hypothesis_id: str
    hypothesis_text: str
    intent: Literal["validate_direction", "probe_sensitivity", "stress_test"] = "validate_direction"
    data_source: Literal["synthetic", "public_dataset", "user_provided"] = "synthetic"
    seed: int = 42  # Fixed by default


class SearchPapersRequest(BaseModel):
    """Request for pre-fetching candidate papers before full discovery."""
    query: str
    max_papers: int = 15


class CandidatePaper(BaseModel):
    """Paper returned by /search_papers for curation."""
    id: str
    title: str
    year: int
    venue: str
    abstract: str
    rationale: str = "Matched search query"


@app.post("/search_papers")
async def search_papers(request: SearchPapersRequest):
    """
    Fetch candidate papers BEFORE full discovery.
    
    This allows users to curate which papers go into the analysis.
    Returns a list of candidate papers with metadata.
    """
    logger.info(f"[SearchPapers] Fetching papers for: {request.query}")
    
    try:
        from app.tools.openalex import search_papers as openalex_search
        
        # Use query directly - LLM refinement was causing 500 errors
        search_query = request.query
        
        logger.info(f"[SearchPapers] Searching OpenAlex for: '{search_query}'")
        
        # Search OpenAlex
        papers = await openalex_search(search_query, limit=request.max_papers)
        
        logger.info(f"[SearchPapers] Found {len(papers)} papers")
        
        # Convert to CandidatePaper format
        candidates = []
        for paper in papers:
            # The 'abstract' field from openalex.py is actually the inverted index
            # Reconstruct abstract from inverted index
            inverted_index = paper.get("abstract")  # This IS the inverted index
            abstract = ""
            if inverted_index and isinstance(inverted_index, dict):
                word_positions = []
                for word, positions in inverted_index.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                abstract = " ".join(w for _, w in word_positions)
            
            # host_venue from openalex.py is already extracted as a string
            venue = paper.get("host_venue", "Unknown venue")
            if not venue:
                venue = "Unknown venue"
            
            candidates.append(CandidatePaper(
                id=paper.get("id", f"paper-{len(candidates)}"),
                title=paper.get("title") or "Untitled",
                year=paper.get("publication_year") or 2024,
                venue=venue,
                abstract=abstract[:500] if abstract else "No abstract available",
                rationale="Relevant to search query"
            ))
        
        return {"papers": [c.dict() for c in candidates], "refined_query": search_query}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"[SearchPapers] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Paper search failed: {e}")


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
            # REMOVED: run_experiments - experiments are user-triggered only
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
        
        # TEMPORARY: Disabled checkpointer, so no thread_id needed
        # import uuid
        # thread_id = str(uuid.uuid4())
        # config = {"configurable": {"thread_id": thread_id}}
        
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

                elif kind == "on_tool_start":
                    # e.g., name="duckduckgo_search" or "read_paper"
                    tool_input = str(data.get('input'))[:100]
                    yield f"data: {json.dumps({'type': 'log', 'data': f'[TOOL START] {name}: {tool_input}...'})}\n\n"
                    await asyncio.sleep(0)
                
                elif kind == "on_tool_end":
                    output = str(data.get('output'))
                    # Truncate long outputs for the log stream
                    preview = output[:200] + '...' if len(output) > 200 else output
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
            import traceback
            traceback.print_exc()
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
            # REMOVED: experiments_count - experiments are now separate
            "done": s.done
        }
        for sid, s in sessions.items()
    ]

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


# =============================================================================
# User-triggered Experiment Endpoint (NOT part of discovery)
# =============================================================================
@app.post("/run_experiment")
async def run_experiment(request: ExperimentRequest):
    """
    User-triggered experiment for a specific hypothesis.
    
    This is NOT part of the default discovery pipeline.
    Experiments are optional, user-initiated exploratory tools.
    """
    try:
        from app.experiment_graph import create_experiment_graph
        from app.state import ExperimentState
        
        graph = create_experiment_graph()
        experiment_state = ExperimentState(
            hypothesis_id=request.hypothesis_id,
            hypothesis_text=request.hypothesis_text,
            intent=request.intent,
            data_source=request.data_source,
            seed=request.seed
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Experiment init failed: {e}")

    async def experiment_event_generator():
        yield f"data: {json.dumps({'type': 'activity', 'data': {'agent': 'scientist', 'action': f'Starting experiment for hypothesis: {request.hypothesis_id}', 'status': 'building'}})}\n\n"
        await asyncio.sleep(0.1)
        
        try:
            async for event in graph.astream_events(experiment_state, version="v1"):
                kind = event["event"]
                name = event.get("name", "")
                data = event.get("data", {})
                
                if kind == "on_chain_start" and name in ["experiment", "localized_critique"]:
                    action_map = {
                        "experiment": "Running exploratory experiment...",
                        "localized_critique": "Analyzing experiment results..."
                    }
                    activity = {
                        "agent": "scientist",
                        "action": action_map.get(name, f"Running {name}..."),
                        "status": "building"
                    }
                    yield f"data: {json.dumps({'type': 'activity', 'data': activity})}\n\n"
                    await asyncio.sleep(0)
                    
                elif kind == "on_chain_end" and name == "LangGraph":
                    final_state = event['data']['output']
                    if hasattr(final_state, "dict"):
                        res = final_state.dict()
                    elif isinstance(final_state, dict):
                        res = final_state
                    else:
                        res = {}
                    yield f"data: {json.dumps({'type': 'experiment_result', 'data': res})}\n\n"
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        experiment_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

