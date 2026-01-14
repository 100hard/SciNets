

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json
import asyncio
import uvicorn
import os
import sys
import time

# Structured Logging Setup
from app.logging_config import setup_logging, get_logger
import structlog

# Ensure backend dir is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

load_dotenv()

# Setup logging (JSON in production, colored in dev)
log_level = os.getenv("LOG_LEVEL", "INFO")
json_logs = os.getenv("JSON_LOGS", "false").lower() == "true"
setup_logging(log_level=log_level, json_logs=json_logs)

log = get_logger(__name__)

app = FastAPI(title="SciNets V2 API")

log.info("scinets_startup", version="2.0", log_level=log_level, json_mode=json_logs)


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
    if isinstance(obj, list):
        return [custom_serializer(i) for i in obj]
    return str(obj)

# Concurrency Control
active_threads: set[str] = set()

class RunRequest(BaseModel):
    query: str
    goal: str = "discover"
    lens: str = "none"
    speculation: str = "medium"
    run_experiments: bool = False
    documents: List[str] = []
    thread_id: str | None = None # For resuming sessions
    feedback: str | None = None # User feedback when resuming
    mock: bool = False

class ExperimentRequest(BaseModel):
    thread_id: str
    hypothesis_id: str
    hypothesis_text: str # Required for experiment context
    plan_id: Optional[str] = None
    intent: str = "stress_test"
    data_source: str = "synthetic"

@app.post("/run_experiment")
async def run_experiment_stream(request: ExperimentRequest):
    """
    Trigger execution of a specific experiment plan.
    Resumes the thread, skips completed steps (via idempotency checks), and runs Experiment Agent.
    """
    thread_id = request.thread_id
    request_log = log.bind(request_id=f"exp_{str(uuid.uuid4())[:8]}", thread_id=thread_id)
    
    try:
        from app.experiment_graph import create_experiment_graph
        graph = create_experiment_graph()
        config = {"configurable": {"thread_id": thread_id}}
        
        request_log.info("starting_experiment_execution", 
                         hypothesis=request.hypothesis_id, 
                         plan=request.plan_id)

        # Update State to trigger execution mode
        # The experiment graph relies on the state being populated with the right ID
        graph.update_state(config, {
            "hypothesis_id": request.hypothesis_id,
            "hypothesis_text": request.hypothesis_text,
            "intent": request.intent,
            "data_source": request.data_source,
            "experiment_result": None # Reset result
        })
        
    except Exception as e:
        import traceback
        request_log.error("experiment_init_failed", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Init failed: {e}")

    async def event_generator():
        data = {'agent': 'orchestrator', 'action': 'Initializing experiment environment...', 'status': 'thinking'}
        yield f"data: {json.dumps({'type': 'activity', 'data': data, 'thread_id': thread_id})}\n\n"
        await asyncio.sleep(0.1)

        try:
            accumulated_state = {}
            start_times = {}
            
            # Stream events (resume from current state)
            async for event in graph.astream_events(None, config=config, version="v1"):
                kind = event["event"]
                name = event.get("name", "")
                data = event.get("data", {})
                
                # REUSE LOGGING LOGIC FROM DISCOVERY
                # 1. MAJOR NODE UPDATES
                if kind == "on_chain_start" and name in ["literature", "hypothesis", "evidence", "experiment", "critique", "plan"]:
                    agent_map = {
                        "plan": "planner", "literature": "scientist", "hypothesis": "scientist",
                        "evidence": "critic", "experiment": "scientist", "critique": "critic"
                    }
                    action_map = {
                        "plan": "Structuring research plan...",
                        "literature": "Analyzing literature & context...",
                        "hypothesis": "Generating scientific hypotheses...",
                        "evidence": "Evaluating evidence & contradictions...",
                        "experiment": "Executing experiment plan...",
                        "critique": "Reviewing experiment results..."
                    }
                    if name in agent_map:
                        activity = {
                            "agent": agent_map[name],
                            "action": action_map.get(name, f"Running {name}..."),
                            "status": "thinking" if name == "critique" else "building"
                        }
                        yield f"data: {json.dumps({'type': 'activity', 'data': activity})}\n\n"
                        await asyncio.sleep(0)

                # 2. TOOL & LOG UPDATES
                elif kind == "on_tool_start":
                    run_id = event.get("run_id")
                    if run_id: start_times[run_id] = time.time()
                    msg = f"[TOOL START] {name}"
                    yield f"data: {json.dumps({'type': 'log', 'data': msg})}\n\n"
                    await asyncio.sleep(0)
                
                elif kind == "on_tool_end":
                    run_id = event.get("run_id")
                    duration_str = ""
                    if run_id and run_id in start_times:
                        duration = time.time() - start_times[run_id]
                        duration_str = f" ({duration:.2f}s)"
                        del start_times[run_id]
                    
                    output = str(data.get("output"))
                    preview = output[:150] + "..." if len(output) >= 150 else output
                    msg = f"[Result] {preview}{duration_str}"
                    yield f"data: {json.dumps({'type': 'log', 'data': msg})}\n\n"
                    await asyncio.sleep(0)
                
                elif kind == "on_custom_event" and name == "log":
                    start_time = event.get("metadata", {}).get("created_at") # Optional
                    msg = data.get("message", "")
                    yield f"data: {json.dumps({'type': 'log', 'data': msg})}\n\n"
                    await asyncio.sleep(0)

                elif kind == "on_chain_end":
                    batch_output = event.get('data', {}).get('output', {})
                    if hasattr(batch_output, "dict"): update_dict = batch_output.model_dump()
                    elif isinstance(batch_output, dict): update_dict = batch_output
                    else: update_dict = {}
                    
                    relevant_keys = ["experiments", "critique", "experiment_plans"]
                    has_update = False
                    for key in relevant_keys:
                        if key in update_dict:
                            accumulated_state[key] = update_dict[key]
                            has_update = True
                            
                    if has_update:
                        yield f"data: {json.dumps({'type': 'result', 'data': accumulated_state}, default=custom_serializer)}\n\n"

        except Exception as e:
            request_log.error("stream_loop_error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    )

class SearchRequest(BaseModel):
    query: str
    max_papers: int = 10

@app.post("/search_papers")
async def search_papers_endpoint(request: SearchRequest):
    """
    Stand-alone endpoint for the 'Curation' step in frontend.
    Fetches papers from OpenAlex based on the query.
    """
    log.info("paper_search_request", query=request.query)
    try:
        from app.tools.openalex import search_papers
        
        # Simple refinement: quote the query if it's too simple? 
        # Actually OpenAlex works best with simple keywords or boolean
        results = await search_papers(request.query, limit=request.max_papers)
        
        # Transform for frontend if needed (frontend expects id, title, year, venue, abstract/rationale)
        # Our tool returns: id, title, publication_year, abstract(inverted), host_venue...
        
        papers = []
        from app.tools.openalex import reconstruct_abstract
        
        for p in results:
            abstract_text = reconstruct_abstract(p.get("abstract")) if p.get("abstract") else "No abstract available."
            papers.append({
                "id": p["id"],
                "title": p["title"],
                "year": p["publication_year"],
                "venue": p["host_venue"] or "Unknown Venue",
                "abstract": abstract_text,
                "rationale": abstract_text[:200] + "...",
                "url": p.get("landing_page_url")
            })
            
        return {"papers": papers}
    except Exception as e:
        log.error("paper_search_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_stream")
async def run_discovery_stream(request: RunRequest):
    """
    Trigger the discovery loop and stream events.
    Supports resuming via thread_id and providing feedback.
    """
    request_id = str(uuid.uuid4())[:8]
    
    # Bind request ID to logger for this request
    request_log = log.bind(request_id=request_id, query=request.query[:50])
    
    # Check concurrency
    thread_id = request.thread_id or str(uuid.uuid4())
    if thread_id in active_threads:
        log.warning("concurrency_blocked", thread_id=thread_id)
        raise HTTPException(status_code=409, detail="Pipeline already running for this thread. Please wait.")
    
    # Acquire Lock
    active_threads.add(thread_id)
    
    try:
        from app.graph import create_graph
        from app.state import DiscoveryState
        
        config = {"configurable": {"thread_id": thread_id}}

        request_log.info(
            "discovery_request_started",
            thread_id=thread_id,
            goal=request.goal,
            lens=request.lens,
            speculation=request.speculation,
            run_experiments=request.run_experiments,
            is_resume=bool(request.thread_id)
        )

        graph = create_graph()
        
        # If resuming with feedback
        initial_state = None
        if request.feedback and request.thread_id:
            request_log.info("resuming_with_feedback", feedback=request.feedback[:100])
            graph.update_state(config, {"human_feedback": request.feedback})
            initial_state = None # Resume from current state
        else:
            # Start new
            initial_state = DiscoveryState(
                user_query=request.query,
                goal=request.goal,
                lens=request.lens,
                speculation=request.speculation,
                run_experiments=request.run_experiments,
                documents=request.documents,
                mock=request.mock
            )

    except Exception as e:
        active_threads.discard(thread_id) # Release on init failure
        import traceback
        request_log.error("discovery_init_failed", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Init failed: {e}")

    async def event_generator():
        try:
            # Emit initial thinking event with Thread ID
            data = {'agent': 'orchestrator', 'action': f'Starting discovery for: {request.query}', 'status': 'thinking'}
            yield f"data: {json.dumps({'type': 'activity', 'data': data, 'thread_id': thread_id})}\n\n"
            await asyncio.sleep(0.1) # Force flush
        
            # Accumulator for final result
            accumulated_state = {}
            start_times = {} # Track durations
            
            # Use astream_events to get granular updates
            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                kind = event["event"]
                name = event.get("name", "")
                data = event.get("data", {})

                # DEBUG PRINT (Visible in server console)
                if kind == "on_custom_event":
                    print(f"DEBUG_EVENT: {kind} name={name} data={str(data)[:100]}")
                
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
                    run_id = event.get("run_id")
                    if run_id:
                        start_times[run_id] = time.time()
                    msg = f"[TOOL START] {name}"
                    if name == "get_neighbors":
                        node = data.get("input", {}).get("node", "?")
                        msg = f"[Graph] Exploring neighbors of '{node}'..."
                    elif name == "find_paths":
                        start = data.get("input", {}).get("start_node", "?")
                        end = data.get("input", {}).get("end_node", "?")
                        msg = f"[Graph] Tracing path: {start} -> {end}..."
                    elif name == "get_central_nodes":
                        msg = f"[Graph] Identifying central concepts..."
                    elif "search" in name or "openalex" in name:
                        q = data.get("input", {}).get("query", str(data.get("input", "")))[:40]
                        msg = f"[Search] Querying: {q}..."
                    
                    yield f"data: {json.dumps({'type': 'log', 'data': msg})}\n\n"
                    await asyncio.sleep(0)
                
                elif kind == "on_tool_end":
                    run_id = event.get("run_id")
                    duration_str = ""
                    if run_id and run_id in start_times:
                        duration = time.time() - start_times[run_id]
                        duration_str = f" ({duration:.2f}s)"
                        del start_times[run_id]

                    output = str(data.get("output"))
                    if len(output) < 150:
                        preview = output
                    else:
                        preview = output[:150] + "..."
                    
                    if "error" in output.lower():
                        msg = f"[Tool Error] {preview}"
                    else:
                        msg = f"[Result] {preview}"
                    msg += duration_str
                    yield f"data: {json.dumps({'type': 'log', 'data': msg})}\n\n"
                    await asyncio.sleep(0)
                
                elif kind == "on_custom_event":
                    if event["name"] == "log":
                        log_data = data.get("message", str(data))
                        yield f"data: {json.dumps({'type': 'log', 'data': log_data})}\n\n"
                        await asyncio.sleep(0)
                    elif event["name"] == "activity":
                        msg = data.get("message", str(data))
                        activity_data = {"agent": "scientist", "action": msg, "status": "thinking"}
                        yield f"data: {json.dumps({'type': 'activity', 'data': activity_data})}\n\n"
                        await asyncio.sleep(0)

                elif kind == "on_chain_end":
                    batch_output = event.get('data', {}).get('output', {})
                    if hasattr(batch_output, "dict"): update_dict = batch_output.model_dump()
                    elif isinstance(batch_output, dict): update_dict = batch_output
                    else: update_dict = {}
                    
                    relevant_keys = ["plan", "literature", "hypotheses", "evidence", "experiments", "experiment_plans", "critique", "user_query", "concept_graph", "domain_tags"]
                    has_update = False
                    for key in relevant_keys:
                        if key in update_dict:
                            accumulated_state[key] = update_dict[key]
                            has_update = True
                    if has_update:
                        if "user_query" not in accumulated_state: accumulated_state["user_query"] = request.query
                        yield f"data: {json.dumps({'type': 'result', 'data': accumulated_state}, default=custom_serializer)}\n\n"

        except Exception as e:
            request_log.error("stream_loop_error", error=str(e), thread_id=thread_id)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        except BaseException as e:
            request_log.error("stream_loop_critical_failure", error=str(e), type=type(e).__name__, thread_id=thread_id)
            print(f"DEBUG: Critical failure for {thread_id}: {type(e).__name__} - {e}")
            raise e
        
        finally:
            # RELEASE LOCK
            active_threads.discard(thread_id)
            print(f"DEBUG: Released lock for {thread_id}")

        # Check if we are interrupted or done
        try:
            snapshot = await graph.aget_state(config)
            if snapshot.next:
                request_log.info("workflow_interrupted", next_nodes=list(snapshot.next))
                yield f"data: {json.dumps({'type': 'interrupt', 'data': {'next': list(snapshot.next), 'thread_id': thread_id}})}\n\n"
            else:
                request_log.info("workflow_completed")
                yield f"data: {json.dumps({'type': 'result', 'data': snapshot.values}, default=custom_serializer)}\n\n"
                yield "data: [DONE]\n\n"
        except Exception as e:
            request_log.error("fail_safe_sync_failed", error=str(e))
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

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0"}

if __name__ == "__main__":
    print("Starting SciNets Server on Port 8005...")
    uvicorn.run(app, host="127.0.0.1", port=8005, log_level="info")
