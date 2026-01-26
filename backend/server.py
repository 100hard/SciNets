

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()
from fastapi.responses import StreamingResponse
import json
import asyncio
import uvicorn
import os
import sys
import time
import os
import sys

# Ensure backend dir is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Structured Logging Setup
from app.logging_config import setup_logging, get_logger
import structlog
from app.database import SessionLocal, get_db
from app.models import User, Session as DbSession, MagicLink, DiscoveryRun
from app.auth_utils import create_magic_link_token, verify_magic_link_token, send_magic_link_email, hash_token
from app.config import config as app_config
from fastapi import Response, Request, Depends, Cookie, status
from sqlalchemy.orm import Session
import datetime
import datetime
from app.telemetry.cost_tracker import CostTracker

def custom_serializer(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, uuid.UUID):
        return str(obj)
    return str(obj)

# Hardening
class RateLimiter:
    def __init__(self):
        self.requests = {} # ip -> [timestamps]
    
    def check(self, ip: str, limit: int = 60, window: int = 60) -> bool:
        now = time.time()
        if ip not in self.requests:
            self.requests[ip] = []
        
        # Cleanup old
        self.requests[ip] = [t for t in self.requests[ip] if now - t < window]
        
        if len(self.requests[ip]) >= limit:
            return False
            
        self.requests[ip].append(now)
        return True

rate_limiter = RateLimiter()

# Active runs per user
active_runs: Dict[str, str] = {} # user_id -> thread_id

# Active runs per user
from pydantic import BaseModel, field_validator

# Active runs per user
active_runs: Dict[str, str] = {} # user_id -> thread_id

class EmailRequest(BaseModel):
    email: str

class VerifyRequest(BaseModel):
    token: str

# Active runs per user
# ... (Removed load_dotenv from here)

# Setup logging (JSON in production, colored in dev)
log_level = os.getenv("LOG_LEVEL", "INFO")
json_logs = os.getenv("JSON_LOGS", "false").lower() == "true"
setup_logging(log_level=log_level, json_logs=json_logs)

log = get_logger(__name__)

app = FastAPI(title="SciNets V2 API")

log.info("scinets_startup", version="2.0", log_level=log_level, json_mode=json_logs)

# Initialize Global Graph
app_graph = None

from app.api.auth_google import router as google_auth_router
app.include_router(google_auth_router)

@app.on_event("startup")
async def startup_event():
    global app_graph
    from app.graph import create_graph, global_memory
    log.info("initializing_graph")
    app_graph = create_graph(memory=global_memory)
    log.info("graph_initialized")


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        app_config.FRONTEND_URL,
        "https://scinets.in",
        "https://www.scinets.in",
        "https://scinets-backend.onrender.com", # Added Backend URL itself just in case
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True, # REQUIRED for cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/debug/auth")
async def debug_auth_config():
    """Helper to verify Prod Env Vars are loaded correctly."""
    idx = os.getenv("GOOGLE_CLIENT_ID", "")
    return {
        "frontend_url": app_config.FRONTEND_URL,
        "google_client_id_prefix": idx[:15] + "..." if idx else "NOT_SET",
        "cors_origins": [
            "https://scinets.in", 
            app_config.FRONTEND_URL
        ]
    }

# Middleware: Per-IP Rate Limit
@app.post("/api/auth/request-link")
async def request_magic_link(req: EmailRequest, request: Request, db: Session = Depends(get_db)):
    """Generates a magic link, stores hash, and sends via SMTP."""
    log.info("magic_link_request_received", email=req.email, ip=request.client.host)
    try:
        # Rate Limit Check (stricter for auth)
        client_ip = request.client.host
        if not app_config.DEMO_MODE and not rate_limiter.check(client_ip, limit=5, window=3600):
            log.warning("magic_link_rate_limited", ip=client_ip)
            # Silent failure on rate limit to prevent enumeration? Or minimal error?
            # Let's just return success message to be safe.
            time.sleep(1) # Fake delay
            return {"message": "If that email exists, we sent a magic link."}

        email = req.email.strip().lower()
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Silent Success: Don't reveal user existence
            # But we still want to create users for new signups in this MVP?
            # If open signup: Create user. If closed: Silent fail.
            # Assuming OPEN signup for SciNets V2 demo.
            log.info("magic_link_creating_user", email=email)
            user = User(email=email)
            db.add(user)
            db.commit()
        
        token = create_magic_link_token(email)
        hashed = hash_token(token)
        expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=app_config.MAGIC_LINK_EXPIRE_MINUTES)
        
        db.query(MagicLink).filter(MagicLink.email == email).delete()
        magic_link_record = MagicLink(token_hash=hashed, email=email, expires_at=expires)
        db.add(magic_link_record)
        db.commit()
        
        base_url = app_config.FRONTEND_URL 
        link = f"{base_url}/verify?token={token}"
        send_magic_link_email(email, link)
        
        log.info("magic_link_process_complete", email=email)
        
        # Always return same message
        return {"message": "If that email exists, we sent a magic link."}
    except Exception as e:
        import traceback
        log.error("magic_link_endpoint_error", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/verify-link")
async def verify_magic_link(req: VerifyRequest, response: Response, request: Request, db: Session = Depends(get_db)):
    email = verify_magic_link_token(req.token)
    if not email: 
        # Random delay to prevent timing attacks
        time.sleep(0.5)
        raise HTTPException(status_code=400, detail="Invalid token")
    
    hashed = hash_token(req.token)
    record = db.query(MagicLink).filter(MagicLink.token_hash == hashed).first()
    if not record or record.expires_at < datetime.datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired token")
        
    db.delete(record)
    db.commit()
    
    user = db.query(User).filter(User.email == email).first()
    if not user: raise HTTPException(status_code=400, detail="User not found")
    
    # Create Session with Binding
    session_id = str(uuid.uuid4())
    expires = datetime.datetime.utcnow() + datetime.timedelta(days=7)
    
    # Capture IP Prefix (/24)
    client_ip = request.client.host
    ip_parts = client_ip.split('.')
    if len(ip_parts) == 4:
        ip_prefix = ".".join(ip_parts[:3]) # 192.168.1
    else:
        ip_prefix = client_ip # IPv6 or other
        
    user_agent = request.headers.get("User-Agent", "Unknown")
    
    db_session = DbSession(
        id=session_id, 
        user_id=user.id, 
        expires_at=expires,
        ip_prefix=ip_prefix,
        user_agent=user_agent
    )
    db.add(db_session)
    db.commit()
    
    # FIX: Cross-Site Cookie Settings
    response.set_cookie(
        key="session_id", 
        value=session_id, 
        httponly=True, 
        secure=True,          # REQUIRED for SameSite=None
        samesite="none",      # REQUIRED for Cross-Site
        max_age=7*24*60*60
    )
    return {"message": "Logged in", "user": {"id": user.id, "email": user.email}}

@app.get("/api/auth/me")
async def get_current_user(request: Request, session_id: str | None = Cookie(default=None), db: Session = Depends(get_db)):
    if not session_id: raise HTTPException(status_code=401, detail="Not authenticated")
    session = db.query(DbSession).filter(DbSession.id == session_id).first()
    if not session: raise HTTPException(status_code=401, detail="Invalid session")
    if session.expires_at < datetime.datetime.utcnow(): raise HTTPException(status_code=401, detail="Session expired")
    
    # Validate Session Binding
    current_ip = request.client.host
    current_ua = request.headers.get("User-Agent", "Unknown")
    
    # IP Check (relaxed to /24)
    if session.ip_prefix:
        ip_parts = current_ip.split('.')
        current_prefix = ".".join(ip_parts[:3]) if len(ip_parts) == 4 else current_ip
        if current_prefix != session.ip_prefix and not app_config.DEMO_MODE:
             # RELAXED SECURITY: Log warning but allow session to continue (IP Drift common in Prod)
             log.warning("session_ip_mismatch_detected", stored=session.ip_prefix, current=current_prefix)
             # raise HTTPException(status_code=401, detail="Session expired (IP change)") # DISABLED for stability

    # UA Check
    if session.user_agent and session.user_agent != current_ua:
         log.warning("session_hijack_attempt_ua", stored=session.user_agent, current=current_ua)
         raise HTTPException(status_code=401, detail="Session expired (UA change)")
    
    user = db.query(User).filter(User.id == session.user_id).first()
    if not user: raise HTTPException(status_code=401, detail="User not found")
    
    # Calculate Quota Info for UI
    now = datetime.datetime.utcnow()
    # Auto-reset if window expired (48 hours)
    if not user.window_start_at or (now - user.window_start_at).total_seconds() >= 48 * 3600:
        user.window_start_at = now
        user.discoveries_in_window = 0
        db.commit()
        
    user_limit = user.custom_quota_limit if user.custom_quota_limit is not None else app_config.MAX_RUNS_PER_USER_PER_WEEK
    reset_date = user.window_start_at + datetime.timedelta(hours=48)
    hours_remaining = int((reset_date - now).total_seconds() / 3600)
    if hours_remaining < 0: hours_remaining = 0
    
    return {
        "id": user.id, 
        "email": user.email,
        "quota": {
            "used": user.discoveries_in_window,
            "limit": user_limit,
            "resets_in_hours": hours_remaining
        }
    }

@app.post("/api/auth/logout")
async def logout(response: Response, session_id: str | None = Cookie(default=None), db: Session = Depends(get_db)):
    if session_id:
        db.query(DbSession).filter(DbSession.id == session_id).delete()
        db.commit()
    response.delete_cookie("session_id")
    return {"message": "Logged out"}

# Concurrency Control
active_threads: set[str] = set()

class RunRequest(BaseModel):
    query: str
    goal: str = "discover"
    lens: str = "none"
    speculation: str = "medium"
    timeline: str = "recent"
    max_papers: int = 10
    guidance: Optional[str] = None
    max_papers: int = 10
    guidance: Optional[str] = None
    run_experiments: bool = False
    
    @field_validator('query')
    @classmethod
    def validate_input(cls, v: str) -> str:
        if len(v) > app_config.MAX_INPUT_CHARS:
             raise ValueError(f"Query too long (max {app_config.MAX_INPUT_CHARS} chars)")
        return v

    documents: List[str] = []
    selected_hypothesis_ids: List[str] = [] # Resume: specific IDs to deep-dive
    thread_id: str | None = None # For resuming sessions
    feedback: str | None = None # User feedback when resuming
    num_hypotheses: int = 3 # Configurable hypothesis count
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

        initial_state = {
            "hypothesis_id": request.hypothesis_id,
            "hypothesis_text": request.hypothesis_text,
            "intent": request.intent,
            "data_source": request.data_source,
            "experiment_result": None # Reset result
        }
        
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
            
            # Stream events (start fresh from input state)
            # FIX: We must pass 'initial_state' as input to trigger execution, 
            # as 'update_state' + 'None' input on a fresh MemorySaver graph does not trigger the entry point.
            last_ping = time.time()
            async for event in graph.astream_events(initial_state, config=config, version="v1"):
                # HEARTBEAT (Priority 1: Keepalive)
                if time.time() - last_ping > 15.0:
                    yield "event: ping\ndata: {}\n\n"
                    last_ping = time.time()
                    await asyncio.sleep(0)

                kind = event["event"]
                name = event.get("name", "")
                data = event.get("data", {})
                
                # REUSE LOGGING LOGIC FROM DISCOVERY
                # 1. MAJOR NODE UPDATES
                if kind == "on_chain_start" and name in ["literature", "hypothesis", "evidence", "experiment", "critique", "plan", "localized_critique"]:
                    agent_map = {
                        "plan": "planner", "literature": "scientist", "hypothesis": "scientist",
                        "evidence": "critic", "experiment": "scientist", "critique": "critic",
                        "localized_critique": "critic"
                    }
                    action_map = {
                        "plan": "Structuring research plan...",
                        "literature": "Analyzing literature & context...",
                        "hypothesis": "Generating scientific hypotheses...",
                        "evidence": "Evaluating evidence & contradictions...",
                        "experiment": "Executing experiment plan...",
                        "critique": "Reviewing experiment results...",
                        "localized_critique": "Reviewing experiment results..."
                    }
                    if name in agent_map:
                        msg = action_map.get(name, f"Running {name}...")
                        print(f" -> [Agent: {agent_map[name]}] {msg}")  # Console echo
                        activity = {
                            "agent": agent_map[name],
                            "action": msg,
                            "status": "thinking" if name == "critique" else "building"
                        }
                        yield f"data: {json.dumps({'type': 'activity', 'data': activity})}\n\n"
                        await asyncio.sleep(0)

                # 2. TOOL & LOG UPDATES
                elif kind == "on_tool_start":
                    run_id = event.get("run_id")
                    if run_id: start_times[run_id] = time.time()
                    msg = f"[TOOL START] {name}"
                    print(f"    > {msg}") # Console echo
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
                    print(f"    < {msg}") # Console echo
                    yield f"data: {json.dumps({'type': 'log', 'data': msg})}\n\n"
                    await asyncio.sleep(0)
                
                elif kind == "on_custom_event" and name == "log":
                    start_time = event.get("metadata", {}).get("created_at") # Optional
                    msg = data.get("message", "")
                    print(f" [LOG] {msg}") # Console echo
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
            print(f"ERROR: {str(e)}") # Console echo
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
    Fetches papers from OpenAlex with LLM-refined query to match Backend Graph quality.
    """
    log.info("paper_search_request", query=request.query)
    try:
        from app.tools.search import search_papers
        from app.llm import get_cheap_llm
        from langchain_core.prompts import ChatPromptTemplate
        
        # FIX #1: Bind UI to Graph Quality (Refine Query first)
        llm = get_cheap_llm()
        refine_prompt = ChatPromptTemplate.from_template(
            """Convert the following user research query into a specific keyword-based search query for OpenAlex.
            Use AND/OR operators. Focus on domain-specific terms.
            Expected Output: A single line string.
            
            User Query: {query}
            Refined Query:"""
        )
        chain = refine_prompt | llm
        refined_query_res = await chain.ainvoke({"query": request.query})
        refined_query = refined_query_res.content.replace('"', '').strip()
        
        # Guard against LLM failure/hallucination
        if len(refined_query) < 5 or "search query" in refined_query.lower():
            refined_query = request.query            
            
        log.info("paper_search_refined", original=request.query, refined=refined_query)

        # Search OpenAlex with REFINED query
        # Fetch extra to account for filtering (6x buffer to find Crossref abstracts)
        try:
             results = await search_papers(refined_query, limit=request.max_papers * 6)
        except Exception as e:
             log.warning("paper_search_refined_failed", error=str(e))
             # Fallback to original
             results = await search_papers(request.query, limit=request.max_papers * 6)
             
        log.info("paper_search_results", count=len(results), query=refined_query)
        
        # FALLBACK LOGIC (Simple Keywords)
        if len(results) == 0:
            log.info("paper_search_fallback", original_query=request.query)
            stopwords = ["find", "mechanism", "connecting", "to", "the", "a", "an", "and", "or", "of", "in", "for", "with"]
            keywords = [w for w in request.query.lower().split() if w not in stopwords]
            simple_query = " ".join(keywords)
            
            results = await search_papers(simple_query, limit=request.max_papers * 6)
            log.info("paper_search_fallback_results", count=len(results), simple_query=simple_query)

        papers = []
        # No need to import reconstruct_abstract from openalex if we implement logic here or use it directly
        # But for robustness let's do the manual check like main.py
        
        for p in results:
            # Reconstruct abstract
            # Reconstruct abstract
            raw_abstract = p.get("abstract")
            abstract_text = ""
            
            # CASE A: Already a String (Crossref / Semantic Scholar / Fallback)
            if raw_abstract and isinstance(raw_abstract, str):
                abstract_text = raw_abstract
            
            # CASE B: OpenAlex Inverted Index (Dict)
            elif raw_abstract and isinstance(raw_abstract, dict):
                 try:
                    inverted_index = raw_abstract
                    word_positions = []
                    for word, positions in inverted_index.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort()
                    abstract_text = " ".join(w for _, w in word_positions)
                 except: 
                    abstract_text = "Error reconstructing abstract"
            
            else:
                # No abstract or unknown format
                abstract_text = ""
            
            if not abstract_text or abstract_text == "No abstract available.":
                continue

            papers.append({
                "id": p["id"],
                "title": p["title"],
                "year": p["publication_year"],
                "venue": p["host_venue"] or "Unknown Venue",
                "abstract": abstract_text,
                "rationale": abstract_text[:200] + "...",
                "url": p.get("landing_page_url")
            })

            # Stop once we have enough valid papers
            if len(papers) >= request.max_papers:
                break
            
        return {"papers": papers}
    except Exception as e:
        log.error("paper_search_failed", error=str(e))
        # Return empty list instead of 500
        return {"papers": []}

@app.post("/run_stream")
async def run_discovery_stream(request: RunRequest, http_request: Request, db: Session = Depends(get_db)):
    """
    Trigger the discovery loop and stream events.
    Supports resuming via thread_id and providing feedback.
    """
    request_id = str(uuid.uuid4())[:8]

    # 0. Kill Switch Check
    if app_config.SCINETS_READONLY_MODE:
        raise HTTPException(status_code=503, detail="SciNets is currently in Read-Only mode. Discovery is paused.")

    # Bind request ID to logger for this request
    request_log = log.bind(request_id=request_id, query=request.query[:50])
    
    # 1. Auth & Quota Check
    print(f"DEBUG: Cookies received: {http_request.cookies}")
    session_id = http_request.cookies.get("session_id")
    user_id = None
    
    if session_id:
        session = db.query(DbSession).filter(DbSession.id == session_id).first()
        print(f"DEBUG: Session Query: {session_id}, Result: {session}, Expires: {session.expires_at if session else 'N/A'}")
        if session and session.expires_at > datetime.datetime.utcnow():
             user_id = session.user_id
    
    if not user_id and not request.mock: # Allow mock runs without auth? Maybe not for public demo.
         # For public demo, strictly require auth
         raise HTTPException(status_code=401, detail="Authentication required for discovery.")

    # Check concurrency
    thread_id = request.thread_id or str(uuid.uuid4())
    if thread_id in active_threads:
        log.warning("concurrency_blocked", thread_id=thread_id)
        raise HTTPException(status_code=409, detail="Pipeline already running for this thread. Please wait.")
    
    # Acquire Lock via User/Thread
    lock_key = f"{user_id}:{thread_id}" if user_id else thread_id
    if lock_key in active_runs.values():
         log.warning("concurrency_blocked", user_id=user_id, thread_id=thread_id)
         raise HTTPException(status_code=429, detail="You already have an active search running.")
    
    active_runs[user_id or thread_id] = lock_key
    active_threads.add(thread_id)

    # QUOTA LOGIC
    if user_id:
        # FIX: Only apply Quota Limits to NEW runs (not Resumes/Deep Dives)
        # If thread_id is provided, it's a continuation -> specific Logic Skip.
        if not request.thread_id:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                now = datetime.datetime.utcnow()
                # 1. Reset Window if needed (48h)
                if not user.window_start_at or (now - user.window_start_at).total_seconds() >= 48 * 3600:
                    user.window_start_at = now
                    user.discoveries_in_window = 0
                    db.commit()
                
                # 2. Check Limit
                user_limit = user.custom_quota_limit if user.custom_quota_limit is not None else app_config.MAX_RUNS_PER_USER_PER_WEEK
                
                if user.discoveries_in_window >= user_limit:
                    log.warning("quota_exceeded", user_id=user_id, count=user.discoveries_in_window, limit=user_limit)
                    
                    # Calculate reset time
                    reset_date = user.window_start_at + datetime.timedelta(hours=48)
                    hours_remaining = int((reset_date - now).total_seconds() / 3600)
                    if hours_remaining < 1: hours_remaining = 1 # Avoid 0 hours confusion
                    
                    detail = {
                        "error": "quota_exceeded",
                        "message": f"You've used your {user_limit} SciNets discoveries for this 48h period. (Resets in {hours_remaining} hours)",
                        "resets_in_hours": hours_remaining
                    }
                    raise HTTPException(status_code=429, detail=detail)
                
                # 3. Increment (Optimistic)
                user.discoveries_in_window += 1
                user.last_discovery_at = now
                db.commit()
                
                # Quota Info for Frontend
                quota_info = {
                    "used": user.discoveries_in_window,
                    "limit": user_limit,
                    "remaining": max(0, user_limit - user.discoveries_in_window)
                }
                
                # 4. Record Run History
                try:
                    run_record = DiscoveryRun(
                        id=thread_id,
                        user_id=user_id,
                        query=request.query,
                        status="started",
                        is_demo=False
                    )
                    db.add(run_record)
                    db.commit()
                except Exception as e:
                    log.error("run_recording_failed", error=str(e))
        else:
             # Resume/Deep Dive - No Quota Cost
             quota_info = None 


    else:
        # User not logged in or disabled auth
        quota_info = None

    # Concurrency check already done above.
    # Proceed to Graph Init
    
    try:
        from app.state import DiscoveryState
        
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": app_config.MAX_AGENT_STEPS}

        request_log.info(
            "discovery_request_started",
            thread_id=thread_id,
            goal=request.goal,
            lens=request.lens,
            speculation=request.speculation,
            run_experiments=request.run_experiments,
            is_resume=bool(request.thread_id)
        )

        global app_graph
        graph = app_graph
        
        # ... (State Init Logic Skipped for Brevity - Keeping Existing Logic) ...
        # If resuming with feedback
        initial_state = None
        if request.feedback and request.thread_id:
             # Regular Resume with Feedback (Legacy)
            request_log.info("resuming_with_feedback", feedback=request.feedback[:100])
            graph.update_state(config, {"human_feedback": request.feedback})
            initial_state = None 
        elif request.selected_hypothesis_ids and request.thread_id:
             # NEW: Resume with Selection
             request_log.info("resuming_with_selection", selected_ids=request.selected_hypothesis_ids)
             
             # Fetch current state
             current_state = graph.get_state(config).values
             log.info("resume_state_trace", 
                      thread_id=request.thread_id, 
                      state_keys=list(current_state.keys()), 
                      hypotheses_count=len(current_state.get("hypotheses", []))
             )

             all_hypotheses = current_state.get("hypotheses", [])
             
             # Filter hypotheses
             selected = [h for h in all_hypotheses if h.id in request.selected_hypothesis_ids]
             if not selected:
                 # Fallback if IDs don't match (maybe just take top 3)
                 log.warning("selection_mismatch", requested=request.selected_hypothesis_ids)
                 selected = all_hypotheses[:3]
             
             # Update state with filtered list AND set mode to DEEP
             graph.update_state(config, {
                 "hypotheses": selected, 
                 "selected_hypothesis_ids": request.selected_hypothesis_ids,
                 "hypothesis_mode": "deep" 
             })
             initial_state = None
        else:
             # Start new
             # QUOTA CHECK (Weekly)
             # Note: thread_id is random for new runs, we need USER context.
             # Ideally, we should pass user_id/session_id to run_stream or look it up.
             # However, run_stream is POSTed to by frontend with session cookie.
             # We rely on active_runs[user_id] being set above, but that needs to be robust.
             
             # Fetch User for Quota Update
             # Since we are in an async loop and DB session is not passed in easily here 
             # (we only have get_db dependency in the endpoint signature if we add it),
             # let's assume valid user_id from the endpoint auth check.
             
             if user_id: # Only enforce if we identified a user
                 # We need a fresh DB session here as this is inside the event generator/endpoint
                 # Use the 'db' session we can inject into the endpoint
                 pass 
                 # Wait, we can't easily inject DB into this inner scope if we didn't pass it.
                 # Let's verify if we can do the check BEFORE entering the event loop (at endpoint level).
             
             # Force 6 candidates for selection mode
             target_hypotheses = request.num_hypotheses if request.num_hypotheses > 3 else 6
             initial_state = DiscoveryState(
                 user_query=request.query,
                 goal=request.goal,
                 lens=request.lens,
                 speculation=request.speculation,
                 timeline=request.timeline,
                 max_papers=request.max_papers,
                 guidance=request.guidance,
                 run_experiments=request.run_experiments,
                 num_hypotheses=target_hypotheses, # Use the forced 6 or user value
                 documents=request.documents,
                 mock=request.mock
             )

    except Exception as e:
        active_threads.discard(thread_id)
        if user_id or thread_id in active_runs: del active_runs[user_id or thread_id]
        import traceback
        request_log.error("discovery_init_failed", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Init failed: {e}")

    async def event_generator():
        # Register CostTracker
        cost_tracker = CostTracker()
        ct_token = CostTracker.register_context(cost_tracker)
        
        # Hardening Counters
        step_count = 0
        start_time_glob = time.time()
        
        # Helper to update run status
        def update_run_status(status_val, cost=0.0, tokens=0, duration=0.0, result_path=None):
             try:
                 # Use fresh session for async update
                 db_local = SessionLocal()
                 run = db_local.query(DiscoveryRun).filter(DiscoveryRun.id == thread_id).first()
                 if run:
                     run.status = status_val
                     if status_val in ["completed", "failed", "interrupted"]:
                         run.completed_at = datetime.datetime.utcnow()
                         run.duration = duration
                         run.cost_usd = cost
                         run.tokens = tokens
                         if result_path:
                             run.result_path = result_path
                     db_local.commit()
                 db_local.close()
             except Exception as ex:
                 print(f"Failed to update run status: {ex}")

        try:
            try:
                # Emit initial thinking event
                data = {'agent': 'orchestrator', 'action': f'Starting discovery for: {request.query}', 'status': 'thinking'}
                yield f"data: {json.dumps({'type': 'activity', 'data': data, 'thread_id': thread_id})}\n\n"
                
                if quota_info:
                     yield f"data: {json.dumps({'type': 'quota', 'data': quota_info})}\n\n"

                await asyncio.sleep(0.1) 
            
                accumulated_state = {}
                start_times = {} 
                
                # Stream events
                last_ping = time.time()
                async for event in graph.astream_events(initial_state, config=config, version="v2"):
                    # HEARTBEAT (Priority 1: Keepalive for Render/Vercel)
                    if time.time() - last_ping > 15.0:
                        yield "event: ping\ndata: {}\n\n"
                        last_ping = time.time()
                        await asyncio.sleep(0)

                    kind = event["event"]
                    name = event.get("name", "")
                    
                    # 1. ENFORCE LIMITS (Smarter Counting)
                    # Only count major state changes or tool calls as "steps" to avoid counting token stream events
                    if kind == "on_chain_start" and name in ["plan", "literature", "hypothesis", "hypothesis_preview", "hypothesis_deep", "evidence", "critique", "decision", "experiment"]:
                         step_count += 1
                    elif kind == "on_tool_start":
                         step_count += 1
                    
                    elapsed = time.time() - start_time_glob
                    current_tokens = cost_tracker.total_tokens
                    
                    if step_count > app_config.MAX_AGENT_STEPS:
                        update_run_status("failed")
                        yield f"data: {json.dumps({'type': 'error', 'data': 'Max steps exceeded.'})}\n\n"
                        request_log.warning("limit_hit_steps", steps=step_count)
                        break
                        
                    if elapsed > app_config.MAX_RUN_TIME_SECONDS:
                        update_run_status("failed")
                        yield f"data: {json.dumps({'type': 'error', 'data': 'Max run time exceeded.'})}\n\n"
                        request_log.warning("limit_hit_time", elapsed=elapsed)
                        break
                        
                    if current_tokens > app_config.MAX_TOTAL_TOKENS:
                         update_run_status("failed")
                         yield f"data: {json.dumps({'type': 'error', 'data': 'Max token limit exceeded.'})}\n\n"
                         request_log.warning("limit_hit_tokens", tokens=current_tokens)
                         break

                    kind = event["event"]
                    name = event.get("name", "")
                    data = event.get("data", {})
    
                    # DEBUG PRINT
                    if kind == "on_custom_event":
                        print(f"DEBUG_EVENT: {kind} name={name} data={str(data)[:100]}")
                    
                    # 1. MAJOR NODE UPDATES (High Level)
                    if kind == "on_chain_start" and name in ["literature", "hypothesis", "evidence", "experiment", "critique", "plan", "decision"]:
                        agent_map = {
                            "plan": "planner", 
                            "literature": "literature", 
                            "hypothesis": "hypothesis",
                            "evidence": "critic", 
                            "experiment": "experiment", 
                            "critique": "critic",
                            "decision": "orchestrator"
                        }
                        action_map = {
                            "plan": "Structuring research plan...",
                            "literature": "Searching and reading literature...",
                            "hypothesis": "Generating and refining hypotheses...",
                            "evidence": "Verifying evidence & facts...",
                            "experiment": "Designing and running experiments...",
                            "critique": "Critiquing and validating findings...",
                            "decision": "Synthesizing final decision..."
                        }
                        if name in agent_map:
                            activity = {
                                "agent": agent_map[name],
                                "action": action_map.get(name, f"Starting {name}..."),
                                "status": "thinking" if name in ["plan", "critique", "evidence", "decision"] else "reading" if name == "literature" else "building"
                            }
                            yield f"data: {json.dumps({'type': 'activity', 'data': activity})}\n\n"
                            await asyncio.sleep(0)
    
                    # 2. TOOL & LOG UPDATES (In-Depth)
                    elif kind == "on_tool_start":
                        run_id = event.get("run_id")
                        if run_id:
                            start_times[run_id] = time.time()
                        msg = f"[TOOL START] {name}"
                        # ... (Tool Log formatting logic same as before)
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
                        
                        relevant_keys = ["plan", "literature", "hypotheses", "evidence", "experiments", "experiment_plans", "critique", "user_query", "concept_graph", "domain_tags", "decision_summary"]
                        has_update = False
                        for key in relevant_keys:
                            if key in update_dict:
                                accumulated_state[key] = update_dict[key]
                                has_update = True
                        if has_update:
                            if "user_query" not in accumulated_state: accumulated_state["user_query"] = request.query
                            yield f"data: {json.dumps({'type': 'result', 'data': accumulated_state}, default=custom_serializer)}\n\n"
                            # await asyncio.sleep(0)
    
            except Exception as e:
                update_run_status("failed")
                request_log.error("stream_loop_error", error=str(e), thread_id=thread_id)
                yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            except BaseException as e:
                update_run_status("failed")
                request_log.error("stream_loop_critical_failure", error=str(e), type=type(e).__name__, thread_id=thread_id)
                print(f"DEBUG: Critical failure for {thread_id}: {type(e).__name__} - {e}")
                raise e
            
            finally:
                # RELEASE LOCK
                active_threads.discard(thread_id)
                if user_id or thread_id in active_runs: del active_runs[user_id or thread_id]
                print(f"DEBUG: Released lock for {thread_id}")
    
            # Check if we are interrupted or done
            try:
                snapshot = await graph.aget_state(config)
                
                # --- COST REPORTING ---
                cost_report = cost_tracker.get_report()
                
                # 1. Log Summary
                summary_str = f"[COST] Total: ${cost_report['estimated_cost_usd']} | Tokens: {cost_report['total_tokens']}"
                print(summary_str)
                request_log.info("run_completed", 
                                 cost_usd=cost_report['estimated_cost_usd'], 
                                 tokens=cost_report['total_tokens'],
                                 duration=time.time() - start_time_glob,
                                 status="finished" if not snapshot.next else "interrupted")
                
                # Update Status to Completed (or Interrupted)
                update_run_status(
                    "completed" if not snapshot.next else "interrupted",
                    cost=cost_report['estimated_cost_usd'],
                    tokens=cost_report['total_tokens'],
                    duration=time.time() - start_time_glob,
                    result_path=f"data/runs/{thread_id}.json" # Standard path
                )

                # 2. Save JSON to disk
                try:
                    os.makedirs("data/runs", exist_ok=True)
                    cost_file = f"data/runs/{thread_id}_cost.json"
                    with open(cost_file, "w") as f:
                        json.dump(cost_report, f, indent=2)
                except Exception as e:
                    print(f"Failed to save cost file: {e}")

                # 3. Attach to state (if possible) or just result payload
                final_data = dict(snapshot.values) if snapshot.values else {}
                final_data["cost_report"] = cost_report

                if snapshot.next:
                    # request_log.info("workflow_interrupted", next_nodes=list(snapshot.next)) -- already logged in run_completed
                    yield f"data: {json.dumps({'type': 'result', 'data': final_data}, default=custom_serializer)}\n\n"
                    yield f"data: {json.dumps({'type': 'interrupt', 'data': {'next': list(snapshot.next), 'thread_id': thread_id}})}\n\n"
                else:
                    # request_log.info("workflow_completed") -- already logged
                    
                    # PERSIST FOR EXPORT
                    try:
                        from app.storage import save_run_result
                        save_run_result(thread_id, final_data)
                    except Exception as save_err:
                         print(f"Error saving run result: {save_err}")
    
                    yield f"data: {json.dumps({'type': 'result', 'data': final_data}, default=custom_serializer)}\n\n"
                    yield "data: [DONE]\n\n"
            except Exception as e:
                request_log.error("fail_safe_sync_failed", error=str(e))
                yield "data: [DONE]\n\n"
        finally:
            CostTracker.reset_context(ct_token)

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

@app.get("/api/discovery/{thread_id}/export/pdf")
async def export_pdf(thread_id: str):
    """
    Exports a completed discovery run as a PDF report.
    """
    from app.storage import get_run_result
    from app.reporting import generate_markdown_report, render_pdf
    from io import BytesIO
    
    state = get_run_result(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found or expired")
        
    try:
        md = generate_markdown_report(state)
        pdf_bytes = render_pdf(md)
        
        headers = {
            "Content-Disposition": f"attachment; filename=scinets_report_{thread_id}.pdf"
        }
        return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log.error("pdf_export_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate PDF")

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0"}

from app.database import init_db

@app.on_event("startup")
async def startup_event():
    print("Startup: Initializing Database Tables...")
    init_db()

if __name__ == "__main__":
    from app.database import engine, Base
    # Models are already imported at top level, which registers them with Base
    
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    port = int(os.getenv("PORT", 8005))
    print(f"Starting SciNets Server on Port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
