
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
from app.database import SessionLocal, engine, Base
from app.models import User, Session as DbSession, DiscoveryRun, MagicLink
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.auth_utils import create_magic_link_token, verify_magic_link_token, send_magic_link_email, hash_token
from app.config import config as app_config
import datetime
from fastapi import Response, Request, Depends, Cookie, status

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
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ],
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



# =============================================================================
# Database Selection
# =============================================================================
# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================================================================
# Auth Models
# =============================================================================
class EmailRequest(BaseModel):
    email: str

class VerifyRequest(BaseModel):
    token: str

class UserResponse(BaseModel):
    id: str
    email: str
    created_at: datetime.datetime

async def get_current_user(session_id: str | None = Cookie(default=None), db: Session = Depends(get_db)):
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session = db.query(DbSession).filter(DbSession.id == session_id).first()
    if not session:
        # Invalid session
        raise HTTPException(status_code=401, detail="Session invalid")
        
    if session.expires_at < datetime.datetime.utcnow():
        # Expired session
        db.delete(session)
        db.commit()
        raise HTTPException(status_code=401, detail="Session expired")
        
    return session.user

async def get_current_admin(user: User = Depends(get_current_user)):
    """Dependency to check if user is admin."""
    # Ensure ADMIN_EMAILS is a list
    admin_list = app_config.ADMIN_EMAILS
    if isinstance(admin_list, str):
        # Fallback if config parsing failed or env var was raw string
        try:
            admin_list = json.loads(admin_list)
        except:
             admin_list = [admin_list]

    if user.email not in admin_list:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user

async def dep_enforce_rate_limit(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not app_config.ENABLE_DISCOVERY:
         # Check if admin to bypass kill switch
         admin_list = app_config.ADMIN_EMAILS
         if isinstance(admin_list, str):
            try:
                admin_list = json.loads(admin_list)
            except:
                 admin_list = [admin_list]
                 
         if user.email not in admin_list:
             raise HTTPException(status_code=503, detail={"error": "DISCOVERY_DISABLED"})

    # Rate Limit Logic
    now = datetime.datetime.utcnow()
    # Default window start to now if none
    if not user.window_start_at:
        user.window_start_at = now
        user.discoveries_in_window = 0
        
    window_start = user.window_start_at
    
    # Check if window expired
    if (now - window_start).days >= app_config.WINDOW_LENGTH_DAYS:
        # Reset window
        user.discoveries_in_window = 0
        user.window_start_at = now
        db.commit()
    
    # Check Limit
    if user.discoveries_in_window >= app_config.MAX_DISCOVERIES_PER_WINDOW:
        # Check admin bypass
        admin_list = app_config.ADMIN_EMAILS
        if isinstance(admin_list, str):
             try: admin_list = json.loads(admin_list)
             except: admin_list = [admin_list]

        if user.email not in admin_list:
            unlocks_at = user.window_start_at + datetime.timedelta(days=app_config.WINDOW_LENGTH_DAYS)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "LIMIT_REACHED",
                    "unlocks_at": unlocks_at.isoformat()
                }
            )

    # Increment (optimistic)
    user.discoveries_in_window += 1
    user.last_discovery_at = now
    db.commit()
    
    return user


@app.post("/search_papers")
async def search_papers(request: SearchPapersRequest):
    """
    Fetch candidate papers BEFORE full discovery.
    
    This allows users to curate which papers go into the analysis.
    Returns a list of candidate papers with metadata.
    """
    logger.info(f"[SearchPapers] Fetching papers for: {request.query}")
    
    try:
        from app.tools.search import search_papers as openalex_search
        
        # Use query directly - LLM refinement was causing 500 errors
        search_query = request.query
        
        logger.info(f"[SearchPapers] Searching OpenAlex for: '{search_query}'")
        
        # Search OpenAlex
        papers = await openalex_search(search_query, limit=request.max_papers)
        
        logger.info(f"[SearchPapers] Found {len(papers)} papers for query '{search_query}'")

        # FALLBACK: If 0 results, try simplified keywords
        if len(papers) == 0:
            logger.info("[SearchPapers] 0 results. Use simplified keyword search fallback.")
            # Simple stopword removal
            stopwords = ["find", "mechanism", "connecting", "to", "the", "a", "an", "and", "or", "of", "in", "for", "with"]
            keywords = [w for w in search_query.lower().split() if w not in stopwords]
            simple_query = " ".join(keywords)
            
            logger.info(f"[SearchPapers] Fallback Query: '{simple_query}'")
            papers = await openalex_search(simple_query, limit=request.max_papers)
            logger.info(f"[SearchPapers] Fallback Found {len(papers)} papers")

        # Convert to CandidatePaper format
        candidates = []
        for paper in papers:
            # The 'abstract' field from openalex.py is actually the inverted index
            inverted_index = paper.get("abstract")
            abstract = ""
            if inverted_index and isinstance(inverted_index, dict):
                try:
                    word_positions = []
                    for word, positions in inverted_index.items():
                        for pos in positions:
                            word_positions.append((pos, word))
                    word_positions.sort()
                    abstract = " ".join(w for _, w in word_positions)
                except Exception as e:
                    logger.warning(f"Error reconstructing abstract: {e}")

            venue = paper.get("host_venue") or "Unknown venue"
            
            # Return dict directly to avoid Pydantic validation issues if any
            candidates.append({
                "id": paper.get("id"), # keep original ID
                "title": paper.get("title") or "Untitled",
                "year": paper.get("publication_year") or 2024,
                "venue": venue,
                "abstract": abstract[:500] if abstract else "No abstract available",
                "rationale": abstract[:200] + "..." if abstract else "Relevant to search query",
                "selected": True,
                "locked": False
            })
        
        logger.info(f"[SearchPapers] Returning {len(candidates)} candidates to frontend")
        return {"papers": candidates, "refined_query": search_query}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"[SearchPapers] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Paper search failed: {e}")


@app.post("/run_stream")
async def run_discovery_stream(request: RunRequest, user: User = Depends(dep_enforce_rate_limit), db: Session = Depends(get_db)):
    """
    Trigger the discovery loop and stream events.
    Rate limited via dependency.
    """
    
    # Log Discovery Run
    run_id = str(uuid.uuid4())
    discovery_run = DiscoveryRun(
        id=run_id,
        user_id=user.id,
        query=request.query,
        status="started",
        is_demo=False
    )
    db.add(discovery_run)
    db.commit()

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
        
        discovery_run.status = "failed"
        db.commit()
        
        raise HTTPException(status_code=500, detail=f"Init failed: {e}")

    async def event_generator():
        # Emit initial thinking event
        yield f"data: {json.dumps({'type': 'activity', 'data': {'agent': 'orchestrator', 'action': f'Starting discovery for: {request.query}', 'status': 'thinking'}})}\n\n"
        await asyncio.sleep(0.1) # Force flush
        
        success = False
        
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
                    success = True

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            
            # DB Update: Failed (Need new session or handle carefully as generator)
            # We can't reuse the injected db session easily if it's closed, but here generator scope matters
            # For simplicity, we just won't update to Completed.
            # Ideally we'd commit via scoped session.
            
        yield "data: [DONE]\n\n"
        
        # Post-run DB update (Status)
        try:
             # Need a fresh DB session here ideally if the stream runs long
             # Reusing 'db' depends on fastapi dependency scope lifetime which matches request
             # So 'db' should still be valid until response closes
             discovery_run.status = "completed" if success else "failed"
             db.commit()
        except:
             pass

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


@app.post("/api/auth/request-link")
async def request_magic_link(req: EmailRequest, db: Session = Depends(get_db)):
    """Generates a magic link, stores hash, and sends via SMTP."""
    email = req.email.strip().lower()
    
    # Check if user exists, if not create
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(email=email)
        db.add(user)
        db.commit()
    
    # 1. Generate Token
    token = create_magic_link_token(email)
    
    # 2. Store Hash (Stateful)
    hashed = hash_token(token)
    expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=app_config.MAGIC_LINK_EXPIRE_MINUTES)
    
    # Remove old tokens for this email (optional, cleanup)
    db.query(MagicLink).filter(MagicLink.email == email).delete()
    
    magic_link_record = MagicLink(token_hash=hashed, email=email, expires_at=expires)
    db.add(magic_link_record)
    db.commit()
    
    # 3. Construct Link
    base_url = app_config.FRONTEND_URL
    link = f"{base_url}/verify?token={token}"
    
    # 4. Send Email
    send_magic_link_email(email, link)
    
    return {"message": "Magic link sent"}

@app.post("/api/auth/verify-link")
async def verify_magic_link(req: VerifyRequest, response: Response, db: Session = Depends(get_db)):
    """Verifies token (stateful check), sets session cookie."""
    
    # 1. Stateless signature check
    email = verify_magic_link_token(req.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired token (sig)")
    
    # 2. Stateful check (Single Use)
    hashed = hash_token(req.token)
    record = db.query(MagicLink).filter(MagicLink.token_hash == hashed).first()
    
    if not record:
        raise HTTPException(status_code=400, detail="Token invalid or already used")
        
    if record.expires_at < datetime.datetime.utcnow():
        db.delete(record)
        db.commit()
        raise HTTPException(status_code=400, detail="Token expired")
    
    # 3. User check
    user = db.query(User).filter(User.email == email).first()
    if not user:
         raise HTTPException(status_code=404, detail="User not found")
         
    # 4. Create Session
    expires = datetime.datetime.utcnow() + datetime.timedelta(days=app_config.SESSION_EXPIRE_DAYS)
    session = DbSession(user_id=user.id, expires_at=expires)
    db.add(session)
    
    # 5. Consume Token (Delete)
    db.delete(record)
    
    db.commit()
    
    # 6. Set Cookie
    response.set_cookie(
        key="session_id",
        value=session.id,
        httponly=True,
        secure=app_config.SECRET_KEY != "dev-secret-key-change-in-prod", # Secure if prod
        samesite="lax",
        expires=app_config.SESSION_EXPIRE_DAYS * 24 * 60 * 60
    )
    
    return {"message": "Logged in", "user": {"email": user.email, "id": user.id}}


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    return user

@app.post("/api/auth/logout")
async def logout(response: Response, session_id: str | None = Cookie(default=None), db: Session = Depends(get_db)):
    if session_id:
        db.query(DbSession).filter(DbSession.id == session_id).delete()
        db.commit()
        
    response.delete_cookie("session_id")
    return {"message": "Logged out"}


# =============================================================================
# Admin Endpoints
# =============================================================================
@app.get("/api/admin/stats")
async def admin_stats(admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Admin only: Usage statistics."""
    total_users = db.query(User).count()
    total_runs = db.query(DiscoveryRun).count()
    
    # Active Users (Last 7 days)
    seven_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    active_7d = db.query(User).filter(User.last_discovery_at >= seven_days_ago).count()
    
    # Active Users (Last 30 days)
    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    active_30d = db.query(User).filter(User.last_discovery_at >= thirty_days_ago).count()
    
    # Repeat Users (>1 discovery)
    repeat_users = db.query(User).filter(User.discoveries_in_window > 1).count()

    return {
        "total_users": total_users,
        "total_discoveries": total_runs,
        "repeat_users": repeat_users,
        "active_users_7d": active_7d,
        "active_users_30d": active_30d
    }

@app.post("/api/admin/reset-limit")
async def reset_limit(email: str, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Admin only: Reset rate limits for a user."""
    target = db.query(User).filter(User.email == email).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    
    target.discoveries_in_window = 0
    target.window_start_at = datetime.datetime.utcnow()
    db.commit()
    return {"message": f"Limits reset for {email}"}

@app.post("/api/admin/disable-discovery")
async def disable_discovery(admin: User = Depends(get_current_admin)):
    app_config.ENABLE_DISCOVERY = False
    return {"message": "Discovery DISABLED"}

@app.post("/api/admin/enable-discovery")
async def enable_discovery(admin: User = Depends(get_current_admin)):
    app_config.ENABLE_DISCOVERY = True
    return {"message": "Discovery ENABLED"}


# =============================================================================
# Contact & Demo
# =============================================================================
class ContactRequest(BaseModel):
    name: str
    email: str
    message: str

@app.post("/api/contact")
async def contact_us(req: ContactRequest):
    """
    Handle contact form submissions.
    Saves message to local file system for now.
    """
    logger.info(f"[Contact] From: {req.name} <{req.email}>")
    
    # Save to file
    try:
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "name": req.name,
            "email": req.email,
            "message": req.message
        }
        
        with open("contact_messages.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            
        logger.info(f"[Contact] Message saved to contact_messages.jsonl")
        return {"message": "Message sent successfully"}
        
    except Exception as e:
        logger.error(f"[Contact] Failed to save message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")

@app.get("/api/demo/{slug}")
async def get_demo_run(slug: str):
    """Returns static demo data."""
    # Assuming files are in backend/app/demo_runs/{slug}.json
    import os
    # Secure path traversal check (basic)
    if ".." in slug or "/" in slug:
         raise HTTPException(status_code=400, detail="Invalid slug")
         
    path = f"app/demo_runs/{slug}.json"
    if not os.path.exists(path):
         raise HTTPException(status_code=404, detail="Demo not found")
    
    with open(path, "r") as f:
        data = json.load(f)
    return data


