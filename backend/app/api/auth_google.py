from fastapi import APIRouter, HTTPException, Depends, Response, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
import datetime
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, Session as DbSession
from app.config import config
import logging

# Google Auth Libraries
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# Setup Logger
logger = logging.getLogger("scinets")

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/google")
async def google_login(request: Request, response: Response, db: Session = Depends(get_db)):
    google_client_id = os.getenv("GOOGLE_CLIENT_ID")

    logger.info("[Auth] Google login hit")

    try:
        body = await request.json()
        logger.info(f"[Auth] Raw body keys: {list(body.keys())}")
        # logger.info(f"[Auth] Raw body: {body}") # Security risk, usually don't log full body in prod, but fine for debug

        token = body.get("credential") or body.get("token")
        
        logger.info(f"[Auth] Token present: {bool(token)}")
        logger.info(f"[Auth] Token length: {len(token) if token else 'NONE'}")
        logger.info(f"[Auth] Expected client ID prefix: {google_client_id[:10] if google_client_id else 'NONE'}")

        if not google_client_id:
            logger.error("[Auth] GOOGLE_CLIENT_ID not set")
            return JSONResponse(status_code=500, content={"error": "Server configuration error"})

        if not token:
             return JSONResponse(status_code=400, content={"error": "Missing credential/token in body"})

        # Verify token with Google Library
        # This checks signature, expiration, and audience
        idinfo = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            google_client_id
        )

        email = idinfo["email"]
        name = idinfo.get("name")
        
        logger.info(f"[Auth] Google Login Verified: {email}")

        # Upsert User
        user = db.query(User).filter(User.email == email).first()
        if not user:
            logger.info(f"[Auth] Creating new user via Google: {email}")
            user = User(email=email)
            db.add(user)
            db.commit()
            db.refresh(user)

        # Create Session
        session_id = str(uuid.uuid4())
        expires = datetime.datetime.utcnow() + datetime.timedelta(days=7) # 7 Day Session
        
        # Capture IP Prefix (/24)
        client_ip = request.client.host
        ip_parts = client_ip.split('.')
        if len(ip_parts) == 4:
            ip_prefix = ".".join(ip_parts[:3]) 
        else:
            ip_prefix = client_ip 
            
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

        # Set Session Cookie
        # FIX: Cross-Site Cookie Settings for Production
        response.set_cookie(
            key="session_id", 
            value=session_id, 
            httponly=True, 
            max_age=7*24*60*60, 
            samesite="none",      # REQUIRED for Cross-Site
            secure=True           # REQUIRED for SameSite=None
        )

        return {
            "email": email,
            "name": name,
            "id": user.id,
            "message": "Login successful"
        }

    except ValueError as e:
        # Invalid token
        logger.error(f"[Auth] Google Token Verification Failed: {e}")
        return JSONResponse(status_code=401, content={"error": "Invalid Google Token", "details": str(e)})

    except Exception as e:
        # Generic error
        logger.error(f"[Auth] Google Login System Error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

@router.post("/logout")
async def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    """
    Terminates the server-side session and clears cookie.
    """
    try:
        session_id = request.cookies.get("session_id")
        if session_id:
            # Delete from DB
            db.query(DbSession).filter(DbSession.id == session_id).delete()
            db.commit()
            logger.info(f"[Auth] Session terminated: {session_id}")
    except Exception as e:
        logger.error(f"[Auth] Logout DB error: {e}")

    # Always clear cookie
    response.delete_cookie(
        key="session_id", 
        httponly=True, 
        samesite="none", 
        secure=True
    )
    return {"message": "Logged out"}
