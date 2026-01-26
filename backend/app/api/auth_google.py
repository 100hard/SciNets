from fastapi import APIRouter, HTTPException, Depends, Response, Request
from pydantic import BaseModel
import os
import requests
import uuid
import datetime
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, Session as DbSession
from app.config import config
import logging

# Setup Logger
logger = logging.getLogger("scinets")

router = APIRouter(prefix="/api/auth", tags=["auth"])

class GoogleTokenRequest(BaseModel):
    credential: str

@router.post("/google")
def google_login(data: GoogleTokenRequest, response: Response, request: Request, db: Session = Depends(get_db)):
    google_client_id = os.getenv("GOOGLE_CLIENT_ID")

    if not google_client_id:
        raise HTTPException(status_code=500, detail="Google Client ID not configured")

    # Verify token with Google
    try:
        resp = requests.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": data.credential},
            timeout=10
        )
    except Exception as e:
        logger.error(f"[Auth] Google verification error: {e}")
        raise HTTPException(status_code=500, detail="Failed to reach Google Auth")

    if resp.status_code != 200:
        logger.warning(f"[Auth] Invalid Google Token: {resp.text}")
        raise HTTPException(status_code=401, detail="Invalid Google token")

    payload = resp.json()

    # Validate Audience
    if payload["aud"] != google_client_id:
        logger.warning(f"[Auth] Audience mismatch. Expected {google_client_id}, got {payload['aud']}")
        raise HTTPException(status_code=401, detail="Token audience mismatch")

    email = payload["email"]
    name = payload.get("name")
    
    logger.info(f"[Auth] Google Login Verified: {email}")

    # Upsert User
    user = db.query(User).filter(User.email == email).first()
    if not user:
        logger.info(f"[Auth] Creating new user via Google: {email}")
        user = User(email=email)
        # Use random quota/defaults
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
        ip_prefix = ".".join(ip_parts[:3]) # 192.168.1
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
    response.set_cookie(
        key="session_id", 
        value=session_id, 
        httponly=True, 
        max_age=7*24*60*60, 
        samesite="lax",
        secure=not config.DEMO_MODE # Secure in Prod
    )

    return {
        "email": email,
        "name": name,
        "id": user.id,
        "message": "Login successful"
    }
