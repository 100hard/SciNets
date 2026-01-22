import jwt
import datetime
import smtplib
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.config import config
import logging

logger = logging.getLogger("scinets")

def create_magic_link_token(email: str) -> str:
    """Creates a JWT token for magic links."""
    payload = {
        "sub": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=config.MAGIC_LINK_EXPIRE_MINUTES),
        "iat": datetime.datetime.utcnow(),
        "type": "magic_link"
    }
    encoded_jwt = jwt.encode(payload, config.SECRET_KEY, algorithm="HS256")
    return encoded_jwt

def verify_magic_link_token(token: str) -> str | None:
    """Verifies JWT token and returns email if valid."""
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=["HS256"])
        if payload.get("type") != "magic_link":
            return None
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        logger.warning("[Auth] Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("[Auth] Invalid token")
        return None

def hash_token(token: str) -> str:
    """Hashes the token for storage."""
    return hashlib.sha256(token.encode()).hexdigest()

def send_magic_link_email(email: str, link: str):
    """Sends magic link via SMTP (or log if config missing)."""
    
    # Always log for dev visibility/backup
    logger.info(f"========== LOGIN LINK ==========")
    logger.info(f"To: {email}")
    logger.info(f"Link: {link}")
    logger.info(f"================================")

    if not config.SMTP_EMAIL or not config.SMTP_PASSWORD:
        logger.warning("[Auth] SMTP credentials missing. Email NOT sent.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = f"SciNets <{config.SMTP_EMAIL}>"
        msg['To'] = email
        msg['Subject'] = "Your SciNets login link"

        body = f"""Hi,

Here's your secure login link for SciNets:

{link}

This link expires in {config.MAGIC_LINK_EXPIRE_MINUTES} minutes and can be used once.

SciNets is an experimental autonomous scientific discovery system. 
Outputs are hypotheses, not conclusions.

- SciNets
"""
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_EMAIL, config.SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"[Auth] Email sent to {email}")
        
    except Exception as e:
        logger.error(f"[Auth] Failed to send email: {e}")

