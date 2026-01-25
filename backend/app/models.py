from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
import uuid
import datetime
from app.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Rate limiting
    last_discovery_at = Column(DateTime, nullable=True)
    discoveries_in_window = Column(Integer, default=0)
    window_start_at = Column(DateTime, default=datetime.datetime.utcnow)
    custom_quota_limit = Column(Integer, nullable=True)

    sessions = relationship("Session", back_populates="user")

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Session Binding
    ip_prefix = Column(String, nullable=True) # Store /24 prefix
    user_agent = Column(String, nullable=True)
    
    user = relationship("User", back_populates="sessions")

class DiscoveryRun(Base):
    __tablename__ = "discovery_runs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"))
    query = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="started") # started, completed, failed
    is_demo = Column(Boolean, default=False)
    
    # Enhanced Stats
    thread_id = Column(String, nullable=True) # Explicit alias for ID or separate
    completed_at = Column(DateTime, nullable=True)
    duration = Column(Float, default=0.0)
    cost_usd = Column(Float, default=0.0)
    tokens = Column(Integer, default=0)
    result_path = Column(String, nullable=True)
    
    user = relationship("User", back_populates="discovery_runs")

class MagicLink(Base):
    __tablename__ = "magic_links"
    
    token_hash = Column(String, primary_key=True) # Storing hash of token
    email = Column(String, index=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

User.discovery_runs = relationship("DiscoveryRun", back_populates="user")
