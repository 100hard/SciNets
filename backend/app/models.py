from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean
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

    sessions = relationship("Session", back_populates="user")

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime)
    
    user = relationship("User", back_populates="sessions")

class DiscoveryRun(Base):
    __tablename__ = "discovery_runs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"))
    query = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="started") # started, completed, failed
    is_demo = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="discovery_runs")

class MagicLink(Base):
    __tablename__ = "magic_links"
    
    token_hash = Column(String, primary_key=True) # Storing hash of token
    email = Column(String, index=True)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

User.discovery_runs = relationship("DiscoveryRun", back_populates="user")
