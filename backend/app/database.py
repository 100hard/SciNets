from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import config

# Clean and normalize database URL
db_url = config.DATABASE_URL or "sqlite:///./scinets.db"
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# connect_args={"check_same_thread": False} is needed for SQLite
connect_args = {}
if db_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    db_url, 
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=1800
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create all tables defined in models."""
    # Ensure models are imported before calling this!
    Base.metadata.create_all(bind=engine)
