import sys
import os
import argparse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Ensure we can import app code
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.append(backend_dir)

from app.models import User
from app.config import config
from app.database import Base

def migrate_db(engine):
    """Checks if custom_quota_limit exists, adds it if not (SQLite specific)."""
    with engine.connect() as conn:
        try:
            # Check if column exists by trying to select it
            conn.execute(text("SELECT custom_quota_limit FROM users LIMIT 1"))
        except Exception:
            print("Column 'custom_quota_limit' missing. Adding it...")
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN custom_quota_limit INTEGER"))
                conn.commit()
                print("Migration successful.")
            except Exception as e:
                print(f"Migration failed: {e}")

def set_quota(email, limit):
    engine = create_engine(config.DATABASE_URL)
    
    # 1. Ensure Schema is Valid
    migrate_db(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        user = session.query(User).filter(User.email == email).first()
        if not user:
            print(f"User not found: {email}")
            # Optional: Create user?
            create = input("Create user? (y/n): ")
            if create.lower() == 'y':
                user = User(email=email)
                session.add(user)
            else:
                return

        if limit < 0:
            user.custom_quota_limit = None
            print(f"Reset quota for {email} to default.")
        else:
            user.custom_quota_limit = limit
            print(f"Set quota for {email} to {limit}.")
        
        session.commit()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set custom quota for a user.")
    parser.add_argument("email", type=str, help="User email")
    parser.add_argument("limit", type=int, help="New limit (use -1 to reset to default)")
    
    args = parser.parse_args()
    set_quota(args.email, args.limit)
