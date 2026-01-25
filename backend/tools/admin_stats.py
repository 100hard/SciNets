import sys
import os
import argparse
import csv
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Ensure we can import app code
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.append(backend_dir)

from app.models import User, DiscoveryRun
from app.config import config

def migrate_db(engine):
    """Auto-adds new columns if missing (SQLite specific)."""
    with engine.connect() as conn:
        try:
            # Check for new columns
            conn.execute(text("SELECT completed_at, duration, cost_usd, tokens, result_path FROM discovery_runs LIMIT 1"))
        except Exception:
            print("New stats columns missing in discovery_runs. Migrating...")
            columns = [
                ("completed_at", "DATETIME"),
                ("duration", "FLOAT DEFAULT 0.0"),
                ("cost_usd", "FLOAT DEFAULT 0.0"),
                ("tokens", "INTEGER DEFAULT 0"),
                ("result_path", "VARCHAR"),
                ("thread_id", "VARCHAR")
            ]
            for col, dtype in columns:
                try:
                    conn.execute(text(f"ALTER TABLE discovery_runs ADD COLUMN {col} {dtype}"))
                except Exception as e:
                    pass # Ignore if exists (sloppy but works for iterative dev)
            conn.commit()
            print("Migration successful.")

def format_time(dt):
    if not dt: return "-"
    return dt.strftime("%Y-%m-%d %H:%M")

def cmd_users(session):
    users = session.query(User).order_by(User.last_discovery_at.desc()).all()
    print(f"\n{'Email':<30} | {'Used':<5} | {'Limit':<5} | {'Last Active':<16} | {'ID'}")
    print("-" * 80)
    for u in users:
        limit = u.custom_quota_limit if u.custom_quota_limit is not None else config.MAX_RUNS_PER_USER_PER_WEEK
        limit_str = str(limit) + ("*" if u.custom_quota_limit is not None else "")
        print(f"{u.email[:30]:<30} | {u.discoveries_in_window:<5} | {limit_str:<5} | {format_time(u.last_discovery_at):<16} | {u.id}")

def cmd_runs(session, email):
    user = session.query(User).filter(User.email == email).first()
    if not user:
        print(f"User not found: {email}")
        return

    runs = session.query(DiscoveryRun).filter(DiscoveryRun.user_id == user.id).order_by(DiscoveryRun.created_at.desc()).limit(50).all()
    
    print(f"\nRuns for {email} (Last 50):")
    print(f"{'Date':<16} | {'Status':<10} | {'Cost ($)':<8} | {'Tokens':<8} | {'Duration':<8} | {'Query'}")
    print("-" * 100)
    
    for r in runs:
        q = r.query.replace("\n", " ")[:40]
        cost = f"{r.cost_usd:.4f}"
        dur = f"{r.duration:.1f}s"
        print(f"{format_time(r.created_at):<16} | {r.status:<10} | {cost:<8} | {r.tokens:<8} | {dur:<8} | {q}")

def cmd_run_detail(session, thread_id):
    run = session.query(DiscoveryRun).filter(DiscoveryRun.id == thread_id).first()
    if not run:
        print("Run not found.")
        return
        
    print(f"\nRUN DETAIL: {thread_id}")
    print(f"User:      {run.user.email if run.user else 'Unknown'}")
    print(f"Status:    {run.status}")
    print(f"Time:      {format_time(run.created_at)} -> {format_time(run.completed_at)}")
    print(f"Duration:  {run.duration:.2f}s")
    print(f"Cost:      ${run.cost_usd:.5f}")
    print(f"Tokens:    {run.tokens}")
    print(f"Query:     {run.query}")
    print(f"Result:    {run.result_path}")
    
    # Try to verify file existence
    if run.result_path:
        # Adjust path relative to backend root 
        full_path = os.path.join(backend_dir, run.result_path)
        if os.path.exists(full_path):
             print(f"File Size: {os.path.getsize(full_path)} bytes")
        else:
             print(f"File Status: MISSING (Checked {full_path})")

def cmd_export(session, since_date):
    query = session.query(DiscoveryRun)
    if since_date:
        try:
            dt = datetime.strptime(since_date, "%Y-%m-%d")
            query = query.filter(DiscoveryRun.created_at >= dt)
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD.")
            return

    runs = query.order_by(DiscoveryRun.created_at.desc()).all()
    
    filename = f"export_runs_{datetime.now().strftime('%Y%m%d')}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "user_email", "status", "created_at", "completed_at", "duration", "cost_usd", "tokens", "query"])
        
        for r in runs:
            writer.writerow([
                r.id,
                r.user.email if r.user else "Unknown",
                r.status,
                r.created_at,
                r.completed_at,
                r.duration,
                r.cost_usd,
                r.tokens,
                r.query
            ])
            
    print(f"Exported {len(runs)} rows to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("users", help="List user stats")
    
    p_runs = subparsers.add_parser("runs", help="List runs for email")
    p_runs.add_argument("email", type=str)
    
    p_run = subparsers.add_parser("run", help="Run detail")
    p_run.add_argument("thread_id", type=str)
    
    p_export = subparsers.add_parser("export", help="Export to CSV")
    p_export.add_argument("--since", type=str, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    engine = create_engine(config.DATABASE_URL)
    migrate_db(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        if args.command == "users":
            cmd_users(session)
        elif args.command == "runs":
            cmd_runs(session, args.email)
        elif args.command == "run":
            cmd_run_detail(session, args.thread_id)
        elif args.command == "export":
            cmd_export(session, args.since)
        else:
            parser.print_help()
    finally:
        session.close()
