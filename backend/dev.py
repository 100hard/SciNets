#!/usr/bin/env python3
"""
SciNets Development CLI

A unified development script to replace multiple server instances and manual workflows.

Usage:
    python dev.py start [--port 8000] [--debug]  # Start dev server
    python dev.py test [--watch] [--coverage]    # Run tests
    python dev.py logs [--level INFO]            # View logs with filtering
    python dev.py clean                          # Clean debug files and logs
    python dev.py stop                           # Kill all lingering Python processes
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def start_server(port: int = 8000, debug: bool = False):
    """Start the development server with auto-reload."""
    print(f"🚀 Starting SciNets server on port {port}...")
    
    env = os.environ.copy()
    
    # Enable debug logging
    if debug:
        env["LOG_LEVEL"] = "DEBUG"
        env["PYTHONUNBUFFERED"] = "1"
    
    # Set Python path
    backend_dir = Path(__file__).parent
    env["PYTHONPATH"] = str(backend_dir)
    
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--reload",
        "--timeout-keep-alive",
        "300",
    ]
    
    if debug:
        cmd.extend(["--log-level", "debug"])
    
    try:
        subprocess.run(cmd, cwd=backend_dir, env=env, check=True)
    except KeyboardInterrupt:
        print("\n✅ Server stopped")


def run_tests(watch: bool = False, coverage: bool = True, verbose: bool = True):
    """Run pytest tests."""
    print("🧪 Running tests...")
    
    backend_dir = Path(__file__).parent
    
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing", "--cov-report=html"])
    
    if watch:
        # Install pytest-watch if not available
        try:
            import pytest_watch
        except ImportError:
            print("Installing pytest-watch...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest-watch"], check=True)
        
        cmd = [sys.executable, "-m", "ptw", "--", "-v"]
        if coverage:
            cmd.extend(["--cov=app"])
    
    subprocess.run(cmd, cwd=backend_dir)


def view_logs(level: str = "INFO"):
    """View and filter logs."""
    print(f"📋 Viewing logs (level: {level})...")
    # For structured logs, we can add filtering logic here
    print("Note: Install 'jq' for advanced JSON log filtering")
    print("Example: python dev.py start --debug | jq 'select(.level==\"error\")'")


def clean_debug_files():
    """Clean up debug files and log files."""
    print("🧹 Cleaning debug files...")
    
    backend_dir = Path(__file__).parent
    
    patterns = [
        "server_debug_*.log",
        "*.pyc",
        "__pycache__",
        ".pytest_cache",
        "htmlcov",
        ".coverage",
    ]
    
    removed = 0
    for pattern in patterns:
        for path in backend_dir.rglob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
                print(f"  Removed: {path.name}")
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
                removed += 1
                print(f"  Removed: {path.name}/")
    
    print(f"✅ Cleaned {removed} items")


def install_deps(dev: bool = True):
    """Install dependencies."""
    print("📦 Installing dependencies...")
    
    backend_dir = Path(__file__).parent
    
    if dev:
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".[dev]"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    
    subprocess.run(cmd, cwd=backend_dir, check=True)
    print("✅ Dependencies installed")


def run_debug(script: str):
    """Run a script in debug mode with logging to file."""
    print(f"🐞 Debugging {script}...")
    
    backend_dir = Path(__file__).parent
    script_path = backend_dir / script
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return

    env = os.environ.copy()
    env["LOG_LEVEL"] = "DEBUG"
    env["LOG_FILE"] = "debug.log"
    env["PYTHONPATH"] = str(backend_dir)
    
    cmd = [sys.executable, str(script_path)]
    
    try:
        # Run with output to console (and logging capturing it to file via app config)
        subprocess.run(cmd, cwd=backend_dir, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Debug run failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n⚠️ Debug run interrupted")



def main():
    parser = argparse.ArgumentParser(description="SciNets Development CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start development server")
    start_parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    start_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--watch", action="store_true", help="Watch mode (auto-rerun on changes)")
    test_parser.add_argument("--no-coverage", action="store_true", help="Disable coverage report")
    test_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View logs")
    logs_parser.add_argument("--level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # Clean command
    subparsers.add_parser("clean", help="Clean debug files and logs")
    
    # Stop command
    subparsers.add_parser("stop", help="Stop all python servers")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install dependencies")
    install_parser.add_argument("--no-dev", action="store_true", help="Skip dev dependencies")
    
    # Debug command
    debug_parser = subparsers.add_parser("debug", help="Run script in debug mode")
    debug_parser.add_argument("script", help="Path to script to run (relative to backend/)")

    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == "start":
        start_server(port=args.port, debug=args.debug)
    elif args.command == "test":
        run_tests(watch=args.watch, coverage=not args.no_coverage, verbose=args.verbose)
    elif args.command == "logs":
        view_logs(level=args.level)
    elif args.command == "clean":
        clean_debug_files()
    elif args.command == "stop":
        stop_servers()
    elif args.command == "install":
        install_deps(dev=not args.no_dev)
    elif args.command == "debug":
        run_debug(args.script)


if __name__ == "__main__":
    main()
