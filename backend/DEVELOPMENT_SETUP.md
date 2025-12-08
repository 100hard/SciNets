# SciNets V2 - Debugging & Development Setup

## Quick Start

### 1. Install Development Dependencies

```bash
cd backend
python -m pip install structlog colorama pytest pytest-asyncio pytest-cov pytest-mock debugpy ruff black mypy sentry-sdk[fastapi]
```

### 2. Start Development Server (New Way!)

**Instead of running 7 servers on different ports**, use the dev CLI:

```bash
# Start server with colored logs
python dev.py start

# Start with debug logging
python dev.py start --debug

# Use a different port
python dev.py start --port 8001
```

### 3. Run Tests

```bash
# Run all tests with coverage
python dev.py test

# Watch mode (auto-rerun on file changes)
python dev.py test --watch

# Run specific test file
pytest tests/test_hypothesis.py -v
```

### 4. Clean Debug Files

```bash
# Clean up all those debug log files
python dev.py clean
```

---

## What Changed?

### Structured Logging

Instead of:
```python
print(f"[Hypothesis] Generating hypotheses...")
```

Now:
```python
from app.logging_config import get_logger

log = get_logger(__name__)
log.info("hypothesis_generation_started", query=query, lens=lens)
```

**Benefits:**
- Searchable, filterable logs
- Automatic request ID tracking
- Colored console output in dev
- JSON output for production
- Full stack traces on errors

### Testing Framework

Run tests to catch Pydantic validation errors before starting the server:

```bash
pytest tests/test_sanity.py  # Basic tests
pytest tests/test_hypothesis.py  # Agent tests
```

### VS Code Debugging

Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Server",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["server:app", "--reload"],
      "cwd": "${workspaceFolder}/backend"
    }
  ]
}
```

Press F5 to start debugging, set breakpoints, and step through code!

---

## Environment Variables

Add to your `.env`:

```bash
# Logging
LOG_LEVEL=DEBUG  # or INFO, WARNING, ERROR
JSON_LOGS=false  # true for production (JSON output)

# LangSmith (sign up at smith.langchain.com)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
```

---

## Next Steps (Optional)

### 1. Set up LangSmith

1. Sign up (free): https://smith.langchain.com
2. Get API key from settings
3. Add to `.env`:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=ls_xxx
   ```
4. Run a discovery - see the trace in LangSmith dashboard!

### 2. Set up Sentry (Error Tracking)

1. Sign up (free): https://sentry.io
2. Create project, get DSN
3. Add to `server.py`:
   ```python
   import sentry_sdk

   sentry_sdk.init(dsn="your_dsn_here")
   ```

---

## Troubleshooting

### Logs not colored?
- Make sure `colorama` is installed: `pip install colorama`
- Set `LOG_LEVEL=INFO` in `.env`

### Tests failing?
- Install pytest: `pip install pytest pytest-asyncio`
- Make sure you're in the `backend` directory

### Can't import app modules in tests?
- The `PYTHONPATH` should include `backend/`
- Or install in editable mode: `pip install -e .`

---

## Benefits Summary

| Before | After |
|--------|-------|
| 7 servers on different ports | 1 server with better logging |
| Manual log file inspection | Searchable, colored logs |
| No tests | Pytest with coverage |
| Print debugging | VS Code breakpoints |
| Hours to debug Pydantic errors | Tests catch them immediately |

**Time saved: ~15-20 hours/week**
