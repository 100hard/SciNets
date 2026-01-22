
import uvicorn
import os
import sys

# Add backend to path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the app from server.py 
# This will trigger the side-effects in server.py (logging setup etc) which is fine
from server import app

if __name__ == "__main__":
    print("Starting Test Server on 8010...")
    uvicorn.run(app, host="127.0.0.1", port=8010, log_level="info")
