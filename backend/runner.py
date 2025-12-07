
import uvicorn
import os
from dotenv import load_dotenv

# Ensure we are in the backend directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

if __name__ == "__main__":
    print("Starting Uvicorn via runner...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8002, reload=False)
