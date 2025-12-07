
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from app.memory import MemoryManager
    print("MemoryManager imported.")
    mm = MemoryManager()
    print("MemoryManager instantiated.")
except Exception as e:
    print(f"MemoryManager FAIL: {e}")
