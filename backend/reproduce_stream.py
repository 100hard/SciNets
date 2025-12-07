
import asyncio
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

async def test_stream():
    print("Attempting to STREAM with model: gpt-5-mini")
    try:
        llm = ChatOpenAI(
            model="gpt-5-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            streaming=True
        )
        print("Starting stream...")
        async for chunk in llm.astream("Write a haiku about bandwidth."):
            print(f"Chunk: {chunk.content}", end="|", flush=True)
        print("\nStream complete.")
    except Exception as e:
        print("\n--- STREAM ERROR DETECTED ---")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {e}")
        print("----------------------------\n")

if __name__ == "__main__":
    asyncio.run(test_stream())
