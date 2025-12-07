
import asyncio
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class TestSchema(BaseModel):
    reason: str = Field(description="Why this is a test")
    score: int = Field(description="A score from 1 to 10")

async def test_structure():
    print("Attempting STRUCTURED OUTPUT with model: gpt-5-mini")
    try:
        llm = ChatOpenAI(
            model="gpt-5-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        structured_llm = llm.with_structured_output(TestSchema)
        
        print("Invoking...")
        result = await structured_llm.ainvoke("Rate this test.")
        print("Success!")
        print(f"Result: {result}")
    except Exception as e:
        print("\n--- STRUCTURED ERROR DETECTED ---")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {e}")
        print("---------------------------------\n")

if __name__ == "__main__":
    asyncio.run(test_structure())
