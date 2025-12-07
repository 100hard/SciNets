
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

print("Attempting to use model: gpt-5-mini")
try:
    llm = ChatOpenAI(
        model="gpt-5-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    response = llm.invoke("Hello, are you there?")
    print("Success!")
    print(response)
except Exception as e:
    print("\n--- ERROR DETECTED ---")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {e}")
    print("----------------------\n")
