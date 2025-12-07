
try:
    from langchain.callbacks import get_openai_callback
    print("Found in langchain.callbacks")
except ImportError:
    print("Not in langchain.callbacks")

try:
    from langchain_community.callbacks import get_openai_callback
    print("Found in langchain_community.callbacks")
except ImportError:
    print("Not in langchain_community.callbacks")
