
try:
    from langchain_community.callbacks import get_openai_callback
    print("SUCCESS: langchain_community.callbacks detected")
except ImportError:
    try:
        from langchain.callbacks import get_openai_callback
        print("SUCCESS: langchain.callbacks detected")
    except ImportError:
        print("FAILURE: Could not import get_openai_callback")
