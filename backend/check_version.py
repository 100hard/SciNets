
try:
    from langgraph.prebuilt import create_react_agent
    import inspect
    print(f"Signature: {inspect.signature(create_react_agent)}")
except Exception as e:
    print(f"Error: {e}")
