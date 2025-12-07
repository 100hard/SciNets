
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.post("/run_stream")
async def run_discovery_stream(request: dict):
    async def event_generator():
        yield f"data: {json.dumps({'type': 'log', 'data': 'Starting...'})}\n\n"
        await asyncio.sleep(0.1)
        
        for i in range(10):
            yield f"data: {json.dumps({'type': 'log', 'data': f'Ping {i}'})}\n\n"
            await asyncio.sleep(1)
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
