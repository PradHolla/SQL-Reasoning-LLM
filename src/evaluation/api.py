import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI(title="SQL Reasoning Gateway")

# Connect to your local vLLM instance running on port 8000
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed" # vLLM does not require an API key
)

class SQLRequest(BaseModel):
    schema: str
    question: str

@app.post("/generate-stream")
async def generate_sql_stream(req: SQLRequest):
    # 1. Enforce the exact system prompt and formatting from training
    messages = [
        {"role": "system", "content": "You are a database expert. You must think step-by-step inside <think></think> tags, and output ONLY the final SQL query inside <answer></answer> tags."},
        {"role": "user", "content": f"Schema: {req.schema}\nQuestion: {req.question}"}
    ]

    # 2. Create an async generator to yield tokens as vLLM produces them
    async def token_generator():
        stream = await client.chat.completions.create(
            model="models/qwen-0.5b-production-vllm",
            messages=messages,
            max_tokens=512,
            temperature=0.6,
            stream=True # This unlocks continuous streaming (SSE)
        )
        async for chunk in stream:
            # Check if the chunk contains text, then yield it
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    # 3. Return the streaming response to the client
    return StreamingResponse(token_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # We run this gateway on port 8080 so it doesn't clash with vLLM on 8000
    uvicorn.run(app, host="0.0.0.0", port=8080)