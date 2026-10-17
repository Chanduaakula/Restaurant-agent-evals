import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel

from tools import tools

load_dotenv()

system_prompt = """
You are a helpful restaurant recommendation assistant.

You have access to these tools:

1. Search restaurants by cuisine
2. Get restaurant details
3. Check reservation availability

Use the appropriate tool whenever necessary.

Do not invent restaurant information.

If the available tools cannot answer the question,
clearly say that the requested information is outside
your available capabilities.

Be friendly, conversational, and concise.
"""

# needs GOOGLE_API_KEY in the environment
model = init_chat_model(os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite"))

agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)

app = FastAPI(title="Restaurant Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []  # prior turns, kept by the frontend


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = [m.model_dump() for m in req.history]
    messages.append({"role": "user", "content": req.message})
    try:
        result = await agent.ainvoke({"messages": messages})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent error: {e}")

    out = result["messages"]
    tool_calls = [
        {"name": m.name, "output": _text(m.content)} for m in out if m.type == "tool"
    ]
    return ChatResponse(reply=_text(out[-1].content), tool_calls=tool_calls)


@app.get("/health")
async def health():
    return {"status": "ok"}

