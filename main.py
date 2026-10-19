
import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langsmith import traceable
from pydantic import BaseModel, Field

from tools import tools
from jev_decision import safe_classify_restaurant_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

model = init_chat_model(
    os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite")
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

app = FastAPI(title="Restaurant Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict]
    jev: dict


def _text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return str(content) if content is not None else ""


@traceable(name="Gemini Restaurant Agent", run_type="chain")
async def invoke_gemini_agent(messages: list[dict]):
    return await agent.ainvoke({"messages": messages})


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = [m.model_dump() for m in req.history]
    messages.append({"role": "user", "content": req.message})

    # Run both systems independently.
    gemini_task = invoke_gemini_agent(messages)
    jev_task = safe_classify_restaurant_request(req.message)

    try:
        result, jev_result = await asyncio.gather(
            gemini_task,
            jev_task,
        )
    except Exception:
        logger.exception("Gemini agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The restaurant assistant could not process your request.",
        )

    out = result["messages"]

    tool_calls = [
        {
            "name": m.name,
            "output": _text(m.content),
        }
        for m in out
        if getattr(m, "type", None) == "tool"
    ]

    reply = _text(out[-1].content) if out else ""

    return ChatResponse(
        reply=reply,
        tool_calls=tool_calls,
        jev=jev_result,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
