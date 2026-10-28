import asyncio
import logging
import os
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.chat_models import init_chat_model
from langsmith import traceable
from pydantic import BaseModel, Field

import flow
from evals.api import router as eval_router
from jev_decision import safe_interpret_answer
from tools import tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = """
You are a helpful restaurant recommendation assistant.

You have access to these tools:

1. Search restaurants by cuisine
2. Get restaurant details
3. Check reservation availability

Use the appropriate tool whenever necessary.

When the user asks where they can get a food (e.g. "where can I get
biryani?"), work out which cuisine the food belongs to, search that
cuisine, then get the details of each restaurant found and recommend
the ones whose speciality includes that food. If none do, say so and
suggest the restaurants of that cuisine instead.

Do not invent restaurant information.

For reservations, never guess the date or party size. If either is
missing, ask the user for it before calling the availability tool;
do not assume a party size such as 2.
You can only check availability, not make bookings: never say a table
is booked, reserved or "all set".
Each restaurant has a few time slots per day; check without a time to
list the free slots, or with a time to check one slot.

If the available tools cannot answer the question,
clearly say that the requested information is outside
your available capabilities.

Be friendly, conversational, and concise.
"""

@dynamic_prompt
def prompt_with_today(request: ModelRequest) -> str:
    # Today's date, so Gemini can resolve "tomorrow" or "this Saturday".
    return f"{system_prompt}\nToday is {date.today():%A %Y-%m-%d}."


agent = create_agent(
    model=init_chat_model(os.getenv("MODEL", "google_genai:gemini-3.5-flash-lite")),
    tools=tools,
    middleware=[prompt_with_today],
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
    history: list[Message] = Field(default_factory=list)  # Jev's conversation
    gemini_history: list[Message] | None = None  # Gemini's own conversation
    state: dict | None = None  # echo back the `state` from the last response


class ChatResponse(BaseModel):
    # Gemini's separate run on the same input: source, available, reply, tool_calls
    google: dict
    # Jev is the agent: source, available, understood, reply, tool_calls, next_question
    jev: dict
    state: dict  # send this back with the next request


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
async def run_gemini(messages: list[dict]) -> dict:
    """Gemini answers with its own tool calls; failures are reported, not raised."""
    try:
        result = await agent.ainvoke({"messages": messages})
    except Exception:
        logger.exception("Gemini agent invocation failed")
        return {"available": False, "reply": "", "tool_calls": []}

    out = result["messages"]

    # Arguments the model passed to each tool, keyed by tool-call id.
    call_args = {
        c["id"]: c.get("args", {})
        for m in out
        for c in (getattr(m, "tool_calls", None) or [])
    }

    tool_calls = [
        {
            "name": m.name,
            "args": call_args.get(getattr(m, "tool_call_id", None), {}),
            "output": _text(m.content),
        }
        for m in out
        if getattr(m, "type", None) == "tool"
    ]

    return {
        "available": True,
        "reply": _text(out[-1].content) if out else "",
        "tool_calls": tool_calls,
    }


def _question_line(question: dict) -> str:
    """The question as plain text, so Gemini sees what the user was asked."""
    options = [c["label"] for c in question["choices"] if c["key"] != "restart"]
    return f"{question['text']} Options: {', '.join(options)}."


@traceable(name="Jev Agent Turn", run_type="chain")
async def run_turn(req: ChatRequest) -> ChatResponse:
    history = [m.model_dump() for m in req.history]
    gemini_history = [m.model_dump() for m in (req.gemini_history or req.history)]
    state = req.state or dict(flow.START)
    asked = flow.question_for(state)

    # Two separate executions on the same input, run at the same time:
    # Jev reads the answer to the question it asked; Gemini answers on its own
    # with its own history and tools. Neither sees the other's output.
    gemini_messages = gemini_history + [
        {"role": "assistant", "content": _question_line(asked)},
        {"role": "user", "content": req.message},
    ]
    jev_result, gemini = await asyncio.gather(
        safe_interpret_answer(asked, req.message, history),
        run_gemini(gemini_messages),
    )

    if jev_result["available"]:
        method = "jev"
        choice = jev_result["choice"]
    else:
        method = "fallback"
        choice = flow.match_label(asked, req.message)

    # Jev's decision drives its tools, its reply and the next question.
    advance = flow.step(state, choice)

    new_state = advance["state"]
    next_question = flow.question_for(new_state)
    if advance["notice"]:
        next_question["text"] = f"{advance['notice']} {next_question['text']}"

    return ChatResponse(
        google={"source": "Google Gemini", **gemini},
        jev={
            "source": "Jev",
            "available": jev_result["available"],
            "understood": {
                "question_id": asked["id"],
                "choice": choice,
                "confidence": jev_result["confidence"],
                "method": method,
            },
            "reply": advance["reply"],
            "tool_calls": [advance["tool"]] if advance["tool"] else [],
            "next_question": next_question,
        },
        state=new_state,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return await run_turn(req)


app.include_router(eval_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
