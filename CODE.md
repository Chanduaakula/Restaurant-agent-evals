# Restaurant Assistant Backend

FastAPI backend. `POST /chat` returns three labeled parts: `google` (Gemini agent answer and tool calls), `jev` (what Jev understood from the user's answer, plus the next question in Jev `Choice` format) and `state` (send it back with the next request).

Guided flow: Q1 cuisine (tool 1) -> Q2 restaurant -> Q3 date -> Q4 party size -> availability check. If the user dislikes the options, or the table is fully booked, the flow restarts at Q1.

Endpoints: `POST /chat` and `GET /health`. Run: `uvicorn main:app --reload`.

Environment variables (set in `.env` locally or in the host's settings; never commit them): `GOOGLE_API_KEY`, `OPENROUTER_API_KEY` (Jev via OpenRouter; or `TYPESAFE_API_KEY`), `CORS_ORIGINS`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.

## main.py

```python

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
import flow
from jev_decision import safe_interpret_answer

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

For reservations, never guess the date or party size. If either is
missing, ask the user for it before calling the availability tool.

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
    state: dict | None = None  # echo back the `state` from the last response


class ChatResponse(BaseModel):
    google: dict  # Gemini's answer: source, reply, tool_calls
    jev: dict     # Jev: source, available, understood, next_question
    state: dict   # send this back with the next request


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
    history = [m.model_dump() for m in req.history]
    state = req.state or dict(flow.START)

    # Jev reads the user's answer to the question it asked last turn.
    asked = flow.question_for(state)
    jev_result = await safe_interpret_answer(asked, req.message, history)
    if jev_result["available"]:
        method = "jev"
        choice = jev_result["choice"]
    else:
        method = "fallback"
        choice = flow.match_label(asked, req.message)

    advance = flow.step(state, choice)

    # Google answers the resolved step (or the user's own words if unresolved).
    messages = history + [
        {"role": "user", "content": advance["gemini_message"] or req.message}
    ]

    try:
        result = await invoke_gemini_agent(messages)
    except Exception:
        logger.exception("Gemini agent invocation failed")
        raise HTTPException(
            status_code=502,
            detail="The restaurant assistant could not process your request.",
        )

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

    reply = _text(out[-1].content) if out else ""

    new_state = advance["state"]
    next_question = flow.question_for(new_state)
    if advance["notice"]:
        next_question["text"] = f"{advance['notice']} {next_question['text']}"

    return ChatResponse(
        google={"source": "Google Gemini", "reply": reply, "tool_calls": tool_calls},
        jev={
            "source": "Jev",
            "available": jev_result["available"],
            "understood": {
                "question_id": asked["id"],
                "choice": choice,
                "confidence": jev_result["confidence"],
                "method": method,
            },
            "next_question": next_question,
        },
        state=new_state,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
```

## flow.py

```python
"""The guided booking conversation, asked as Jev-style Choice questions.

Q1 cuisine (tool 1) -> Q2 restaurant -> Q3 date -> Q4 party size -> booking.
If the user dislikes the options, or the table is fully booked, the flow
restarts at Q1. The server is stateless: `state` travels with each request.
"""

from tools import check_reservation_availability, search_restaurants_by_cuisine

CUISINES = ["Indian", "Italian", "Chinese", "Mexican"]

# Dates that exist in the sample availability data in tools.py.
DATES = [
    ("15 Nov 2024", "2024-11-15"),
    ("16 Nov 2024", "2024-11-16"),
    ("17 Nov 2024", "2024-11-17"),
]

PARTY_SIZES = [2, 4, 6]

START = {"asking": "cuisine"}

RESTART_LABEL = "None of these - start over"
RESTART_DESCRIPTION = (
    "The user dislikes these options, wants something different, "
    "or wants to start over"
)


def _choice(key: str, label: str, description: str) -> dict:
    # `message` is what the frontend sends when the user clicks the choice.
    return {"key": key, "label": label, "description": description, "message": label}


def _question(qid: str, text: str, choices: list[dict], restartable: bool = True) -> dict:
    if restartable:
        choices = choices + [
            _choice("restart", RESTART_LABEL, RESTART_DESCRIPTION)
        ]
    return {"id": qid, "type": "choice", "text": text, "choices": choices}


def restaurants_for(cuisine: str) -> list[str]:
    found = search_restaurants_by_cuisine.invoke({"cuisine_type": cuisine})
    return [r["name"] for r in found]


def question_for(state: dict) -> dict:
    """The Jev-format question the user is being asked right now."""
    asking = state.get("asking", "cuisine")

    if asking == "restaurant":
        cuisine = state["cuisine"]
        return _question(
            "restaurant",
            f"Which {cuisine} restaurant would you like?",
            [_choice(n, n, f"The restaurant {n}") for n in restaurants_for(cuisine)],
        )

    if asking == "date":
        return _question(
            "date",
            f"When would you like to book {state['restaurant']}?",
            [_choice(iso, label, f"The date {label}") for label, iso in DATES],
        )

    if asking == "party":
        return _question(
            "party",
            "How many people?",
            [_choice(str(n), f"{n} people", f"A party of {n}") for n in PARTY_SIZES],
        )

    return _question(
        "cuisine",
        "What cuisine would you like?",
        [_choice(c.lower(), c, f"{c} cuisine") for c in CUISINES],
        restartable=False,
    )


def match_label(question: dict, message: str) -> str | None:
    """Fallback when Jev is unavailable: match the text against the choices."""
    text = message.strip().lower()
    for c in question["choices"]:
        if text in (c["label"].lower(), c["key"].lower()):
            return c["key"]
    return None


def step(state: dict, choice: str | None) -> dict:
    """Advance the flow by one answer.

    Returns {state, gemini_message, notice}. `gemini_message` is what Google
    Gemini is asked this turn (None = pass through what the user typed);
    `notice` is a line shown before the next question.
    """
    asking = state.get("asking", "cuisine")

    if choice in (None, "other"):
        # Free text: Gemini answers it, then the same question is asked again.
        return {"state": state, "gemini_message": None, "notice": None}

    if choice == "restart" and asking != "cuisine":
        return {
            "state": dict(START),
            "gemini_message": (
                "I don't like these options and want to start over. Reply with one "
                "short friendly sentence only; do not call any tools."
            ),
            "notice": "Okay, let's start over.",
        }

    if asking == "cuisine":
        cuisine = next((c for c in CUISINES if c.lower() == choice), None)
        if cuisine is None:
            return {"state": state, "gemini_message": None, "notice": None}
        return {
            "state": {"asking": "restaurant", "cuisine": cuisine},
            "gemini_message": f"Show me {cuisine} restaurants",
            "notice": None,
        }

    if asking == "restaurant":
        return {
            "state": {**state, "asking": "date", "restaurant": choice},
            "gemini_message": f"Tell me about {choice}",
            "notice": None,
        }

    if asking == "date":
        return {
            "state": {**state, "asking": "party", "date": choice},
            "gemini_message": (
                f"I'd like to book {state['restaurant']} on {choice}. Acknowledge "
                "this in one short sentence; do not call any tools yet."
            ),
            "notice": None,
        }

    if asking == "party":
        party = int(choice)
        result = check_reservation_availability.invoke(
            {
                "restaurant_name": state["restaurant"],
                "date": state["date"],
                "party_size": party,
            }
        )
        booked = "has availability" in result
        return {
            "state": dict(START),
            "gemini_message": (
                f"Check table availability at {state['restaurant']} "
                f"on {state['date']} for {party} people. Report the result in one "
                "or two sentences. If it is not available, say we will start over "
                "from the beginning; do not suggest other dates."
            ),
            "notice": (
                "Good news, that table is available! Want to look for another?"
                if booked
                else "Sorry, that table isn't available. Let's start over."
            ),
        }

    return {"state": dict(START), "gemini_message": None, "notice": None}
```

## jev_decision.py

```python
import asyncio
import logging
import os

from langsmith import traceable
from typesafe_sdk import AsyncTypeSafeClient, Choice

logger = logging.getLogger(__name__)


def _jev_client() -> AsyncTypeSafeClient:
    # Jev is also served through OpenRouter; use it when that key is set.
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return AsyncTypeSafeClient(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api",
            model=os.getenv("JEV_MODEL", "jev-latest"),
        )
    return AsyncTypeSafeClient()


@traceable(name="Jev Interpret Answer")
async def interpret_answer(
    question: dict, user_answer: str, history: list[dict] | None = None
) -> dict:
    """Ask Jev which choice of `question` the user's answer selects."""
    criteria = {c["key"]: c["description"] for c in question["choices"]}
    criteria["other"] = (
        "Anything else: a new question, or something that is not one of the choices"
    )
    conversation = "\n".join(
        f"{m['role']}: {m['content']}" for m in (history or [])
    )

    async with _jev_client() as client:
        response = await client.system_one(
            state={
                "conversation_so_far": conversation,
                "question_asked": question["text"],
                "user_answer": user_answer,
            },
            questions={
                "answer": Choice(
                    instructions=(
                        "The assistant asked the user the question in state. "
                        "Decide which choice the user's answer selects."
                    ),
                    criteria=criteria,
                )
            },
        )

    answer = response.answers["answer"]
    return {
        "choice": answer.choice,
        "confidence": getattr(answer, "confidence", None),
        "available": True,
    }


async def safe_interpret_answer(
    question: dict, user_answer: str, history: list[dict] | None = None
) -> dict:
    try:
        return await asyncio.wait_for(
            interpret_answer(question, user_answer, history),
            timeout=10,
        )
    except Exception:
        logger.exception("Jev interpretation failed")
        return {"choice": None, "confidence": None, "available": False}
```

## tools.py

```python
from langchain_core.tools import tool

# ============================================================
# TOOL 1: Search Restaurants by Cuisine
# ============================================================

@tool
def search_restaurants_by_cuisine(cuisine_type: str) -> list:
    """
    Search for restaurants by cuisine type.
    Supported cuisines: Indian, Italian, Chinese, Mexican.
    """

    restaurants = {
        "indian": [
            {"name": "Spice Palace", "rating": 4.5, "price": "₹₹"},
            {"name": "Curry House", "rating": 4.3, "price": "₹₹"},
            {"name": "Taj Flavors", "rating": 4.7, "price": "₹₹₹"}
        ],
        "italian": [
            {"name": "Pizza Bella", "rating": 4.4, "price": "₹₹"},
            {"name": "Pasta Dreams", "rating": 4.6, "price": "₹₹₹"},
            {"name": "Roma Kitchen", "rating": 4.2, "price": "₹₹"}
        ],
        "chinese": [
            {"name": "Dragon House", "rating": 4.5, "price": "₹₹"},
            {"name": "Ming's Kitchen", "rating": 4.3, "price": "₹₹"},
            {"name": "Golden Wok", "rating": 4.8, "price": "₹₹₹"}
        ],
        "mexican": [
            {"name": "Taco Fiesta", "rating": 4.4, "price": "₹₹"},
            {"name": "Salsa Nights", "rating": 4.6, "price": "₹₹₹"},
            {"name": "Burrito House", "rating": 4.2, "price": "₹₹"}
        ]
    }

    cuisine_lower = cuisine_type.lower().strip()

    return restaurants.get(cuisine_lower, [])


# ============================================================
# TOOL 2: Get Restaurant Details
# ============================================================

@tool
def get_restaurant_details(restaurant_name: str) -> dict:
    """
    Get detailed information about a restaurant,
    including address, phone, hours and specialties.
    """

    details = {
        "Spice Palace": {
            "cuisine": "Indian",
            "rating": 4.5,
            "address": "123 Food Street, City Center",
            "phone": "9876543210",
            "hours": "11 AM - 11 PM",
            "speciality": "Butter Chicken, Biryani"
        },

        "Pizza Bella": {
            "cuisine": "Italian",
            "rating": 4.4,
            "address": "456 Dine Avenue, Downtown",
            "phone": "9123456789",
            "hours": "12 PM - 10:30 PM",
            "speciality": "Margherita Pizza, Tiramisu"
        },

        "Dragon House": {
            "cuisine": "Chinese",
            "rating": 4.5,
            "address": "789 Noodle Lane, Market Area",
            "phone": "9111222333",
            "hours": "11:30 AM - 11 PM",
            "speciality": "Hakka Noodles, Fried Rice"
        },

        "Taco Fiesta": {
            "cuisine": "Mexican",
            "rating": 4.4,
            "address": "321 Spice Row, Hip District",
            "phone": "9222333444",
            "hours": "1 PM - 12 AM",
            "speciality": "Fish Tacos, Guacamole"
        },

        "Curry House": {
            "cuisine": "Indian",
            "rating": 4.3,
            "address": "555 Flavor Street, Old Town",
            "phone": "9333444555",
            "hours": "11 AM - 10 PM",
            "speciality": "Paneer Tikka, Dal Makhani"
        },

        "Taj Flavors": {
            "cuisine": "Indian",
            "rating": 4.7,
            "address": "999 Taste Lane, Premium Zone",
            "phone": "9444555666",
            "hours": "12 PM - 11 PM",
            "speciality": "Biryani, Tandoori Chicken"
        },

        "Golden Wok": {
            "cuisine": "Chinese",
            "rating": 4.8,
            "address": "777 Wok Street, Central Market",
            "phone": "9555666777",
            "hours": "11 AM - 11 PM",
            "speciality": "Szechuan Noodles, Dim Sum"
        },

        "Pasta Dreams": {
            "cuisine": "Italian",
            "rating": 4.6,
            "address": "12 Olive Road, Riverside",
            "phone": "9666777888",
            "hours": "12 PM - 11 PM",
            "speciality": "Truffle Pasta, Lasagna"
        },

        "Roma Kitchen": {
            "cuisine": "Italian",
            "rating": 4.2,
            "address": "34 Piazza Lane, Old Town",
            "phone": "9777888999",
            "hours": "11:30 AM - 10:30 PM",
            "speciality": "Risotto, Wood-fired Pizza"
        },

        "Ming's Kitchen": {
            "cuisine": "Chinese",
            "rating": 4.3,
            "address": "56 Lantern Street, Chinatown",
            "phone": "9888999000",
            "hours": "12 PM - 11 PM",
            "speciality": "Manchurian, Hot and Sour Soup"
        },

        "Salsa Nights": {
            "cuisine": "Mexican",
            "rating": 4.6,
            "address": "78 Fiesta Avenue, Hip District",
            "phone": "9999000111",
            "hours": "2 PM - 12:30 AM",
            "speciality": "Enchiladas, Churros"
        },

        "Burrito House": {
            "cuisine": "Mexican",
            "rating": 4.2,
            "address": "90 Cantina Road, Market Area",
            "phone": "9000111222",
            "hours": "11 AM - 10 PM",
            "speciality": "Loaded Burritos, Nachos"
        }
    }

    return details.get(
        restaurant_name.strip(),
        {"error": f"Restaurant '{restaurant_name}' not found"}
    )


# ============================================================
# TOOL 3: Check Reservation Availability
# ============================================================

@tool
def check_reservation_availability(
    restaurant_name: str,
    date: str,
    party_size: int
) -> str:
    """
    Check if a restaurant has reservation availability
    for a specific date and number of people.
    """

    available_slots = {
        "Spice Palace": {
            "2024-11-15": True,
            "2024-11-16": False,
            "2024-11-17": True
        },

        "Pizza Bella": {
            "2024-11-15": False,
            "2024-11-16": True,
            "2024-11-17": True
        },

        "Dragon House": {
            "2024-11-15": True,
            "2024-11-16": True,
            "2024-11-17": False
        },

        "Taco Fiesta": {
            "2024-11-15": True,
            "2024-11-16": True,
            "2024-11-17": True
        },

        "Curry House": {
            "2024-11-15": True,
            "2024-11-16": False,
            "2024-11-17": True
        },

        "Taj Flavors": {
            "2024-11-15": False,
            "2024-11-16": True,
            "2024-11-17": True
        },

        "Golden Wok": {
            "2024-11-15": True,
            "2024-11-16": False,
            "2024-11-17": True
        },

        "Pasta Dreams": {
            "2024-11-15": True,
            "2024-11-16": True,
            "2024-11-17": False
        },

        "Roma Kitchen": {
            "2024-11-15": False,
            "2024-11-16": True,
            "2024-11-17": True
        },

        "Ming's Kitchen": {
            "2024-11-15": True,
            "2024-11-16": True,
            "2024-11-17": True
        },

        "Salsa Nights": {
            "2024-11-15": False,
            "2024-11-16": False,
            "2024-11-17": True
        },

        "Burrito House": {
            "2024-11-15": True,
            "2024-11-16": True,
            "2024-11-17": False
        }
    }

    if restaurant_name not in available_slots:
        return f"Restaurant '{restaurant_name}' not found."

    is_available = available_slots[restaurant_name].get(
        date,
        False
    )

    if is_available:
        return (
            f"✓ {restaurant_name} has availability "
            f"for {party_size} people on {date}"
        )

    return (
        f"✗ {restaurant_name} is fully booked "
        f"for {date}"
    )


tools = [search_restaurants_by_cuisine, get_restaurant_details, check_reservation_availability]
```

## requirements.txt

```text
fastapi
uvicorn[standard]
langchain>=1.0
langchain-google-genai
pydantic
langsmith
python-dotenv
typesafe-sdk
```

## .gitignore

```text
.env
__pycache__/
*.pyc
.venv/
graphify-out/
```

