# Restaurant Assistant Backend

FastAPI backend. `POST /chat` returns three labeled parts: `google` (Gemini agent answer and tool calls), `jev` (Jev intent classification over the whole conversation) and `options` (clickable choices the frontend shows so the user doesn't have to type).

Endpoints: `POST /chat` and `GET /health`. Run: `uvicorn main:app --reload`.

Environment variables (set in `.env` locally or in the host's settings; never commit them): `GOOGLE_API_KEY`, `OPENROUTER_API_KEY` (Jev via OpenRouter; or `TYPESAFE_API_KEY`), `CORS_ORIGINS`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`.

## main.py

```python

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
from options import build_options

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


class ChatResponse(BaseModel):
    google: dict   # Gemini's answer: source, reply, tool_calls
    jev: dict      # Jev's classification: source, intent, confidence, available
    options: dict  # clickable choices: prompt + choices[{label, message}]


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
    jev_task = safe_classify_restaurant_request(
        req.message,
        [m.model_dump() for m in req.history],
    )

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

    return ChatResponse(
        google={"source": "Google Gemini", "reply": reply, "tool_calls": tool_calls},
        jev={"source": "Jev", **jev_result},
        options=build_options(tool_calls),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
```

## options.py

```python
import json

CUISINES = ["Indian", "Italian", "Chinese", "Mexican"]

# Dates that exist in the sample availability data in tools.py.
BOOKING_DATES = [
    ("15 Nov 2024", "2024-11-15"),
    ("16 Nov 2024", "2024-11-16"),
    ("17 Nov 2024", "2024-11-17"),
]

# Shown last in every list so the user can always type their own message instead.
TYPE_MY_OWN = {"label": "None of these - I'll type my own", "message": None}


def _choice(label: str, message: str) -> dict:
    return {"label": label, "message": message}


def _cuisine_options(prompt: str = "What cuisine would you like?") -> dict:
    return {
        "prompt": prompt,
        "choices": [_choice(c, f"Show me {c} restaurants") for c in CUISINES]
        + [TYPE_MY_OWN],
    }


def _parse(output: str):
    try:
        return json.loads(output)
    except (TypeError, ValueError):
        return None


def build_options(tool_calls: list[dict]) -> dict:
    """Pick the next clickable choices from the last tool used this turn.

    Each choice has a `label` to display and a `message` the frontend sends to
    /chat when the user clicks it (`message: null` means let the user type).
    """
    if not tool_calls:
        return _cuisine_options()

    last = tool_calls[-1]
    name = last["name"]
    args = last.get("args") or {}
    data = _parse(last["output"])

    if name == "search_restaurants_by_cuisine":
        # The model may search several cuisines in one turn; offer all of them.
        found = []
        for call in tool_calls:
            if call["name"] != "search_restaurants_by_cuisine":
                continue
            results = _parse(call["output"])
            if isinstance(results, list):
                found += [r["name"] for r in results if isinstance(r, dict) and "name" in r]
        if not found:
            return _cuisine_options("I couldn't find that cuisine. Pick one:")
        return {
            "prompt": "Which restaurant would you like to know more about?",
            "choices": [_choice(n, f"Tell me about {n}") for n in found]
            + [_choice("Choose a different cuisine", "I want a different cuisine")]
            + [TYPE_MY_OWN],
        }

    restaurant = args.get("restaurant_name")

    if name == "get_restaurant_details":
        if isinstance(data, dict) and "error" in data or not restaurant:
            return _cuisine_options("I couldn't find that restaurant. Pick a cuisine:")
        return {
            "prompt": f"Book a table at {restaurant}? Pick a date (for 2 people):",
            "choices": [
                _choice(
                    label,
                    f"Check table availability at {restaurant} on {date} for 2 people",
                )
                for label, date in BOOKING_DATES
            ]
            + [
                _choice("Choose a different cuisine", "I want a different cuisine"),
                TYPE_MY_OWN,
            ],
        }

    if name == "check_reservation_availability" and restaurant:
        return {
            "prompt": "Anything else?",
            "choices": [
                _choice(f"More about {restaurant}", f"Tell me about {restaurant}"),
                _choice("Choose a different cuisine", "I want a different cuisine"),
                TYPE_MY_OWN,
            ],
        }

    return _cuisine_options()
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


@traceable(name="Jev Restaurant Classification")
async def classify_restaurant_request(message: str, history: list[dict] | None = None) -> dict:
    conversation = "\n".join(
        f"{m['role']}: {m['content']}" for m in (history or [])
    )
    async with _jev_client() as client:
        response = await client.system_one(
            state={"conversation_so_far": conversation, "user_request": message},
            questions={
                "intent": Choice(
                    instructions=(
                        "Classify what the user wants in their latest request, "
                        "using the earlier conversation for context."
                    ),
                    criteria={
                        "search": "Search for restaurants",
                        "details": "Get restaurant details",
                        "reservation": "Check reservation availability",
                        "general": "Other questions",
                    },
                )
            },
        )

    answer = response.answers["intent"]

    return {
        "intent": answer.choice,
        "confidence": getattr(answer, "confidence", None),
        "available": True,
    }


async def safe_classify_restaurant_request(message: str, history: list[dict] | None = None) -> dict:
    try:
        return await asyncio.wait_for(
            classify_restaurant_request(message, history),
            timeout=10,
        )
    except Exception:
        logger.exception("Jev classification failed")
        return {
            "intent": "unknown",
            "confidence": None,
            "available": False,
        }
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

