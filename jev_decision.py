import asyncio
import logging

from langsmith import traceable
from typesafe_sdk import AsyncTypeSafeClient, Choice

logger = logging.getLogger(__name__)


@traceable(name="Jev Restaurant Classification")
async def classify_restaurant_request(message: str, history: list[dict] | None = None) -> dict:
    conversation = "\n".join(
        f"{m['role']}: {m['content']}" for m in (history or [])
    )
    async with AsyncTypeSafeClient() as client:
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