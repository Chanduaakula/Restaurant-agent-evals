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
