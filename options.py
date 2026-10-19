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
