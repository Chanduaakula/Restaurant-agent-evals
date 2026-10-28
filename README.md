# Restaurant Agent Evals

A restaurant booking assistant with two agents answering side by side, plus the code to evaluate them.

- **Jev** runs a guided booking flow. It reads each user answer, picks the matching option, and the backend runs the right tool.
- **Google Gemini** is a LangChain agent (`create_agent`) that gets the same input and decides on its own which tools to call.

The frontend shows both answers next to each other, so you can compare how each agent handles the same conversation.

## Booking flow

```
cuisine -> restaurant -> date -> party size -> time slot -> availability
```

- **Dates:** the next 7 days, starting tomorrow. Each restaurant is closed one weekday, and that day isn't offered.
- **Time slots:** each restaurant has 4 slots per day, each with a number of free seats. On Saturdays and Sundays, dinner slots (19:00 and later) have half the seats.
- **Restarting:** if no slot fits the party, or the user says "none of these", the flow starts again from the cuisine question.

The sample data (12 restaurants across Indian, Italian, Chinese and Mexican) is in `backend/tools.py`.

## Project structure

```
backend/                 FastAPI server
  main.py                POST /chat, GET /health, the Gemini LangChain agent
  flow.py                The guided booking questions (Jev)
  jev_decision.py        Jev: maps the user's answer to one of the choices
  tools.py               Tools: search by cuisine, restaurant details, availability
  evaluation.py          Standalone LangChain evaluation of the Gemini agent
  evals/
    cases.py             27 test cases (flow, Jev, Gemini)
    run_evals.py         End-to-end evaluation through POST /chat
    api.py               /eval/... HTTP endpoints
    langchain_eval.py    LangSmith evaluation of the Gemini agent
  CODE.md                Full source listing
frontend/                Vite + TypeScript chat UI
```

## Setup

You need Python 3.12 and Node.js 18 or later.

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then fill in your keys
.\.venv\Scripts\python -m uvicorn main:app --reload --port 8000
```

On macOS or Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python`.

Environment variables (`backend/.env`):

| Variable | Needed for |
|---|---|
| `GOOGLE_API_KEY` | Gemini |
| `OPENROUTER_API_KEY` | Jev (via OpenRouter). Without it, answers are matched by exact text only. |
| `MODEL` | Optional. Defaults to `google_genai:gemini-3.5-flash-lite`. |
| `CORS_ORIGINS` | Optional. Comma-separated; defaults to `*`. |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Optional. LangSmith tracing and evaluation. |

### Frontend

```powershell
cd frontend
npm install
copy .env.example .env      # VITE_API_URL=http://localhost:8000
npm run dev
```

Open http://localhost:5173.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send `{ message, history, gemini_history, state }`. Returns `{ google, jev, state }`. Send `state` back unchanged with the next request. |
| `GET` | `/health` | `{ "status": "ok" }` |
| `GET` | `/eval/cases` | List the evaluation cases. |
| `POST` | `/eval/run` | Start an evaluation run in the background. Body (all optional): `{ only, category, judge, concurrency }`. |
| `GET` | `/eval/runs/{run_id}` | A run's status, progress, summary and results. |

Interactive docs: http://localhost:8000/docs

## Evaluation

There are three ways to evaluate the agents. Run all commands from `backend/`.

### 1. `evaluation.py`: the Gemini agent, standalone

Runs the LangChain agent on 10 questions and scores each answer:

- **Tool calls:** the right tool, with the right arguments (restaurant, date, party size, time).
- **Forbidden tools:** e.g. it must not check availability before it knows the party size.
- **Mentions:** required facts in the reply, such as a phone number or restaurant names.
- **LLM judge:** a LangChain chat model with structured output grades the reply against a written criterion.

```powershell
.\.venv\Scripts\python evaluation.py
.\.venv\Scripts\python evaluation.py --no-judge
```

### 2. `evals/run_evals.py`: the whole `/chat` endpoint

27 cases covering:

- **flow:** full multi-turn bookings, a party too big for any slot, restarting, weekend seat limits.
- **jev:** understanding typed answers, e.g. "in the mood for pasta" → Italian, "me and my wife" → 2, "dinner at 7" → 19:00, and saying "other" for anything that isn't a choice.
- **gemini:** grounded answers and correct tool use.

Jev and Gemini are scored separately. Results are printed and saved to `evals/results/`.

```powershell
.\.venv\Scripts\python -m evals.run_evals
.\.venv\Scripts\python -m evals.run_evals --category jev
.\.venv\Scripts\python -m evals.run_evals --only F1,C3 --no-judge
```

### 3. `evals/langchain_eval.py`: LangSmith experiment

Runs the Gemini cases with `langsmith.aevaluate`. When `LANGSMITH_API_KEY` is allowed to use datasets, results are saved as a LangSmith experiment; otherwise the run happens in memory.

```powershell
.\.venv\Scripts\python -m evals.langchain_eval
```

### Notes

- The Gemini free tier allows 15 requests a minute. The evaluation scripts wait and retry when they hit that limit, so a full run takes a few minutes.
- Test dates are worked out from today's date, so the cases stay valid as the booking window moves.
