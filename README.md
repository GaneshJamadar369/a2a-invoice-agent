# A2A Invoice Agent — TDS GA3

FastAPI implementation of the supplied A2A 1.0 invoice-agent contract. It proposes exactly one action per invoice package using a batched AI request, stores Task state in SQLite, caches validated package decisions by canonical package content, and records receipts only for matching `ACCEPTED` results from the grader.

## Local setup (PowerShell)

```powershell
cd C:\Users\HP\Downloads\TDS Assn IITM\a2a-invoice-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `GROQ_API_KEY` in `.env`. For local launch, PowerShell does not automatically load `.env`, so set it in the current shell:

```powershell
$env:GROQ_API_KEY = "YOUR_GROQ_API_KEY"
$env:A2A_BASE_URL = "http://localhost:8000/a2a/"
$env:DATABASE_PATH = "./data/a2a-invoice-agent.sqlite3"
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

In a second PowerShell window, check the public Agent Card locally:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/.well-known/agent-card.json
```

Use `pytest -q` to run the local protocol tests.

## Deploy on Render

1. Push this repository to GitHub and create a Render **Web Service** from it.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Configure `GROQ_API_KEY` as a Render secret/environment variable.
5. After Render gives you the HTTPS origin, set `A2A_BASE_URL` to the **exact base URL that will be submitted**, with trailing slash; e.g. `https://YOUR-SERVICE.onrender.com/a2a/`.
6. Redeploy and verify `https://YOUR-SERVICE.onrender.com/.well-known/agent-card.json` contains that exact `supportedInterfaces[0].url`.
7. Submit `https://YOUR-SERVICE.onrender.com/a2a/` as the A2A interface base URL.

SQLite keeps state across HTTP requests while the instance filesystem persists. A service restart/redeploy can erase an ephemeral filesystem; if persistent storage is available, set `DATABASE_PATH` to its mounted disk path. Keep the database file private.

## Contract behaviors

- Public `GET /.well-known/agent-card.json`.
- Authenticated A2A routes under `/a2a/`: `POST /message:send`, `GET /tasks/{id}`, `GET /tasks`, and `POST /tasks/{id}:cancel`.
- Requires `A2A-Version: 1.0`; POST routes require `Content-Type: application/a2a+json`.
- Each Bearer token is an isolated principal; raw tokens are not stored.
- Idempotency key is `(principal, messageId)`, hashed from recursively key-sorted compact JSON of `message` only. `configuration` is ignored.
- The proposal task stays `TASK_STATE_INPUT_REQUIRED`. Only a result continuation matching task, context, batch, package, action ID and action can complete it. `ACCEPTED` outcomes create receipts; `REJECTED` outcomes do not.
- Cancellation and result continuation serialize through SQLite `BEGIN IMMEDIATE` transactions.
- Package decisions are cached by normalized package content; action IDs are fresh per new task.

No payment API is called by this agent. Its “execution” records are acceptance receipts from the grader, not actual financial transfers.
