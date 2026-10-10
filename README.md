# Regulatory Safety Reporting

Generate, review, and finalize safety reports from CSV data. Safety metrics are calculated deterministically; an LLM can optionally write the narrative. Reports stay drafts until human review is complete.

## Prerequisites
Git, Python 3.11+, Node.js + npm.

## Quick start

```bash
git clone <YOUR-REPOSITORY-URL>
cd project_victo
```

**Terminal 1: backend**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```
Check it works: http://127.0.0.1:8000/health

**Terminal 2: frontend**
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## Generate a report
1. Upload your CSV.
2. Leave **Enable LLM narrative** off for your first run (no API key needed).
3. Click **Generate report**.
4. Review the evidence and complete human review before finalizing.

## Optional: enable LLM narrative
Get a key at https://console.groq.com/keys, then in the backend terminal, before starting uvicorn:

```bash
read -s -p "Groq API key: " GROQ_API_KEY; echo
export GROQ_API_KEY LLM_PROVIDER=groq GROQ_MODEL=openai/gpt-oss-120b
```
(Zsh: use `read -s "GROQ_API_KEY?Groq API key: "`.)

Never commit the key.

## API (optional)
```bash
curl -sS -X POST http://127.0.0.1:8000/runs \
  -F 'file=@path/to/your.csv' \
  -F 'use_llm=false'   # or true
```

## Troubleshooting
- **Backend won't start:** activate `.venv`, rerun `pip install -r requirements.txt` from the project root.
- **Frontend won't start:** run `npm install` inside `frontend/`.
- **Port in use:** stop the other process or change the port.
- **Invalid API key:** set `GROQ_API_KEY` and `LLM_PROVIDER=groq` in the same terminal that starts the backend.
- **LLM fails:** check the backend terminal for the error, plus model access and quota.
- **CSV upload fails:** the file must match the project's expected format.

## Safety note
Generated reports don't replace qualified review. Verify findings against source data and reporting requirements before finalizing.


