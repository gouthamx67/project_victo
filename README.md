# Regulatory Safety Reporting

Generate, review, and finalize safety reports from CSV data. Safety metrics are calculated deterministically; an LLM can optionally write the narrative. Reports stay drafts until human review is complete.

## How it works

![Architecture](architecture.png)

Raw CSV → ingest & validate → normalize (one case row per `safetyreportid`, one row per reaction) → deterministic analyses → evidence trace → per-section context → optional LLM narrative → draft report → human review → finalize.

The core rule is a hard boundary between computation and language:

- **Deterministic Python computes every number.** Counts, rates, distributions, trends, and case listings are produced by registered analysis functions. Results are reproducible and traceable back to source case/reaction IDs.
- **The LLM only phrases approved results.** Each section receives a minimal `SectionContext` holding only the metrics that section is allowed to use, inserted as metric slots. Generated text is validated against the approved values and falls back to a deterministic draft if it fails. The model never sees the raw dataset.

Report structure is driven by YAML configs: `configs/pader.yaml` (default) and `configs/psur_lite.yaml` (a second report type, used to show the pipeline generalizes).

## Prerequisites
Git, Python 3.11+, Node.js + npm.

## Quick start

```bash
git clone https://github.com/gouthamx67/project_victo.git
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

Or put the key in a local `.env` (already gitignored) and load it:

```bash
printf 'GROQ_API_KEY=%s\n' 'gsk_your_key_here' > .env
set -a; source .env; set +a   # before starting uvicorn
```

Never commit the key.

The backend selects the provider from `LLM_PROVIDER` (default `groq`):

| `LLM_PROVIDER` | SDK path | Default model | Key variable |
| --- | --- | --- | --- |
| `groq` (default) | Chat Completions (`/chat/completions`) | `openai/gpt-oss-120b` | `GROQ_API_KEY` |
| `openai` | Responses API (`/responses`) | `gpt-5-mini` | `OPENAI_API_KEY` |

Override the model with `GROQ_MODEL` / `OPENAI_MODEL`. With LLM off (the default), no key is needed and narratives are deterministic drafts.

### Generate from the CLI

```bash
python -m src.report_cli \
  --csv "path/to/input.csv" \
  --config configs/pader.yaml \
  --prompt prompts/sections/default.txt \
  --template prompts/report_template.html \
  --output reports/pader-groq.html \
  --provider groq --model openai/gpt-oss-120b
```

Use `--provider none` (default) for a deterministic report, or `--provider mock` for an offline test double.

### Prompts and grounding
Narrative instructions live in `prompts/sections/default.txt`; the report scaffold is `prompts/report_template.html`. Every generated section is checked against the approved metrics it was given; rejected text is retried and then replaced by a deterministic draft (see `src/grounding_validator.py` and `src/metric_slots.py`).

## Project documentation
- `DESIGN_DECISIONS.md` — the deterministic-vs-LLM boundary, normalization rules, and scope decisions.
- `KNOWN_LIMITATIONS.md` — prototype scope and known constraints.
- `configs/pader.yaml`, `configs/psur_lite.yaml` — report-type definitions (the second proves generalization).
- `architecture.png` — system diagram. Generated reports are local artifacts and should not be committed.

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


