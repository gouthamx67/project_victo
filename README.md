# Regulatory Safety Reporting

Generate, review, and finalize safety reports from CSV data. Safety metrics are calculated deterministically; an LLM can optionally write the narrative. Reports stay drafts until human review is complete.

## How it works

![Architecture](architecture.png)

Raw CSV → ingest & validate → normalize (one case row per `safetyreportid`, one row per reaction) → deterministic analyses → evidence trace → per-section context → optional LLM narrative → draft report → human review → finalize.

The core rule is a hard boundary between computation and language:

- **Deterministic Python computes every number.** Counts, rates, distributions, trends, and case listings are produced by registered analysis functions. Results are reproducible and traceable back to source case/reaction IDs.
- **The LLM only phrases approved results.** Each section receives a minimal `SectionContext` holding only the metrics that section is allowed to use, inserted as metric slots. Generated text is validated against the approved values and falls back to a deterministic draft if it fails. The model never sees the raw dataset.

Report structure is driven by YAML configs; the engine itself has no report-specific code. Two report types ship:

- `configs/pader.yaml` — default config used by the web UI and API demo.
- `configs/psur_lite.yaml` — second report type, whose section set follows the exercise guide's expected PADER-style section list. **The committed example report was generated from this config.**

Adding a report type is a configuration change; if a needed metric does not exist yet, it is added once as a registered analysis and becomes available to every report type (see `version1/VERSION1.md`). This is the lens for the whole project: asked to add PSUR, PBRER, DSUR, or CSR support, the differences land as new YAML configs and any new registered analyses — not new code paths.

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

## Committed example report

`report_output.html` (repo root) is a generated, finalized example report for the sample dataset — open it in a browser. It was produced by a single script that runs the full pipeline, approves every section through the review gate, and writes the report:

```bash
python3 -m scripts.generate_example_report --provider groq   # or: --provider none
```

(The script expects the sample CSV at `data/Bisoprolol_icsr_sample_1068rows.csv`. The dataset is not redistributed with this repo.)

In the committed report all five LLM-authored sections are provider-generated with grounding status `passed`; the remaining sections are deterministic by config (`ai: false`). Per-section generation source and grounding status are recorded in the report's metadata section.

## Optional: enable LLM narrative
Get a key at https://console.groq.com/keys, then in the backend terminal, before starting uvicorn:

```bash
read -s -p "Groq API key: " GROQ_API_KEY; echo
export GROQ_API_KEY LLM_PROVIDER=groq GROQ_MODEL=openai/gpt-oss-120b
```
(Zsh: use `read -s "GROQ_API_KEY?Groq API key: "`.)

Or put the key in a local `.env` (already gitignored). The backend loads
`.env` automatically from the project root at startup:

```bash
printf 'GROQ_API_KEY=%s\n' 'gsk_your_key_here' > .env
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Never commit the key. Pasted keys are sanitized on read, so stray
whitespace, wrapping quotes, or terminal escape artifacts are removed.

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

#### What the model sees

The model never receives the CSV or any case records. Each section call contains exactly: the section's instructions and style, plus a JSON packet of only the metrics that section is allowed to use. Evidence IDs (`supporting_ids`) are stripped from the packet — they stay in a separate evidence trace for the review UI.

<details>
<summary>Prompt template — <code>prompts/sections/default.txt</code> (shown verbatim)</summary>

```
You are generating one section of a controlled regulatory safety report.

Your task is to write concise, factual narrative using ONLY the approved
evidence supplied below.

STRICT RULES:

1. Use only supplied metrics.
2. Reference metrics using their exact {metric_id} slot.
3. Never write metric numbers yourself.
4. Do not calculate new metrics.
5. Do not modify, reinterpret, round, or contradict supplied values.
6. Do not invent facts, cases, reactions, dates, or populations.
7. Do not infer causality.
8. Do not infer expectedness or labelled/unlabelled status.
9. Do not call anything a safety signal.
10. Do not introduce medical conclusions absent from evidence.
11. Preserve the distinction between case-level and reaction-level metrics.
12. If evidence is insufficient, state that the evidence is insufficient.
13. Do not expose supporting IDs unless specifically requested.
14. Use the exact metric_id inside braces when referencing a metric.
15. Do not use arithmetic or expressions inside metric slots.
16. Write NO digit characters (0-9) anywhere in your output, except
    inside a {metric_id} slot. Do not write years, month labels, dates,
    counts, or percentages as text. Refer to such evidence only through
    its metric slot.
17. For list or table metrics (for example monthly_case_counts,
    top_reactions, monthly_case_deviations, case_listing), reference the
    {metric_id} placeholder once and do NOT enumerate, list, or describe
    the individual items or their values.
```

(The template ends with the rendered placeholders `{{SECTION_TITLE}}`, `{{INSTRUCTIONS}}`, `{{STYLE_TONE}}`, `{{STYLE_LENGTH}}`, and `{{CONTEXT_JSON}}`.)

</details>

<details>
<summary>Example context packet (abridged) — the <code>serious_cases_15_day_alerts</code> section call</summary>

```json
{
  "section_id": "serious_cases_15_day_alerts",
  "section_title": "Serious and Expedited Cases",
  "instructions": "Summarize serious and expedited case counts, fatal outcomes, and case outcomes using supplied metrics only. State that expedited-reporting deadlines, timeliness, and submission compliance cannot be assessed without case-level deadline and submission evidence. Do not infer alert status or lateness.",
  "style": { "tone": "neutral", "length": "medium" },
  "metrics": [
    {
      "metric_id": "serious_case_count",
      "value": 1023,
      "definition": "Number of normalized safety cases whose case-level serious classification is True.",
      "population": "case",
      "supporting_ids": [],
      "version": "1.0.0"
    },
    {
      "metric_id": "case_outcome_distribution",
      "value": [
        { "case_outcome": "fatal", "count": 68 },
        { "case_outcome": "not_recovered", "count": 185 }
      ],
      "definition": "Distribution of one deterministic derived outcome classification per normalized safety case using the documented outcome precedence rule.",
      "population": "case",
      "supporting_ids": [],
      "version": "1.0.0"
    }
  ]
}
```

</details>

## Evaluation

**Grounding validator benchmark** — run `python3 -m eval.grounding_benchmark`. It scores the validator on 23 hand-authored cases: valid narratives that must be accepted, and invalid ones (unsupported numbers, digits outside slots, causal claims, missing required slots) that must be rejected. Last run:

```
Cases:                    23
Correct classifications:  23/23
Valid acceptance rate:    100.0%
Invalid rejection rate:   100.0%
Invalid acceptance rate:  0.0%
Accuracy:                 100.0%
```

**Mutation checks** — `eval/mutation_cases.py` (exercised by `tests/test_mutation_evaluation.py`) deliberately flips computational assumptions (for example, using row counts instead of unique case counts as a denominator) and asserts the pipeline output changes, guarding against analyses that would silently stop responding to their inputs. The full suite, `python3 -m pytest -q` (404 tests), covers the pipeline, API, validator, review gate, and configs.

**For a larger-scale system**, I'd extend this in stages: (1) keep generation-time grounding validation as the first gate and track violation codes, retry counts, and fallback rates per model and prompt version; (2) grow the benchmark with de-identified narratives and report per-rule precision/recall instead of pass/fail; (3) add semantic entailment scoring (an NLI-style model checking each statement against the section's approved metrics) to catch unsupported claims that avoid banned words; (4) sample human review decisions against the automated verdict and track disagreement; (5) red-team the validator adversarially — in practice we already hit definition-echo failures, where metric definition text visible to the model leaked banned phrasing into narratives, so the shipped definitions were reworded to survive echoing.

## Project documentation
- `DESIGN_DECISIONS.md` — the deterministic-vs-LLM boundary, normalization rules, and scope decisions.
- `KNOWN_LIMITATIONS.md` — prototype scope and known constraints.
- `eval/` — grounding benchmark and mutation checks (see Evaluation above).
- `scripts/generate_example_report.py` — regenerates the committed example report in one command.
- `version1/VERSION1.md` — the plan that proves report types are config-only; `configs/psur_lite.yaml` is its shipped config.
- `configs/pader.yaml`, `configs/psur_lite.yaml` — report-type definitions.
- `architecture.png` — system diagram.
- `report_output.html` — the committed example report (see above). Ad-hoc generated reports under `reports/` stay local and are gitignored.

## Packaging for submission

The repo is self-contained; the sample dataset is never redistributed. Keep secrets, data, and dependencies out of any archive:

- **Exclude:** `.env`, `data/` (and any CSV), `.venv/`, `frontend/node_modules/`, `__pycache__/`, caches and logs.
- **Include:** tracked source, tests, configs, prompts, `eval/`, `scripts/`, `version1/`, and `report_output.html`.

With the tree committed, `git archive` emits exactly the tracked files, so gitignored secrets and data cannot leak in:

```bash
git archive --format=zip -o submission.zip HEAD
```

Direct dependencies are pinned in `requirements.in`; a hash-locked candidate is provided in `requirements-lock-candidate.txt`.

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
- **A section shows a deterministic draft instead of LLM text:** the grounding check rejected the generated text after the allowed retries — this is the designed fallback, and the per-section generation source and grounding status are shown in the report metadata. Regenerating (or raising `--max-retries` in the CLI/script) usually resolves it, since validation is strict by design.
- **CSV upload fails:** the file must match the project's expected format.

## Safety note
Generated reports don't replace qualified review. Verify findings against source data and reporting requirements before finalizing.


