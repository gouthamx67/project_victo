# Version 1 — Plan for a Second Report Type

## Goal
Extend the V0 system (a PADER-style report) to a **second regulatory report type**
without changing the analysis engine, so that adding a report type is a
configuration task rather than a code change.

## What V1 is
`PSUR-lite` (Periodic Safety Update Report) — a second section structure over the
**same** normalized dataset and the **same** deterministic analyses. It is defined
entirely in `configs/psur_lite.yaml` and can be run through the existing pipeline.

## Reused unchanged (the point of V1)
ingest + validation → normalization (cases/reactions) → registered analyses →
evidence trace → per-section `SectionContext` → grounded narrative (optional LLM) →
draft → human review → finalize → API + frontend. No engine code forks.

## What changes (configuration only)
- Report identity: `id`, `title`, `report_type`, regulatory basis.
- Section set, order, and per-section `needs` (which metrics each section may use).
- Per-section `ai: true/false` (deterministic vs LLM-authored).
- Section instructions and style (tone/length).

## Why this proves reuse
The engine already separates report *structure* (YAML) from *computation*
(registered analyses). V1 adds only a config file; if a needed metric does not
exist, it is added once as a registered analysis and becomes available to every
report type.

## Roadmap beyond the demo
- Add PSUR-specific analyses as new registered functions (e.g. cumulative
  exposure, expectedness) — no changes to existing reports.
- Finalize the case-level outcome precedence rule (currently a documented
  heuristic) and promote it into normalization.
- Broaden ingestion beyond the exercise dataset (see `KNOWN_LIMITATIONS.md`).

## Validation
- `tests/test_psur_lite_config.py` runs PSUR-lite through the full pipeline.
- Generated report HTML files are local artifacts and should not be committed.

## Open decisions
- Which PSUR sections are mandatory vs optional per jurisdiction.
- Whether outcome precedence belongs in normalization or analysis.
- How much report-specific vocabulary belongs in prompts vs config.
