# GenAR AI Engineering Challenge
## Controlled AI System for Regulatory Safety Reporting

A prototype AI engineering system for controlled, evidence-backed regulatory
safety reporting.

The system demonstrates how deterministic safety-data analysis, constrained
LLM narrative generation, grounding validation, human review, reproducibility,
and evidence tracing can be combined without allowing an LLM to invent or
calculate regulatory metrics.

---

## Project status

**V0 PADER prototype:** implemented and tested.

**V1 PSUR-lite generalization:** implemented as a second configuration over
the same reporting engine.

Current regression suite:

**320 tests passing**

The supplied exercise dataset is intentionally excluded from Git and final
submission artifacts.

---

# 1. Problem

Regulatory safety reports contain quantitative and categorical claims that
must be reproducible and traceable to source safety cases.

A naive approach would send raw safety data to an LLM and ask it to:

- calculate statistics;
- identify trends;
- summarize cases;
- generate the report.

That design creates unnecessary risks:

- arithmetic errors;
- inconsistent aggregation;
- unsupported claims;
- hallucinated numbers;
- unclear evidence provenance;
- weak reproducibility.

This project instead separates deterministic computation from controlled
language generation.

---

# 2. Core design principle

> **Python computes what is true from the supplied data. The LLM may only
> interpret and phrase approved deterministic results.**

The LLM does not receive the raw CSV.

It receives a minimal, section-specific context containing approved
`AnalysisResult` objects.

Metric values are represented in generated narratives using exact metric
slots such as:

```text
{case_count}
{serious_case_percentage}
{fatal_case_count}