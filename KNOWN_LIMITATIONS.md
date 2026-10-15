# Known Limitations

This document records known limitations of the prototype.

## KL-001 — Prototype scope

This is a V0 prototype for the hiring evaluation. It is not a production pharmacovigilance system.

## KL-002 — Dataset scope

The system will operate on the supplied exercise dataset. Generalization to other safety-data sources will be assessed separately.

## KL-003 — Pharmacovigilance expertise

The implementation will explicitly distinguish computational facts from regulatory/medical interpretation. Unsupported medical or regulatory conclusions will not be inferred from the dataset.

## KL-004 — Human review

Generated content is not considered final solely because it passes automated validation. Human review remains part of the workflow.

## Grounding validation: scope and limitations

- **Small regression benchmark:** `eval/grounding_benchmark.py` contains
  a small set of hand-authored cases for known behaviors. Passing all cases
  shows that these regressions are covered; it does not estimate accuracy
  on arbitrary LLM outputs or unseen narratives.
- **Heuristic causal-language checks:** causal claims are identified with
  regular expressions, not a semantic model. Neutral exceptions currently
  cover specific phrasings such as “discusses whether” and “did not establish
  that.” Other neutral wording may still be rejected, and future exceptions
  could accidentally allow unsupported claims unless tested carefully.
- **Numeric text must use metric slots:** the validator rejects ASCII and
  Unicode decimal digits outside brace-delimited metric slots. It also
  rejects recognized spelled-out quantities when followed by common count
  nouns. These are heuristic checks, not a complete natural-language number
  detector. Metric slots are validated against approved metrics and resolved
  deterministically after narrative validation; this does not prove that all
  surrounding prose is semantically correct.
- **Metric-slot checks are limited:** requiring a slot establishes that the
  designated evidence reference appears in the narrative. It does not prove
  every surrounding statement is correct.
- **Upload limits need deployment-level enforcement:** the API reads at
  most one byte beyond its configured file-size limit, but multipart parsing
  occurs before the endpoint handler. Deployments should also enforce an
  overall request-body limit at the server or reverse-proxy ingress to
  reduce the risk of oversized requests consuming excess memory or disk.
- **Human review remains necessary:** passing the validator and benchmark is
  not clinical, regulatory, or safety validation. Generated reporting
  narratives still require appropriate evidence review and human approval.
- **Data confidentiality:** public examples and benchmarks should use
  synthetic data. Do not add restricted evaluation records or identifiable
  safety-report content to public tests, benchmark fixtures, or documentation.
