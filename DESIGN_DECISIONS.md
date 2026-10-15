# Design Decisions

This document records architectural decisions made during the project.

## DD-001 — Deterministic analysis before LLM generation

**Decision:** Numerical and categorical analyses will be computed by deterministic Python code rather than by the LLM.

**Reason:** Safety-report numbers must be reproducible and traceable to source cases. The LLM is better suited to controlled language generation than arithmetic or aggregation.

**Status:** Accepted

---

## DD-002 — Dataset is local-only

**Decision:** The supplied safety dataset will live under `data/` locally and will be excluded from Git and the final submission package.

**Reason:** The exercise explicitly prohibits redistribution/publishing of the supplied data.

**Status:** Accepted

---

## DD-003 — Checkpoint-driven development

**Decision:** The system will be implemented and verified incrementally.

**Reason:** Each component should be understandable and testable before the next component depends on it.

**Status:** Accepted



## CP2 — Raw ingest boundary

### Decision: Read raw CSV values conservatively

The ingest layer reads CSV columns as strings first.

Reason:
- avoids accidental pandas type coercion;
- preserves source representation;
- prevents domain semantics from being introduced at the file boundary;
- allows later normalization to be explicit and tested.

### Decision: Parse receivedate explicitly as YYYYMMDD

The profiled dataset stores receivedate values such as `20241227`.

Generic pandas datetime conversion interpreted these integer values as
nanoseconds after the Unix epoch, producing incorrect 1970 timestamps.

Therefore the ingest validator requires an eight-digit YYYYMMDD shape and
uses explicit `%Y%m%d` parsing.

### Decision: Case identity is safetyreportid

Input CSVs may contain multiple rows per `safetyreportid`, so row-level counts must be distinguished from case-level counts. Public examples use synthetic data; restricted evaluation CSVs and their record counts should not be documented.

Therefore:
- row-level analysis must not be confused with case-level analysis;
- ingest preserves all rows;
- later normalization/analysis will explicitly select the appropriate unit.

### Decision: Validation does not silently normalize categorical values

Several categorical fields require inspection of their actual vocabulary.

The ingest layer records observed values but does not translate them into
canonical values.

Normalization belongs in a later checkpoint where each mapping can be
documented and tested.

### Decision: Missing domain values are not automatically fatal

Fields such as age, age unit, sex, and country contain missing values in the
actual dataset.

The ingest layer reports missingness but does not invent values or reject
otherwise structurally valid cases.

Field-specific handling belongs in normalization and analysis.


## CP3 — Normalization

### Decision: Maintain separate case and reaction tables

The source contains 1,068 rows but 1,024 safetyreportid values.

The normalized representation therefore has:
- one case record per safetyreportid;
- reaction records derived from the reaction field.

This prevents row-level counts from being incorrectly presented as
case-level metrics.

### Decision: Preserve ambiguity instead of selecting arbitrary values

When a case contains multiple distinct values for a field such as country or
age, normalization does not choose the first row.

The normalized record is marked ambiguous and the value is left missing.

This avoids introducing unsupported facts.

### Decision: Country analyses use occurcountry

The source supplies both `occurcountry` and `primarysource_reportercountry`.
The reported country distribution uses `occurcountry`; the reporter country
is normalized separately and cases where the two disagree are flagged
(`country_mismatch`) instead of being silently resolved.

Reporter qualification (`primarysource_qualification`) is normalized to a
single case-level value using the same single-value-or-missing rule, and
feeds the reporter qualification distribution metric.

### Decision: Aggregate conflicting seriousness flags using any-yes

The seriousness criteria are independent flags.

For a case:
- any observed yes -> normalized flag True;
- otherwise, observed no -> False;
- no usable value -> missing.

If both yes and no occur across rows of the same case, a conflict indicator is
also retained.

This makes the case-level result conservative while keeping the source
inconsistency visible.

### Decision: Normalize age into years only when the age/unit pair is unambiguous

The source contains:
- 800
- year
- month
- week
- day

ICH E2B(R2) defines 800 as Decade.

The original age and unit are retained. A comparable age_years field is
derived only when exactly one valid age/unit pair exists for the case.

### Decision: Do not invent reaction outcomes

Reaction terms are split from the comma-separated source field.

Outcomes are aligned positionally where available.

If a reaction has no corresponding source outcome, the outcome remains missing
and an explicit alignment status is stored.

### Decision: Do not collapse reaction outcomes to a case-level outcome yet

Outcome precedence has not been finalized.

The normalized reaction table therefore retains individual outcomes. A
case-level outcome rule will be introduced only after it is explicitly
documented and tested.