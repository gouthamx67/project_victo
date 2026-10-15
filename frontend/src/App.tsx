import { useCallback, useMemo, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

type RunDetails = {
  run_id: string;
  status: "DRAFT" | "FINAL";
  report_id: string;
  report_version: string;
  created_at_utc?: string;
  dataset_sha256?: string;
  analysis_registry_version?: string;
  artifact_files?: string[];
};

type SectionReview = {
  section_id: string;
  status: "UNREVIEWED" | "APPROVED" | "FLAGGED";
  reviewer: string | null;
  comment: string;
  reviewed_at: string | null;
};

type ReviewPayload = {
  reviews: SectionReview[];
};

type EvidenceSection = {
  section_id: string;
  metric_ids: string[];
  supporting_ids: Record<string, string[]>;
};

type NarrativeArtifact = {
  narrative?: unknown;
  narrative_template?: unknown;
  generation_source?: unknown;
};

type NarrativesPayload = Record<string, NarrativeArtifact>;

type RunArtifactState = {
  details: RunDetails;
  review: ReviewPayload;
  evidence: EvidenceSection[];
  narratives: NarrativesPayload;
};

function extractNarrative(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  if (
    typeof value === "object" &&
    value !== null &&
    "text" in value &&
    typeof value.text === "string"
  ) {
    return value.text;
  }

  return "";
}

function statusLabel(status: string): string {
  return status.replaceAll("_", " ");
}

function narrativeSourceLabel(source: string): string {
  if (source === "provider") {
    return "AI-GENERATED NARRATIVE";
  }

  if (source === "deterministic_template") {
    return "DETERMINISTIC NARRATIVE";
  }

  return "NARRATIVE";
}

function formatDate(value?: string): string {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function truncateHash(value?: string): string {
  if (!value) return "—";
  if (value.length <= 20) return value;
  return `${value.slice(0, 12)}…${value.slice(-8)}`;
}

async function requestJson<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);

  const text = await response.text();

  let payload: unknown = null;

  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

  if (!response.ok) {
    const detail =
      typeof payload === "object" &&
      payload !== null &&
      "detail" in payload
        ? String(payload.detail)
        : `Request failed with HTTP ${response.status}`;

    throw new Error(detail);
  }

  return payload as T;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [useLlm, setUseLlm] = useState(false);

  const [run, setRun] = useState<RunArtifactState | null>(null);

  const [reviewer, setReviewer] = useState("");
  const [comments, setComments] = useState<Record<string, string>>({});

  const [loading, setLoading] = useState(false);
  const [reviewLoading, setReviewLoading] = useState<string | null>(null);
  const [finalizing, setFinalizing] = useState(false);

  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const [expandedEvidence, setExpandedEvidence] = useState<
    Record<string, boolean>
  >({});

  const [expandedMetrics, setExpandedMetrics] = useState<
    Record<string, boolean>
  >({});

  const loadRunArtifacts = useCallback(async (runId: string) => {
    const [details, review, evidence, narratives] = await Promise.all([
      requestJson<RunDetails>(`/runs/${runId}`),
      requestJson<ReviewPayload>(`/runs/${runId}/review`),
      requestJson<EvidenceSection[]>(`/runs/${runId}/evidence`),
      requestJson<NarrativesPayload>(`/runs/${runId}/narratives`),
    ]);

    setRun({
      details,
      review,
      evidence,
      narratives,
    });
  }, []);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setFile(event.target.files?.[0] ?? null);
    setError("");
    setSuccess("");
  };

  const handleRun = async (event: FormEvent) => {
    event.preventDefault();

    if (!file) {
      setError("Select a CSV file first.");
      return;
    }

    setLoading(true);
    setError("");
    setSuccess("");
    setRun(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("use_llm", String(useLlm));

      const created = await requestJson<{ run_id: string }>(
        "/runs",
        {
          method: "POST",
          body: formData,
        },
      );

      await loadRunArtifacts(created.run_id);

      setSuccess(
        useLlm
          ? "Report generated with the configured LLM provider."
          : "Deterministic draft generated successfully.",
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to generate the report.",
      );
    } finally {
      setLoading(false);
    }
  };

  const updateReview = async (
    sectionId: string,
    status: "APPROVED" | "FLAGGED",
  ) => {
    if (!run) return;

    if (!reviewer.trim()) {
      setError("Enter reviewer name before approving or flagging a section.");
      return;
    }

    setReviewLoading(sectionId);
    setError("");
    setSuccess("");

    try {
      await requestJson(`/runs/${run.details.run_id}/review`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          section_id: sectionId,
          status,
          reviewer: reviewer.trim(),
          comment: comments[sectionId] ?? "",
        }),
      });

      await loadRunArtifacts(run.details.run_id);

      setSuccess(
        `${sectionId} marked ${status.toLowerCase()}.`,
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to update review.",
      );
    } finally {
      setReviewLoading(null);
    }
  };

  const finalizeReport = async () => {
    if (!run) return;

    setFinalizing(true);
    setError("");
    setSuccess("");

    try {
      await requestJson(
        `/runs/${run.details.run_id}/finalize`,
        {
          method: "POST",
        },
      );

      await loadRunArtifacts(run.details.run_id);

      setSuccess("Report finalized. All configured sections are approved.");
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to finalize report.",
      );
    } finally {
      setFinalizing(false);
    }
  };

  const approvedCount = useMemo(
    () =>
      run?.review.reviews.filter(
        (review) => review.status === "APPROVED",
      ).length ?? 0,
    [run],
  );

  const flaggedCount = useMemo(
    () =>
      run?.review.reviews.filter(
        (review) => review.status === "FLAGGED",
      ).length ?? 0,
    [run],
  );

  const unreviewedCount = useMemo(
    () =>
      run?.review.reviews.filter(
        (review) => review.status === "UNREVIEWED",
      ).length ?? 0,
    [run],
  );

  const allApproved =
    Boolean(run) &&
    run!.review.reviews.length > 0 &&
    run!.review.reviews.every(
      (review) => review.status === "APPROVED",
    );

  const evidenceBySection = useMemo(() => {
    const map = new Map<string, EvidenceSection>();

    for (const section of run?.evidence ?? []) {
      map.set(section.section_id, section);
    }

    return map;
  }, [run]);

  const reviewSections = useMemo(() => {
    if (!run) return [];

    return run.review.reviews.map((review) => {
      const narrativeArtifact = run.narratives[review.section_id];

      return {
        review,
        narrative: extractNarrative(
          narrativeArtifact?.narrative,
        ),
        generationSource:
          typeof narrativeArtifact?.generation_source === "string"
            ? narrativeArtifact.generation_source
            : "",
        evidence: evidenceBySection.get(review.section_id),
      };
    });
  }, [run, evidenceBySection]);

  const toggleEvidence = (sectionId: string) => {
    setExpandedEvidence((current) => ({
      ...current,
      [sectionId]: !current[sectionId],
    }));
  };

  const toggleMetric = (key: string) => {
    setExpandedMetrics((current) => ({
      ...current,
      [key]: !current[key],
    }));
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">RS</div>

          <div>
            <div className="brand-title">
              Regulatory Safety Reporting
            </div>
            <div className="brand-subtitle">
              Controlled AI-assisted reporting workspace
            </div>
          </div>
        </div>

        <div className="topbar-badge">
          Deterministic analysis · Human review
        </div>
      </header>

      <main className="page">
        <section className="hero">
          <div>
            <div className="eyebrow">SAFETY REPORTING WORKSPACE</div>

            <h1>
              Generate, review, ground,
              <br />
              and finalize a safety report.
            </h1>

            <p>
              Deterministic metrics remain authoritative. AI is restricted
              to controlled narrative generation and every report remains
              subject to human review.
            </p>
          </div>

          <div className="hero-flow">
            <span>DATA</span>
            <i />
            <span>ANALYSIS</span>
            <i />
            <span>NARRATIVE</span>
            <i />
            <span>REVIEW</span>
            <i />
            <span>FINAL</span>
          </div>
        </section>

        <section className="workspace">
          <form className="run-card" onSubmit={handleRun}>
            <div className="card-heading">
              <div>
                <div className="section-kicker">01 · INPUT</div>
                <h2>Start a reporting run</h2>
                <p>
                  Upload the source CSV and select whether controlled LLM
                  narrative generation should be enabled.
                </p>
              </div>

              <div className="step-number">01</div>
            </div>

            <label className="file-drop">
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={handleFileChange}
              />

              <span className="upload-icon">↑</span>

              <span className="file-copy">
                <strong>
                  {file ? file.name : "Choose a CSV dataset"}
                </strong>

                <small>
                  {file
                    ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                    : "CSV files only"}
                </small>
              </span>
            </label>

            <div className="run-controls">
              <label className="llm-toggle">
                <input
                  type="checkbox"
                  checked={useLlm}
                  onChange={(event) =>
                    setUseLlm(event.target.checked)
                  }
                />

                <span className="toggle-track">
                  <span />
                </span>

                <span>
                  <strong>Enable LLM narrative</strong>
                  <small>
                    Deterministic metrics remain the source of truth
                  </small>
                </span>
              </label>

              <button
                className="primary-button"
                type="submit"
                disabled={loading || !file}
              >
                {loading ? "Generating…" : "Generate report"}
                <span>→</span>
              </button>
            </div>
          </form>

          {error && (
            <div className="alert alert-error">
              <strong>Error</strong>
              <span>{error}</span>
            </div>
          )}

          {success && (
            <div className="alert alert-success">
              <strong>Success</strong>
              <span>{success}</span>
            </div>
          )}

          {run && (
            <>
              <section className="run-summary">
                <div className="summary-main">
                  <div className="section-kicker">
                    02 · RUN
                  </div>

                  <div className="run-title-row">
                    <h2>Reporting run</h2>

                    <span
                      className={`status-pill ${
                        run.details.status === "FINAL"
                          ? "status-final"
                          : "status-draft"
                      }`}
                    >
                      {run.details.status}
                    </span>
                  </div>

                  <div className="run-id">
                    <span>Run ID</span>
                    <code>{run.details.run_id}</code>
                  </div>
                </div>

                <div className="summary-stats">
                  <div>
                    <span>Approved</span>
                    <strong>{approvedCount}</strong>
                  </div>

                  <div>
                    <span>Flagged</span>
                    <strong>{flaggedCount}</strong>
                  </div>

                  <div>
                    <span>Pending</span>
                    <strong>{unreviewedCount}</strong>
                  </div>

                  <div>
                    <span>Sections</span>
                    <strong>{run.review.reviews.length}</strong>
                  </div>
                </div>
              </section>

              <section className="metadata-strip">
                <div>
                  <span>Report</span>
                  <strong>
                    {run.details.report_id} · v
                    {run.details.report_version}
                  </strong>
                </div>

                <div>
                  <span>Created</span>
                  <strong>
                    {formatDate(run.details.created_at_utc)}
                  </strong>
                </div>

                <div>
                  <span>Dataset SHA-256</span>
                  <strong>
                    {truncateHash(run.details.dataset_sha256)}
                  </strong>
                </div>

                <div>
                  <span>Analysis registry</span>
                  <strong>
                    {run.details.analysis_registry_version ?? "—"}
                  </strong>
                </div>
              </section>

              <section className="report-card">
                <div className="card-heading">
                  <div>
                    <div className="section-kicker">
                      03 · REPORT
                    </div>
                    <h2>Generated report</h2>
                    <p>
                      The rendered report includes deterministic evidence
                      anchors for supported metrics.
                    </p>
                  </div>

                  <a
                    className="secondary-button"
                    href={`${API_BASE}/runs/${run.details.run_id}/report`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open full report ↗
                  </a>
                </div>

                <div className="report-frame">
                  <iframe
                    title="Generated safety report"
                    src={`${API_BASE}/runs/${run.details.run_id}/report`}
                  />
                </div>
              </section>

              <section className="review-section">
                <div className="section-header">
                  <div>
                    <div className="section-kicker">
                      04 · HUMAN REVIEW
                    </div>
                    <h2>Review every narrative section</h2>
                    <p>
                      Review the actual generated narrative alongside its
                      deterministic evidence before approving the report.
                    </p>
                  </div>

                  <div className="review-progress">
                    <div className="progress-label">
                      <span>
                        {approvedCount} /{" "}
                        {run.review.reviews.length} approved
                      </span>

                      <span>
                        {run.review.reviews.length
                          ? Math.round(
                              (approvedCount /
                                run.review.reviews.length) *
                                100,
                            )
                          : 0}
                        %
                      </span>
                    </div>

                    <div className="progress-track">
                      <div
                        className="progress-value"
                        style={{
                          width: `${
                            run.review.reviews.length
                              ? (approvedCount /
                                  run.review.reviews.length) *
                                100
                              : 0
                          }%`,
                        }}
                      />
                    </div>
                  </div>
                </div>

                <div className="reviewer-bar">
                  <div>
                    <label htmlFor="reviewer">
                      Reviewer
                    </label>

                    <input
                      id="reviewer"
                      value={reviewer}
                      onChange={(event) =>
                        setReviewer(event.target.value)
                      }
                      placeholder="Enter reviewer name"
                    />
                  </div>

                  <div className="reviewer-note">
                    Approval is required for every configured section before
                    finalization.
                  </div>
                </div>

                <div className="review-grid">
                  {reviewSections.map(
                    ({
                      review,
                      narrative,
                      generationSource,
                      evidence,
                    }) => {
                      const evidenceOpen =
                        Boolean(
                          expandedEvidence[
                            review.section_id
                          ],
                        );

                      return (
                        <article
                          className={`review-card ${
                            review.status === "APPROVED"
                              ? "review-approved"
                              : review.status === "FLAGGED"
                                ? "review-flagged"
                                : ""
                          }`}
                          key={review.section_id}
                        >
                          <div className="review-card-top">
                            <div>
                              <div className="review-section-id">
                                {review.section_id}
                              </div>

                              <h3>
                                {review.section_id
                                  .replaceAll("_", " ")
                                  .replace(/\b\w/g, (char) =>
                                    char.toUpperCase(),
                                  )}
                              </h3>
                            </div>

                            <span
                              className={`review-status status-${review.status.toLowerCase()}`}
                            >
                              {statusLabel(review.status)}
                            </span>
                          </div>

                          <div className="narrative-panel">
                            <div className="panel-label">
                              {narrativeSourceLabel(
                                generationSource,
                              )}
                            </div>

                            <div className="narrative-text">
                              {narrative ? (
                                narrative
                              ) : (
                                <span className="muted">
                                  Narrative is not available in this
                                  artifact.
                                </span>
                              )}
                            </div>
                          </div>

                          <div className="evidence-summary">
                            <button
                              type="button"
                              className="evidence-summary-button"
                              onClick={() =>
                                toggleEvidence(
                                  review.section_id,
                                )
                              }
                            >
                              <span>
                                <strong>
                                  Deterministic evidence
                                </strong>

                                <small>
                                  {evidence?.metric_ids.length ??
                                    0}{" "}
                                  metrics ·{" "}
                                  {evidence
                                    ? Object.values(
                                        evidence.supporting_ids,
                                      ).reduce(
                                        (total, ids) =>
                                          total +
                                          ids.length,
                                        0,
                                      )
                                    : 0}{" "}
                                  supporting records
                                </small>
                              </span>

                              <span>
                                {evidenceOpen ? "−" : "+"}
                              </span>
                            </button>

                            {evidenceOpen && evidence && (
                              <div className="evidence-list">
                                {evidence.metric_ids.map(
                                  (metricId) => {
                                    const ids =
                                      evidence
                                        .supporting_ids[
                                        metricId
                                      ] ?? [];

                                    const metricKey = `${review.section_id}:${metricId}`;

                                    const metricOpen =
                                      Boolean(
                                        expandedMetrics[
                                          metricKey
                                        ],
                                      );

                                    return (
                                      <div
                                        className="metric-row"
                                        key={metricId}
                                      >
                                        <button
                                          type="button"
                                          className="metric-row-button"
                                          onClick={() =>
                                            toggleMetric(
                                              metricKey,
                                            )
                                          }
                                        >
                                          <span>
                                            <code>
                                              {metricId}
                                            </code>

                                            <small>
                                              {ids.length}{" "}
                                              supporting
                                              records
                                            </small>
                                          </span>

                                          <span>
                                            {metricOpen
                                              ? "−"
                                              : "+"}
                                          </span>
                                        </button>

                                        {metricOpen &&
                                          ids.length > 0 && (
                                            <div className="supporting-id-list">
                                              {ids.map(
                                                (id) => (
                                                  <code
                                                    key={id}
                                                  >
                                                    {id}
                                                  </code>
                                                ),
                                              )}
                                            </div>
                                          )}
                                      </div>
                                    );
                                  },
                                )}
                              </div>
                            )}
                          </div>

                          <div className="review-actions">
                            <label>
                              <span>Reviewer comment</span>

                              <textarea
                                value={
                                  comments[
                                    review.section_id
                                  ] ??
                                  review.comment ??
                                  ""
                                }
                                onChange={(event) =>
                                  setComments(
                                    (current) => ({
                                      ...current,
                                      [review.section_id]:
                                        event.target.value,
                                    }),
                                  )
                                }
                                placeholder="Optional review comment"
                                rows={3}
                              />
                            </label>

                            <div className="action-row">
                              <button
                                type="button"
                                className="flag-button"
                                disabled={
                                  reviewLoading ===
                                    review.section_id ||
                                  !reviewer.trim()
                                }
                                onClick={() =>
                                  updateReview(
                                    review.section_id,
                                    "FLAGGED",
                                  )
                                }
                              >
                                Flag
                              </button>

                              <button
                                type="button"
                                className="approve-button"
                                disabled={
                                  reviewLoading ===
                                    review.section_id ||
                                  !reviewer.trim()
                                }
                                onClick={() =>
                                  updateReview(
                                    review.section_id,
                                    "APPROVED",
                                  )
                                }
                              >
                                {reviewLoading ===
                                review.section_id
                                  ? "Saving…"
                                  : "Approve section"}
                              </button>
                            </div>

                            {!reviewer.trim() && (
                              <p className="review-hint">
                                Enter a reviewer name above to enable
                                Approve and Flag.
                              </p>
                            )}
                          </div>
                        </article>
                      );
                    },
                  )}
                </div>
              </section>

              <section className="finalize-card">
                <div>
                  <div className="section-kicker">
                    05 · FINALIZATION
                  </div>

                  <h2>Finalize report</h2>

                  <p>
                    The report can only transition from DRAFT to FINAL when
                    every configured section has been explicitly approved.
                  </p>
                </div>

                <div className="finalize-action">
                  <div
                    className={
                      allApproved
                        ? "gate-ready"
                        : "gate-blocked"
                    }
                  >
                    <span className="gate-dot" />

                    {allApproved
                      ? "All sections approved"
                      : `${unreviewedCount + flaggedCount} section${
                          unreviewedCount + flaggedCount ===
                          1
                            ? ""
                            : "s"
                        } still require review`}
                  </div>

                  <button
                    type="button"
                    className="finalize-button"
                    disabled={
                      !allApproved ||
                      finalizing ||
                      run.details.status === "FINAL"
                    }
                    onClick={finalizeReport}
                  >
                    {finalizing
                      ? "Finalizing…"
                      : run.details.status === "FINAL"
                        ? "Report finalized"
                        : "Finalize report"}
                  </button>
                </div>
              </section>

              <section className="evidence-explorer">
                <div className="section-header">
                  <div>
                    <div className="section-kicker">
                      06 · EVIDENCE EXPLORER
                    </div>

                    <h2>Deterministic traceability</h2>

                    <p>
                      Explore the metric-to-supporting-record mapping without
                      sending raw case records to the LLM.
                    </p>
                  </div>
                </div>

                <div className="explorer-grid">
                  {run.evidence.map((section) => {
                    const open =
                      Boolean(
                        expandedEvidence[
                          `explorer:${section.section_id}`
                        ],
                      );

                    const totalIds = Object.values(
                      section.supporting_ids,
                    ).reduce(
                      (total, ids) => total + ids.length,
                      0,
                    );

                    return (
                      <article
                        className="explorer-card"
                        key={section.section_id}
                      >
                        <button
                          type="button"
                          className="explorer-heading"
                          onClick={() =>
                            toggleEvidence(
                              `explorer:${section.section_id}`,
                            )
                          }
                        >
                          <span>
                            <strong>
                              {section.section_id
                                .replaceAll("_", " ")
                                .replace(/\b\w/g, (char) =>
                                  char.toUpperCase(),
                                )}
                            </strong>

                            <small>
                              {section.metric_ids.length} metrics ·{" "}
                              {totalIds} supporting records
                            </small>
                          </span>

                          <span>{open ? "−" : "+"}</span>
                        </button>

                        {open && (
                          <div className="explorer-body">
                            {section.metric_ids.map(
                              (metricId) => {
                                const ids =
                                  section.supporting_ids[
                                    metricId
                                  ] ?? [];

                                return (
                                  <div
                                    className="explorer-metric"
                                    key={metricId}
                                  >
                                    <div>
                                      <code>
                                        {metricId}
                                      </code>

                                      <span>
                                        {ids.length} supporting
                                        records
                                      </span>
                                    </div>

                                    {ids.length > 0 && (
                                      <details>
                                        <summary>
                                          View supporting IDs
                                        </summary>

                                        <div className="explorer-ids">
                                          {ids.map(
                                            (id) => (
                                              <code
                                                key={id}
                                              >
                                                {id}
                                              </code>
                                            ),
                                          )}
                                        </div>
                                      </details>
                                    )}
                                  </div>
                                );
                              },
                            )}
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>
              </section>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;