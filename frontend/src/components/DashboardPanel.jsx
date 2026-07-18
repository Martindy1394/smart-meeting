function fmtDate(iso) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function fmtDuration(sec) {
  const s = Math.max(0, Math.round(Number(sec) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) {
    return `${h}h ${m}m`;
  }
  if (m > 0) {
    return `${m}m ${r}s`;
  }
  return `${r}s`;
}

function statusLabel(status) {
  if (status === "finalized") return "Refined";
  if (status === "processing") return "Processing";
  if (status === "recording") return "Draft";
  if (status === "failed") return "Failed";
  return status || "Saved";
}

function startOfWeek(d = new Date()) {
  const out = new Date(d);
  const day = out.getDay(); // 0 = Sun
  out.setHours(0, 0, 0, 0);
  out.setDate(out.getDate() - day);
  return out;
}

function meetingWhen(meeting) {
  return meeting.meeting_date || meeting.created_at || meeting.updated_at;
}

function buildStats(meetings) {
  const weekStart = startOfWeek();
  let withSummary = 0;
  let withTranscript = 0;
  let withAudio = 0;
  let thisWeek = 0;
  let totalDuration = 0;

  for (const m of meetings) {
    if (m.has_summary || (m.summary || "").trim()) withSummary += 1;
    if (m.has_transcript || m.status === "finalized") withTranscript += 1;
    if (m.has_audio) withAudio += 1;
    totalDuration += Number(m.duration_seconds) || 0;
    const when = meetingWhen(m);
    if (when) {
      const t = new Date(when);
      if (!Number.isNaN(t.getTime()) && t >= weekStart) thisWeek += 1;
    }
  }

  return {
    total: meetings.length,
    withSummary,
    withTranscript,
    withAudio,
    thisWeek,
    totalDuration,
  };
}

function StatCard({ label, value, hint }) {
  return (
    <div className="dash-stat">
      <div className="dash-stat-value">{value}</div>
      <div className="dash-stat-label">{label}</div>
      {hint ? <div className="dash-stat-hint">{hint}</div> : null}
    </div>
  );
}

function SummaryCard({ meeting, onSelect }) {
  const summary = (meeting.summary || "").trim();
  const when = meetingWhen(meeting);

  return (
    <article className="dash-summary-card">
      <header className="dash-summary-head">
        <div className="dash-summary-titles">
          <h4 className="dash-summary-title">
            {meeting.title || "Untitled meeting"}
          </h4>
          <div className="dash-summary-meta">
            {when ? <span>{fmtDate(when)}</span> : null}
            {meeting.venue ? <span>{meeting.venue}</span> : null}
            {meeting.duration_seconds > 0 ? (
              <span>{fmtDuration(meeting.duration_seconds)}</span>
            ) : null}
            {meeting.language && meeting.language !== "auto" ? (
              <span>{meeting.language}</span>
            ) : null}
          </div>
        </div>
        <div className="dash-summary-actions">
          <span
            className={`badge ${
              meeting.status === "finalized"
                ? "refined"
                : meeting.status === "processing"
                  ? "processing"
                  : ""
            }`}
          >
            {statusLabel(meeting.status)}
          </span>
          <button
            type="button"
            className="btn secondary"
            onClick={() => onSelect(meeting.id)}
          >
            Open
          </button>
        </div>
      </header>
      <div className="dash-summary-body">
        {summary ? (
          <p className="dash-summary-text">{summary}</p>
        ) : (
          <p className="dash-summary-empty">
            No summary yet. Open the meeting and generate a summary from the
            transcript.
          </p>
        )}
      </div>
    </article>
  );
}

export default function DashboardPanel({
  meetings,
  loading,
  onSelect,
  onCreate,
}) {
  const stats = buildStats(meetings);
  const sorted = [...meetings].sort((a, b) => {
    const ta = new Date(meetingWhen(a) || 0).getTime();
    const tb = new Date(meetingWhen(b) || 0).getTime();
    return tb - ta;
  });
  const withSummaries = sorted.filter(
    (m) => m.has_summary || (m.summary || "").trim()
  );
  const recent = sorted.slice(0, 8);
  const featured = withSummaries.length > 0 ? withSummaries.slice(0, 8) : recent;

  return (
    <div className="content dashboard-panel">
      <div className="dash-intro">
        <div>
          <h2 className="dash-heading">Meeting overview</h2>
          <p className="dash-sub">
            Counts across your saved meetings, plus the latest BART summaries.
          </p>
        </div>
        <div className="dash-intro-actions">
          <button type="button" className="btn" onClick={onCreate}>
            + New meeting
          </button>
        </div>
      </div>

      {loading ? (
        <div className="center-spin">
          <span className="spinner" /> Loading dashboard…
        </div>
      ) : (
        <>
          <div className="dash-stats" aria-label="Meeting statistics">
            <StatCard label="Total meetings" value={stats.total} />
            <StatCard
              label="This week"
              value={stats.thisWeek}
              hint="By meeting date"
            />
            <StatCard label="With transcript" value={stats.withTranscript} />
            <StatCard label="With summary" value={stats.withSummary} />
            <StatCard label="With audio" value={stats.withAudio} />
            <StatCard
              label="Recorded time"
              value={fmtDuration(stats.totalDuration)}
              hint="All meetings"
            />
          </div>

          <div className="card dash-card">
            <div className="card-head">
              <h3>
                {withSummaries.length > 0
                  ? "Meeting summaries"
                  : "Recent meetings"}
              </h3>
              <span className="card-tag">
                {withSummaries.length > 0
                  ? `${withSummaries.length} summarized`
                  : `${meetings.length} total`}
              </span>
            </div>
            <div className="card-body dash-card-body">
              {featured.length === 0 ? (
                <div className="dash-empty">
                  <h4>No meetings yet</h4>
                  <p>
                    Start a meeting to capture live captions, then generate a
                    summary. It will show up here.
                  </p>
                  <button type="button" className="btn" onClick={onCreate}>
                    + New meeting
                  </button>
                </div>
              ) : (
                <div className="dash-summary-list">
                  {featured.map((m) => (
                    <SummaryCard key={m.id} meeting={m} onSelect={onSelect} />
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
