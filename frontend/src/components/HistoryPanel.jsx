function fmtDate(iso) {
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

function statusLabel(status) {
  if (status === "finalized") return "Refined";
  if (status === "processing") return "Processing";
  if (status === "recording") return "Draft";
  if (status === "failed") return "Failed";
  return status || "Saved";
}

export default function HistoryPanel({
  meetings,
  loading,
  search,
  onSearch,
  activeId,
  onSelect,
  onDelete,
  onCreate,
}) {
  return (
    <div className="content history-panel">
      <div className="card history-card">
        <div className="card-head">
          <h3>History</h3>
          <span className="card-tag">Saved records</span>
        </div>
        <div className="card-body">
          <p className="settings-intro">
            All of your saved meeting records live here. Open one to continue
            working, or remove records you no longer need.
          </p>

          <div className="history-toolbar">
            <input
              className="history-search"
              placeholder="Search meetings…"
              value={search}
              onChange={(e) => onSearch(e.target.value)}
            />
            <button type="button" className="btn" onClick={onCreate}>
              + New meeting
            </button>
          </div>

          {loading ? (
            <div className="center-spin">
              <span className="spinner" /> Loading…
            </div>
          ) : meetings.length === 0 ? (
            <div className="placeholder" style={{ height: "auto", padding: 32 }}>
              No saved meetings yet.
            </div>
          ) : (
            <div className="history-list">
              {meetings.map((m) => (
                <div
                  key={m.id}
                  className={`history-row ${m.id === activeId ? "active" : ""}`}
                >
                  <div className="history-row-main">
                    <div className="history-row-title">
                      {m.title || "Untitled meeting"}
                    </div>
                    <div className="history-row-meta">
                      <span>{fmtDate(m.meeting_date || m.created_at)}</span>
                      {m.venue ? <span>{m.venue}</span> : null}
                      <span
                        className={`badge ${
                          m.status === "finalized"
                            ? "refined"
                            : m.status === "processing"
                              ? "processing"
                              : ""
                        }`}
                      >
                        {statusLabel(m.status)}
                      </span>
                      {m.has_audio ? <span className="badge">Audio</span> : null}
                      {m.has_summary ? <span className="badge">Summary</span> : null}
                      {m.has_translation ? (
                        <span className="badge">Translation</span>
                      ) : null}
                    </div>
                  </div>
                  <div className="history-row-actions">
                    <button
                      type="button"
                      className="btn secondary meeting-action-btn"
                      onClick={() => onSelect(m.id)}
                    >
                      Open
                    </button>
                    <button
                      type="button"
                      className="btn ghost meeting-action-btn meeting-remove-btn"
                      onClick={() => onDelete(m.id)}
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
