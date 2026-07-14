import { useAuth } from "../context/AuthContext.jsx";

function fmtDate(iso) {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

export default function Sidebar({
  meetings,
  loading,
  activeId,
  search,
  onSearch,
  onSelect,
  onCreate,
  onDelete,
  open,
  onClose,
}) {
  const { user, logout } = useAuth();

  return (
    <>
      {open && <div className="backdrop" onClick={onClose} />}
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="sidebar-header">
          <div className="brand">
            <span className="dot" /> Smart Meeting
          </div>
          <button
            className="btn"
            style={{ width: "100%", marginTop: 14 }}
            onClick={() => {
              onCreate();
              onClose && onClose();
            }}
          >
            + New meeting
          </button>
        </div>

        <div className="sidebar-search">
          <input
            placeholder="Search meetings…"
            value={search}
            onChange={(e) => onSearch(e.target.value)}
          />
        </div>

        <div className="sidebar-section-label">History</div>

        <div className="meeting-list">
          {loading ? (
            <div className="center-spin">
              <span className="spinner" /> Loading…
            </div>
          ) : meetings.length === 0 ? (
            <div className="placeholder" style={{ height: "auto", padding: 24 }}>
              No meetings yet.
            </div>
          ) : (
            meetings.map((m) => (
              <div
                key={m.id}
                className={`meeting-item ${m.id === activeId ? "active" : ""}`}
              >
                <div className="meeting-item-body">
                  <div className="title">{m.title || "Untitled meeting"}</div>
                  <div className="meta">
                    <span>{fmtDate(m.meeting_date || m.created_at)}</span>
                    {m.status === "finalized" && (
                      <span className="badge refined">Refined</span>
                    )}
                    {m.status === "processing" && (
                      <span className="badge processing">Processing</span>
                    )}
                    {m.status === "recording" && (
                      <span className="badge">Draft</span>
                    )}
                  </div>
                </div>
                <div className="meeting-item-actions">
                  <button
                    type="button"
                    className="btn secondary meeting-action-btn"
                    onClick={() => {
                      onSelect(m.id);
                      onClose && onClose();
                    }}
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
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <span className="user-chip">{user?.email}</span>
          <button className="btn ghost" onClick={logout}>
            Logout
          </button>
        </div>
      </aside>
    </>
  );
}
