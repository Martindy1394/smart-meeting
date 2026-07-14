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
  section,
  onSectionChange,
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
  const display =
    user?.full_name ||
    [user?.first_name, user?.last_name].filter(Boolean).join(" ") ||
    user?.username ||
    "";

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
              onSectionChange("history");
              onCreate();
              onClose && onClose();
            }}
          >
            + New meeting
          </button>
        </div>

        <nav className="side-menu" aria-label="Main menu">
          <button
            type="button"
            className={`side-menu-item ${section === "history" ? "active" : ""}`}
            onClick={() => onSectionChange("history")}
          >
            History
          </button>
          <button
            type="button"
            className={`side-menu-item ${section === "settings" ? "active" : ""}`}
            onClick={() => {
              onSectionChange("settings");
              onClose && onClose();
            }}
          >
            Settings
          </button>
        </nav>

        {section === "history" && (
          <>
            <div className="sidebar-search">
              <input
                placeholder="Search meetings…"
                value={search}
                onChange={(e) => onSearch(e.target.value)}
              />
            </div>

            <div className="sidebar-section-label">Saved records</div>

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
          </>
        )}

        {section === "settings" && (
          <div className="sidebar-settings-note">
            <p>
              View and edit your profile details in the main panel — name,
              position, workplace, working email, username, and password.
            </p>
          </div>
        )}

        <div className="sidebar-footer">
          <div className="user-chip-block">
            <span className="user-chip">{display}</span>
            {user?.username && (
              <span className="user-chip-sub">@{user.username}</span>
            )}
          </div>
          <button className="btn ghost" onClick={logout}>
            Logout
          </button>
        </div>
      </aside>
    </>
  );
}
