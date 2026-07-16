import { useAuth } from "../context/AuthContext.jsx";

export default function Sidebar({
  section,
  onSectionChange,
  onCreate,
  open,
  onClose,
}) {
  const { user, logout } = useAuth();
  const display =
    user?.full_name ||
    [user?.first_name, user?.last_name].filter(Boolean).join(" ") ||
    user?.username ||
    "";

  const menuSection =
    section === "meeting" || section === "recordings" ? "history" : section;

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

        <nav className="side-menu" aria-label="Main menu">
          <button
            type="button"
            className={`side-menu-item ${menuSection === "history" ? "active" : ""}`}
            onClick={() => {
              onSectionChange("history");
              onClose && onClose();
            }}
          >
            History
          </button>
          <button
            type="button"
            className={`side-menu-item ${menuSection === "settings" ? "active" : ""}`}
            onClick={() => {
              onSectionChange("settings");
              onClose && onClose();
            }}
          >
            Settings
          </button>
        </nav>

        <div className="sidebar-settings-note">
          {menuSection === "history" ? (
            <p>
              <strong>History</strong> holds every saved meeting and recording.
              Copy, download, play, open, or remove records from one list.
            </p>
          ) : menuSection === "settings" ? (
            <p>
              View and edit your profile details in the main panel — name,
              position, workplace, working email, username, and password.
            </p>
          ) : (
            <p>Use the menu to switch between history and account settings.</p>
          )}
        </div>

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
