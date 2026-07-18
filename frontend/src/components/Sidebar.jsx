import { useAuth } from "../context/AuthContext.jsx";

export default function Sidebar({
  section,
  onSectionChange,
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
        </div>

        <nav className="side-menu" aria-label="Main menu">
          <button
            type="button"
            className={`side-menu-item ${menuSection === "dashboard" ? "active" : ""}`}
            onClick={() => {
              onSectionChange("dashboard");
              onClose && onClose();
            }}
          >
            Dashboard
          </button>
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
