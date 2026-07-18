import { useAuth } from "../context/AuthContext.jsx";

function userInitials(display, username) {
  const name = (display || "").trim();
  if (name) {
    const parts = name.split(/\s+/).filter(Boolean);
    if (parts.length >= 2) {
      return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
    }
    return name.slice(0, 2).toUpperCase();
  }
  const u = (username || "").trim();
  return u ? u.slice(0, 2).toUpperCase() : "?";
}

function UserAvatar({ display, username }) {
  const initials = userInitials(display, username);
  return (
    <span className="user-avatar" aria-hidden="true">
      {initials ? (
        <span className="user-avatar-initials">{initials}</span>
      ) : (
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
          <circle cx="12" cy="7" r="4" />
        </svg>
      )}
    </span>
  );
}

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
          <button
            type="button"
            className="user-account"
            onClick={() => {
              onSectionChange("settings");
              onClose && onClose();
            }}
            title="Account settings"
            aria-label="Open account settings"
          >
            <UserAvatar display={display} username={user?.username} />
            <div className="user-chip-block">
              <span className="user-chip">{display || "Account"}</span>
              {user?.username && (
                <span className="user-chip-sub">@{user.username}</span>
              )}
            </div>
          </button>
          <button className="btn ghost" onClick={logout}>
            Logout
          </button>
        </div>
      </aside>
    </>
  );
}
