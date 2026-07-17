import { useAuth } from "../context/AuthContext.jsx";

function MenuIcon({ children }) {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

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
            title="History"
            aria-label="History"
          >
            <MenuIcon>
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </MenuIcon>
          </button>
          <button
            type="button"
            className={`side-menu-item ${menuSection === "settings" ? "active" : ""}`}
            onClick={() => {
              onSectionChange("settings");
              onClose && onClose();
            }}
            title="Settings"
            aria-label="Settings"
          >
            <MenuIcon>
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </MenuIcon>
          </button>
        </nav>

        <div className="sidebar-settings-note">
          {menuSection === "history" ? (
            <p>
              Open saved meetings and recordings from History. Copy, download,
              play, or remove records from one list.
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
