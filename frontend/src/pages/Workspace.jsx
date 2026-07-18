import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import ConfirmDialog from "../components/ConfirmDialog.jsx";
import DashboardPanel from "../components/DashboardPanel.jsx";
import HistoryPanel from "../components/HistoryPanel.jsx";
import MeetingRoom from "../components/MeetingRoom.jsx";
import SettingsPanel from "../components/SettingsPanel.jsx";
import Sidebar from "../components/Sidebar.jsx";

export default function Workspace() {
  const [section, setSection] = useState("dashboard");
  const [meetings, setMeetings] = useState([]);
  const [loadingList, setLoadingList] = useState(true);
  const [search, setSearch] = useState("");
  const [activeId, setActiveId] = useState(null);
  const [activeMeeting, setActiveMeeting] = useState(null);
  const [loadingMeeting, setLoadingMeeting] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [saveControls, setSaveControls] = useState(null);
  // True when opening an existing meeting from History (review mode).
  const [historyView, setHistoryView] = useState(false);
  const searchTimer = useRef(null);

  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [saveConfirmOpen, setSaveConfirmOpen] = useState(false);

  const loadMeetings = useCallback(async (q) => {
    setLoadingList(true);
    try {
      const list = await api.listMeetings(q);
      setMeetings(list);
    } catch {
      setMeetings([]);
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => {
    loadMeetings("");
  }, [loadMeetings]);

  // Debounced search for history.
  useEffect(() => {
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => loadMeetings(search), 300);
    return () => clearTimeout(searchTimer.current);
  }, [search, loadMeetings]);

  const handleSectionChange = useCallback((next) => {
    // Recordings was merged into History.
    setSection(next === "recordings" ? "history" : next);
  }, []);

  const selectMeeting = useCallback(async (id) => {
    setSection("meeting");
    setActiveId(id);
    setSaveControls(null);
    setHistoryView(true);
    setLoadingMeeting(true);
    try {
      const detail = await api.getMeeting(id);
      setActiveMeeting(detail);
    } catch {
      setActiveMeeting(null);
    } finally {
      setLoadingMeeting(false);
    }
  }, []);

  const backToHistory = useCallback(() => {
    setSection("history");
    setHistoryView(false);
    setSaveControls(null);
    setLoadingMeeting(false);
    // Keep activeId so the list can highlight the last opened meeting.
  }, []);

  const createMeeting = useCallback(async () => {
    try {
      const detail = await api.createMeeting({
        title: "",
        language: "auto",
        // Always create with the current date/time (local → ISO).
        meeting_date: new Date().toISOString(),
      });
      setSection("meeting");
      setSaveControls(null);
      setHistoryView(false);
      setActiveId(detail.id);
      setActiveMeeting(detail);
      setLoadingMeeting(false);
      Promise.resolve().then(() => loadMeetings(search));
    } catch {
      /* ignore */
    }
  }, [loadMeetings, search]);

  const requestDeleteMeeting = useCallback(
    (id) => {
      const found = meetings.find((m) => m.id === id);
      setDeleteTarget({
        id,
        title: (found?.title || "").trim() || "Untitled meeting",
      });
    },
    [meetings]
  );

  const confirmDeleteMeeting = useCallback(async () => {
    if (!deleteTarget?.id) return;
    setDeleteBusy(true);
    try {
      const id = deleteTarget.id;
      await api.deleteMeeting(id);
      setDeleteTarget(null);
      if (id === activeId) {
        setActiveId(null);
        setActiveMeeting(null);
        setSaveControls(null);
        setHistoryView(false);
        setSection("dashboard");
      }
      loadMeetings(search);
    } catch {
      /* ignore */
    } finally {
      setDeleteBusy(false);
    }
  }, [activeId, deleteTarget, loadMeetings, search]);

  const confirmSaveMeeting = useCallback(async () => {
    if (!saveControls?.save) return;
    setSaveConfirmOpen(false);
    await saveControls.save();
  }, [saveControls]);

  const refreshActive = useCallback(
    async (updated) => {
      if (updated && updated.id) {
        setActiveMeeting((m) => ({ ...m, ...updated }));
      } else if (activeId) {
        try {
          const detail = await api.getMeeting(activeId);
          setActiveMeeting(detail);
        } catch {
          /* keep current */
        }
      }
      loadMeetings(search);
    },
    [activeId, loadMeetings, search]
  );

  const showDashboard = section === "dashboard";
  const showSettings = section === "settings";
  const showHistory = section === "history" || section === "recordings";
  const showMeeting = section === "meeting";

  return (
    <div className="app-shell">
      <Sidebar
        section={section}
        onSectionChange={handleSectionChange}
        onCreate={createMeeting}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="main">
        <div className="topbar">
          <div className="topbar-left">
            <button
              className="btn secondary menu-btn"
              onClick={() => setSidebarOpen(true)}
            >
              ☰
            </button>
            <h1>
              {showDashboard
                ? "Dashboard"
                : showSettings
                  ? "Settings"
                  : showHistory
                    ? "History"
                    : activeMeeting
                      ? activeMeeting.title || "Untitled meeting"
                      : "Smart Meeting"}
            </h1>
          </div>
          {showMeeting && activeMeeting && saveControls && (
            <div className="topbar-right">
              <button
                className="btn"
                onClick={() => setSaveConfirmOpen(true)}
                disabled={saveControls.disabled || saveControls.saving}
              >
                {saveControls.saving ? <span className="spinner" /> : "Save"}
              </button>
            </div>
          )}
        </div>

        {showDashboard ? (
          <DashboardPanel
            meetings={meetings}
            loading={loadingList}
            onSelect={selectMeeting}
            onCreate={createMeeting}
          />
        ) : showSettings ? (
          <SettingsPanel />
        ) : showHistory ? (
          <HistoryPanel
            meetings={meetings}
            loading={loadingList}
            search={search}
            onSearch={setSearch}
            activeId={activeId}
            onSelect={selectMeeting}
            onDelete={requestDeleteMeeting}
            onCreate={createMeeting}
          />
        ) : loadingMeeting ? (
          <div className="center-spin">
            <span className="spinner" /> Loading meeting…
          </div>
        ) : activeMeeting ? (
          <MeetingRoom
            key={activeMeeting.id}
            meeting={activeMeeting}
            onMeetingUpdated={refreshActive}
            onSaveControls={setSaveControls}
            historyView={historyView}
            onBack={historyView ? backToHistory : undefined}
          />
        ) : (
          <DashboardPanel
            meetings={meetings}
            loading={loadingList}
            onSelect={selectMeeting}
            onCreate={createMeeting}
            onGoHistory={() => setSection("history")}
          />
        )}
      </div>

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        tone="danger"
        title="Delete meeting?"
        message={
          deleteTarget
            ? `Remove “${deleteTarget.title}” permanently? The transcript, summary, translation, and audio cannot be recovered.`
            : ""
        }
        confirmLabel="Delete"
        cancelLabel="Keep meeting"
        busy={deleteBusy}
        onCancel={() => {
          if (!deleteBusy) setDeleteTarget(null);
        }}
        onConfirm={confirmDeleteMeeting}
      />

      <ConfirmDialog
        open={saveConfirmOpen}
        title="Save meeting details?"
        message="Save the current title, venue, date & time, and attendees for this meeting?"
        confirmLabel="Save"
        cancelLabel="Cancel"
        busy={Boolean(saveControls?.saving)}
        onCancel={() => setSaveConfirmOpen(false)}
        onConfirm={confirmSaveMeeting}
      />
    </div>
  );
}
