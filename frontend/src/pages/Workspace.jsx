import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import HistoryPanel from "../components/HistoryPanel.jsx";
import MeetingRoom from "../components/MeetingRoom.jsx";
import RecordingsPanel from "../components/RecordingsPanel.jsx";
import SettingsPanel from "../components/SettingsPanel.jsx";
import Sidebar from "../components/Sidebar.jsx";

export default function Workspace() {
  const [section, setSection] = useState("history");
  const [meetings, setMeetings] = useState([]);
  const [recordings, setRecordings] = useState([]);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingRecordings, setLoadingRecordings] = useState(false);
  const [search, setSearch] = useState("");
  const [recordingSearch, setRecordingSearch] = useState("");
  const [activeId, setActiveId] = useState(null);
  const [activeMeeting, setActiveMeeting] = useState(null);
  const [loadingMeeting, setLoadingMeeting] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [saveControls, setSaveControls] = useState(null);
  const searchTimer = useRef(null);
  const recordingSearchTimer = useRef(null);

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

  const loadRecordings = useCallback(async (q) => {
    setLoadingRecordings(true);
    try {
      const list = await api.listMeetings(q, { hasAudio: true });
      setRecordings(list);
    } catch {
      setRecordings([]);
    } finally {
      setLoadingRecordings(false);
    }
  }, []);

  useEffect(() => {
    loadMeetings("");
  }, [loadMeetings]);

  useEffect(() => {
    if (section === "recordings") {
      loadRecordings(recordingSearch);
    }
  }, [section, loadRecordings]); // eslint-disable-line react-hooks/exhaustive-deps

  // Debounced search for history.
  useEffect(() => {
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => loadMeetings(search), 300);
    return () => clearTimeout(searchTimer.current);
  }, [search, loadMeetings]);

  // Debounced search for recordings.
  useEffect(() => {
    if (section !== "recordings") return undefined;
    if (recordingSearchTimer.current) clearTimeout(recordingSearchTimer.current);
    recordingSearchTimer.current = setTimeout(
      () => loadRecordings(recordingSearch),
      300
    );
    return () => clearTimeout(recordingSearchTimer.current);
  }, [recordingSearch, loadRecordings, section]);

  const selectMeeting = useCallback(async (id) => {
    setSection("meeting");
    setActiveId(id);
    setSaveControls(null);
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

  const createMeeting = useCallback(async () => {
    try {
      const detail = await api.createMeeting({
        title: "",
        language: "hil",
        meeting_date: new Date().toISOString(),
      });
      setSection("meeting");
      setSaveControls(null);
      setActiveId(detail.id);
      setActiveMeeting(detail);
      loadMeetings(search);
      loadRecordings(recordingSearch);
    } catch {
      /* ignore */
    }
  }, [loadMeetings, loadRecordings, search, recordingSearch]);

  const deleteMeeting = useCallback(
    async (id) => {
      if (!window.confirm("Remove this meeting permanently?")) return;
      try {
        await api.deleteMeeting(id);
        if (id === activeId) {
          setActiveId(null);
          setActiveMeeting(null);
          setSaveControls(null);
          setSection("history");
        }
        loadMeetings(search);
        loadRecordings(recordingSearch);
      } catch {
        /* ignore */
      }
    },
    [activeId, loadMeetings, loadRecordings, search, recordingSearch]
  );

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
      loadRecordings(recordingSearch);
    },
    [activeId, loadMeetings, loadRecordings, search, recordingSearch]
  );

  const showSettings = section === "settings";
  const showHistory = section === "history";
  const showRecordings = section === "recordings";
  const showMeeting = section === "meeting";

  return (
    <div className="app-shell">
      <Sidebar
        section={section}
        onSectionChange={setSection}
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
              {showSettings
                ? "Settings"
                : showHistory
                  ? "History"
                  : showRecordings
                    ? "Recordings"
                    : activeMeeting
                      ? activeMeeting.title || "Untitled meeting"
                      : "Smart Meeting"}
            </h1>
          </div>
          {showMeeting && activeMeeting && saveControls && (
            <div className="topbar-right">
              <button
                className="btn"
                onClick={() => saveControls.save()}
                disabled={saveControls.disabled || saveControls.saving}
              >
                {saveControls.saving ? <span className="spinner" /> : "Save"}
              </button>
            </div>
          )}
        </div>

        {showSettings ? (
          <SettingsPanel />
        ) : showHistory ? (
          <HistoryPanel
            meetings={meetings}
            loading={loadingList}
            search={search}
            onSearch={setSearch}
            activeId={activeId}
            onSelect={selectMeeting}
            onDelete={deleteMeeting}
            onCreate={createMeeting}
          />
        ) : showRecordings ? (
          <RecordingsPanel
            recordings={recordings}
            loading={loadingRecordings}
            search={recordingSearch}
            onSearch={setRecordingSearch}
            onOpen={selectMeeting}
            onDelete={deleteMeeting}
            onRefresh={() => loadRecordings(recordingSearch)}
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
          />
        ) : (
          <div className="empty-state">
            <div style={{ fontSize: 48 }}>🗒️</div>
            <h2>Welcome to Smart Meeting</h2>
            <p>
              Create a new meeting to start live transcription, then summarize
              and translate the results. Open History or Recordings anytime to
              revisit saved work.
            </p>
            <button className="btn" onClick={createMeeting}>
              + New meeting
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
