import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import MeetingRoom from "../components/MeetingRoom.jsx";
import Sidebar from "../components/Sidebar.jsx";

export default function Workspace() {
  const [meetings, setMeetings] = useState([]);
  const [loadingList, setLoadingList] = useState(true);
  const [search, setSearch] = useState("");
  const [activeId, setActiveId] = useState(null);
  const [activeMeeting, setActiveMeeting] = useState(null);
  const [loadingMeeting, setLoadingMeeting] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const searchTimer = useRef(null);

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

  // Debounced search.
  useEffect(() => {
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => loadMeetings(search), 300);
    return () => clearTimeout(searchTimer.current);
  }, [search, loadMeetings]);

  const selectMeeting = useCallback(async (id) => {
    setActiveId(id);
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
      });
      setActiveId(detail.id);
      setActiveMeeting(detail);
      loadMeetings(search);
    } catch {
      /* ignore */
    }
  }, [loadMeetings, search]);

  const deleteMeeting = useCallback(
    async (id) => {
      if (!window.confirm("Delete this meeting permanently?")) return;
      try {
        await api.deleteMeeting(id);
        if (id === activeId) {
          setActiveId(null);
          setActiveMeeting(null);
        }
        loadMeetings(search);
      } catch {
        /* ignore */
      }
    },
    [activeId, loadMeetings, search]
  );

  const refreshActive = useCallback(
    (updated) => {
      if (updated && updated.id) {
        setActiveMeeting((m) => ({ ...m, ...updated }));
      }
      loadMeetings(search);
    },
    [loadMeetings, search]
  );

  return (
    <div className="app-shell">
      <Sidebar
        meetings={meetings}
        loading={loadingList}
        activeId={activeId}
        search={search}
        onSearch={setSearch}
        onSelect={selectMeeting}
        onCreate={createMeeting}
        onDelete={deleteMeeting}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="main">
        <div className="topbar">
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button
              className="btn secondary menu-btn"
              onClick={() => setSidebarOpen(true)}
            >
              ☰
            </button>
            <h1>
              {activeMeeting
                ? activeMeeting.title || "Untitled meeting"
                : "Smart Meeting"}
            </h1>
          </div>
        </div>

        {loadingMeeting ? (
          <div className="center-spin">
            <span className="spinner" /> Loading meeting…
          </div>
        ) : activeMeeting ? (
          <MeetingRoom
            key={activeMeeting.id}
            meeting={activeMeeting}
            onMeetingUpdated={refreshActive}
          />
        ) : (
          <div className="empty-state">
            <div style={{ fontSize: 48 }}>🗒️</div>
            <h2>Welcome to Smart Meeting</h2>
            <p>
              Create a new meeting to start live transcription, then summarize
              and translate the results.
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
