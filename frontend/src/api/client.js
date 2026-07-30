// Thin API client wrapping fetch with JWT auth + JSON handling.

const TOKEN_KEY = "sm_token";
const REFRESH_KEY = "sm_refresh";
const ACCESS_EXPIRES_AT_KEY = "sm_access_expires_at";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function getRefreshToken() {
  return localStorage.getItem(REFRESH_KEY);
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

export function setRefreshToken(token) {
  if (token) localStorage.setItem(REFRESH_KEY, token);
  else localStorage.removeItem(REFRESH_KEY);
}

/** Seconds until JWT ``exp`` (0 if missing/unreadable). */
export function accessTokenSecondsRemaining(token = getToken()) {
  if (!token) return 0;
  try {
    const payload = JSON.parse(atob(token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")));
    const exp = Number(payload.exp) || 0;
    return Math.max(0, exp - Math.floor(Date.now() / 1000));
  } catch {
    const stored = Number(localStorage.getItem(ACCESS_EXPIRES_AT_KEY) || 0);
    if (stored > 0) return Math.max(0, Math.floor((stored - Date.now()) / 1000));
    return 0;
  }
}

export function setSessionTokens({ access_token, refresh_token, expires_in } = {}) {
  setToken(access_token || null);
  setRefreshToken(refresh_token || null);
  if (access_token && Number(expires_in) > 0) {
    localStorage.setItem(
      ACCESS_EXPIRES_AT_KEY,
      String(Date.now() + Number(expires_in) * 1000)
    );
  } else if (access_token) {
    const secs = accessTokenSecondsRemaining(access_token);
    if (secs > 0) {
      localStorage.setItem(ACCESS_EXPIRES_AT_KEY, String(Date.now() + secs * 1000));
    }
  } else {
    localStorage.removeItem(ACCESS_EXPIRES_AT_KEY);
  }
}

export function clearSessionTokens() {
  setToken(null);
  setRefreshToken(null);
  localStorage.removeItem(ACCESS_EXPIRES_AT_KEY);
}

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

let refreshInFlight = null;

export async function refreshAccessToken() {
  const refresh = getRefreshToken();
  if (!refresh) return null;
  if (refreshInFlight) return refreshInFlight;
  refreshInFlight = (async () => {
    try {
      const res = await fetch("/api/auth/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refresh }),
      });
      if (!res.ok) {
        clearSessionTokens();
        return null;
      }
      const data = await res.json();
      setSessionTokens(data);
      return data.access_token || null;
    } catch {
      clearSessionTokens();
      return null;
    } finally {
      refreshInFlight = null;
    }
  })();
  return refreshInFlight;
}

/**
 * Return a usable access token, refreshing proactively when near expiry.
 * Used by REST retries and long-lived WebSocket reconnects.
 */
export async function ensureFreshAccessToken({ minValiditySeconds = 120 } = {}) {
  const current = getToken();
  const remaining = accessTokenSecondsRemaining(current);
  if (current && remaining > minValiditySeconds) {
    return current;
  }
  if (!getRefreshToken()) {
    return current;
  }
  const next = await refreshAccessToken();
  return next || getToken();
}

async function request(path, { method = "GET", body, auth = true, _retry = true } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (auth) {
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }
  let res;
  try {
    res = await fetch(`/api${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new ApiError(
      "Network error — cannot reach the API. Open the app via the forwarded port (5173 or 8000) in Cursor → Ports, and keep the backend running on http://127.0.0.1:8000/.",
      0
    );
  }

  if (res.status === 401 && auth && _retry && getRefreshToken()) {
    const next = await refreshAccessToken();
    if (next) {
      return request(path, { method, body, auth, _retry: false });
    }
  }

  if (res.status === 204) return null;

  let data = null;
  const text = await res.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }

  if (!res.ok) {
    let detail = data?.detail;
    if (Array.isArray(detail)) {
      detail = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }
  return data;
}

export const api = {
  // Auth
  signup: (payload) => request("/auth/signup", { method: "POST", body: payload, auth: false }),
  login: (payload) => request("/auth/login", { method: "POST", body: payload, auth: false }),
  refresh: (refresh_token) =>
    request("/auth/refresh", {
      method: "POST",
      body: { refresh_token },
      auth: false,
    }),
  logout: async () => {
    const access = getToken();
    const refresh = getRefreshToken();
    try {
      await request("/auth/logout", {
        method: "POST",
        body: { access_token: access, refresh_token: refresh },
      });
    } catch {
      /* still clear local session */
    }
    clearSessionTokens();
  },
  me: () => request("/auth/me"),
  updateProfile: (payload) => request("/auth/me", { method: "PATCH", body: payload }),

  // Meetings
  listMeetings: (search, { hasAudio } = {}) => {
    const params = new URLSearchParams();
    if (search) params.set("search", search);
    if (hasAudio === true) params.set("has_audio", "true");
    if (hasAudio === false) params.set("has_audio", "false");
    const qs = params.toString();
    return request(`/meetings${qs ? `?${qs}` : ""}`);
  },
  createMeeting: (payload) => request("/meetings", { method: "POST", body: payload }),
  getMeeting: (id) => request(`/meetings/${id}`),
  updateMeeting: (id, payload) => request(`/meetings/${id}`, { method: "PATCH", body: payload }),
  deleteMeeting: (id) => request(`/meetings/${id}`, { method: "DELETE" }),
  retranscribeMeeting: (id) =>
    request(`/meetings/${id}/retranscribe`, { method: "POST" }),
  /** Finalize a live recording even if the WebSocket stop message was lost. */
  stopMeetingRecording: (id) =>
    request(`/meetings/${id}/stop`, { method: "POST" }),
  /**
   * Download meeting package (transcript + English + summary) as txt|docx|pdf.
   */
  exportMeeting: async (id, format = "txt") => {
    const token = getToken();
    const fmt = encodeURIComponent(format || "txt");
    let res;
    try {
      res = await fetch(`/api/meetings/${id}/export?format=${fmt}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
    } catch {
      throw new ApiError(
        "Network error — cannot reach the API. Open the app via the forwarded port (5173 or 8000) in Cursor → Ports, and keep the backend running on http://127.0.0.1:8000/.",
        0
      );
    }
    if (res.status === 401 && getRefreshToken()) {
      const next = await refreshAccessToken();
      if (next) {
        return api.exportMeeting(id, format);
      }
    }
    if (!res.ok) {
      let detail = "Could not export meeting.";
      try {
        const data = await res.json();
        if (data?.detail) detail = data.detail;
      } catch {
        /* ignore */
      }
      throw new ApiError(detail, res.status);
    }
    const blob = await res.blob();
    const disposition = res.headers.get("Content-Disposition") || "";
    const match = /filename=\"([^\"]+)\"/i.exec(disposition);
    const filename = match?.[1] || `smart_meeting.${format || "txt"}`;
    return { blob, filename };
  },
  /**
   * Fetch meeting audio as a blob URL for <audio> playback.
   * Returns null when no recording exists yet.
   */
  getMeetingAudioUrl: async (id) => {
    const token = getToken();
    let res;
    try {
      res = await fetch(`/api/meetings/${id}/audio`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
    } catch {
      throw new ApiError(
        "Network error — cannot reach the API. Open the app via the forwarded port (5173 or 8000) in Cursor → Ports, and keep the backend running on http://127.0.0.1:8000/.",
        0
      );
    }
    if (res.status === 401 && getRefreshToken()) {
      const next = await refreshAccessToken();
      if (next) {
        return api.getMeetingAudioUrl(id);
      }
    }
    if (res.status === 404) return null;
    if (!res.ok) {
      throw new ApiError("Could not load meeting audio.", res.status);
    }
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  },

  // AI
  languages: () => request("/ai/languages"),
  summarize: (payload) => request("/ai/summarize", { method: "POST", body: payload }),
  translate: (payload) => request("/ai/translate", { method: "POST", body: payload }),

  health: () => request("/health", { auth: false }),
};

export { ApiError };
