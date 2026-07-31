// Thin API client wrapping fetch with JWT auth + JSON handling.

const TOKEN_KEY = "sm_token";
const REFRESH_KEY = "sm_refresh";
const ACCESS_EXPIRES_AT_KEY = "sm_access_expires_at";

/** Same-origin in Vite dev (proxied). Optional absolute override via VITE_API_URL. */
function apiUrl(path) {
  const base = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
  const p = path.startsWith("/") ? path : `/${path}`;
  return base ? `${base}${p}` : p;
}

const NETWORK_HELP =
  "Network error — cannot reach the API. Open the app via Cursor → Ports → 5173 (Vite), keep the API on port 8000, then retry. If the backend just reloaded, wait a few seconds and try again.";

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

/** Flatten FastAPI ``detail`` (string | list | object) into one message. */
export function flattenApiDetail(detail) {
  if (detail == null || detail === "") return "";
  if (Array.isArray(detail)) {
    return detail.map((d) => d?.msg || JSON.stringify(d)).join("; ");
  }
  if (typeof detail === "object") {
    return detail.message || detail.detail || JSON.stringify(detail);
  }
  return String(detail);
}

/** True when the error is a decrypt / corrupt / audio-cache integrity failure. */
export function isDataIntegrityError(err) {
  const msg = String(err?.message || err || "");
  return /decrypt|corrupt|encryption key|audio cache failure|data corrupted/i.test(
    msg
  );
}

export function isNetworkError(err) {
  return err instanceof ApiError && err.status === 0;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Low-level fetch with retries for transient tunnel / uvicorn-reload failures.
 * Does not parse JSON — callers handle the Response.
 */
async function fetchWithRetry(url, init = {}, { retries = 3 } = {}) {
  let lastErr = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fetch(url, init);
    } catch (e) {
      lastErr = e;
      if (attempt >= retries) break;
      // 300ms, 800ms, 1600ms — covers brief --reload Whisper warmups.
      await sleep(300 * 2 ** attempt + 100);
    }
  }
  throw new ApiError(NETWORK_HELP, 0);
}

let refreshInFlight = null;

export async function refreshAccessToken() {
  const refresh = getRefreshToken();
  if (!refresh) return null;
  if (refreshInFlight) return refreshInFlight;
  refreshInFlight = (async () => {
    try {
      const res = await fetchWithRetry(
        apiUrl("/api/auth/refresh"),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refresh }),
        },
        { retries: 2 }
      );
      if (!res.ok) {
        // Only clear session on an actual auth rejection, not a proxy blip.
        if (res.status === 401 || res.status === 403) {
          clearSessionTokens();
        }
        return null;
      }
      const data = await res.json();
      setSessionTokens(data);
      return data.access_token || null;
    } catch {
      // Keep tokens on network failure — user can retry without re-login.
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

/** Quick connectivity probe (no auth). Returns true when /api/health is OK. */
export async function pingApi() {
  try {
    const res = await fetchWithRetry(apiUrl("/api/health"), { method: "GET" }, { retries: 1 });
    return res.ok;
  } catch {
    return false;
  }
}

async function request(path, { method = "GET", body, auth = true, _retry = true } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (auth) {
    const token = await ensureFreshAccessToken({ minValiditySeconds: 30 });
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetchWithRetry(apiUrl(`/api${path}`), {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

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
    // Vite often returns 500/502 with a proxy stack when uvicorn is mid-reload.
    if (
      (res.status === 502 || res.status === 503 || res.status === 504) ||
      (res.status === 500 && /ECONNREFUSED|proxy error|socket hang up/i.test(text || ""))
    ) {
      throw new ApiError(NETWORK_HELP, 0);
    }
    const detail = flattenApiDetail(data?.detail);
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
    const token = await ensureFreshAccessToken({ minValiditySeconds: 30 });
    const fmt = encodeURIComponent(format || "txt");
    const res = await fetchWithRetry(apiUrl(`/api/meetings/${id}/export?format=${fmt}`), {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
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
        detail = flattenApiDetail(data?.detail) || detail;
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
   * Returns null when no recording exists yet (404).
   * Throws ApiError on decrypt/cache/server failures (never silent empty).
   */
  getMeetingAudioUrl: async (id) => {
    const token = await ensureFreshAccessToken({ minValiditySeconds: 30 });
    const res = await fetchWithRetry(apiUrl(`/api/meetings/${id}/audio`), {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (res.status === 401 && getRefreshToken()) {
      const next = await refreshAccessToken();
      if (next) {
        return api.getMeetingAudioUrl(id);
      }
    }
    if (res.status === 404) return null;
    if (!res.ok) {
      let detail = "Could not load meeting audio.";
      try {
        const data = await res.json();
        detail = flattenApiDetail(data?.detail) || detail;
      } catch {
        /* ignore */
      }
      if (isDataIntegrityError({ message: detail })) {
        detail = detail.includes("Decryption failed")
          ? detail
          : `Decryption failed / Data corrupted. ${detail}`;
      }
      throw new ApiError(detail, res.status);
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
