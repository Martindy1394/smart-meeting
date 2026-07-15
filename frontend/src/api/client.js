// Thin API client wrapping fetch with JWT auth + JSON handling.

const TOKEN_KEY = "sm_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, { method = "GET", body, auth = true } = {}) {
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
      "Network error — cannot reach the API. Use http://127.0.0.1:8000/ and make sure port 8000 is Forwarded in Cursor → Ports.",
      0
    );
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
  /**
   * Upload a WAV/PCM file and run Whisper ASR on it (default).
   */
  uploadMeetingAudio: async (id, file, { transcribe = true } = {}) => {
    const token = getToken();
    const body = new FormData();
    body.append("file", file);
    const qs = transcribe ? "" : "?transcribe=false";
    let res;
    try {
      res = await fetch(`/api/meetings/${id}/audio${qs}`, {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body,
      });
    } catch {
      throw new ApiError(
        "Network error — cannot reach the API. Use http://127.0.0.1:8000/ and make sure port 8000 is Forwarded in Cursor → Ports.",
        0
      );
    }
    const text = await res.text();
    let data = null;
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
      throw new ApiError(detail || `Upload failed (${res.status})`, res.status);
    }
    return data;
  },
  /**
   * Fetch meeting audio as a blob URL for <audio> playback.
   * Returns null when no recording exists yet.
   */
  getMeetingAudioUrl: async (id) => {
    const token = getToken();
    const res = await fetch(`/api/meetings/${id}/audio`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (res.status === 404) return null;
    if (!res.ok) {
      throw new ApiError("Could not load meeting audio.", res.status);
    }
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  },
  downloadMeetingAudio: async (id, filename = "recording.wav") => {
    const token = getToken();
    const res = await fetch(`/api/meetings/${id}/audio?download=true`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!res.ok) {
      throw new ApiError("Could not download meeting audio.", res.status);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename.endsWith(".wav") ? filename : `${filename}.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  },

  // AI
  languages: () => request("/ai/languages"),
  summarize: (payload) => request("/ai/summarize", { method: "POST", body: payload }),
  translate: (payload) => request("/ai/translate", { method: "POST", body: payload }),

  health: () => request("/health", { auth: false }),
};

export { ApiError };
