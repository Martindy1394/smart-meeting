/**
 * Rank name suggestions for attendee / presiding-officer comboboxes.
 * No calendar or org-directory APIs — those stay integration stubs.
 */

const RECENT_KEY = "smart-meeting:recent-names";
const MAX_RECENT = 12;

const NICKNAMES = {
  maria: ["mary", "marie", "marya"],
  mary: ["maria", "marie"],
  juan: ["john", "johnny"],
  john: ["juan", "johnny"],
  jose: ["joseph", "jo"],
  ana: ["anna", "anne"],
  anna: ["ana", "anne"],
  pedro: ["peter"],
  peter: ["pedro"],
  cristina: ["christina", "tina"],
  michael: ["mike"],
  mike: ["michael"],
  robert: ["rob", "bob"],
  bob: ["robert", "rob"],
};

export function soundex(raw) {
  const word = String(raw || "")
    .toUpperCase()
    .replace(/[^A-Z]/g, "");
  if (!word) return "";
  const first = word[0];
  const map = {
    B: "1",
    F: "1",
    P: "1",
    V: "1",
    C: "2",
    G: "2",
    J: "2",
    K: "2",
    Q: "2",
    S: "2",
    X: "2",
    Z: "2",
    D: "3",
    T: "3",
    L: "4",
    M: "5",
    N: "5",
    R: "6",
  };
  let out = first;
  let prev = map[first] || "";
  for (let i = 1; i < word.length && out.length < 4; i += 1) {
    const code = map[word[i]] || "";
    if (code && code !== prev) out += code;
    prev = code;
  }
  return (out + "000").slice(0, 4);
}

function nickSet(token) {
  const t = token.toLowerCase();
  const extra = NICKNAMES[t] || [];
  return new Set([t, ...extra]);
}

export function loadRecentNames() {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return {
      officers: Array.isArray(parsed.officers) ? parsed.officers : [],
      attendees: Array.isArray(parsed.attendees) ? parsed.attendees : [],
    };
  } catch {
    return { officers: [], attendees: [] };
  }
}

export function rememberRecentName(role, name) {
  const cleaned = String(name || "").trim();
  if (!cleaned || typeof localStorage === "undefined") return;
  const store = loadRecentNames();
  const key = role === "officer" ? "officers" : "attendees";
  const next = [cleaned, ...store[key].filter((n) => n.toLowerCase() !== cleaned.toLowerCase())].slice(
    0,
    MAX_RECENT
  );
  try {
    localStorage.setItem(
      RECENT_KEY,
      JSON.stringify({ ...store, [key]: next })
    );
  } catch {
    /* quota / private mode */
  }
}

export function sessionPeople(meeting) {
  const att = meeting?.attendance;
  const out = [];
  if (!att || typeof att !== "object") return out;
  for (const row of att.present || []) {
    const name = String(row?.name || "").trim();
    if (name) {
      out.push({
        name,
        source: "identified",
        confidence: Number(row.confidence) || 0,
        guest: false,
      });
    }
  }
  for (const row of att.guests || []) {
    const name = String(row?.name || row || "").trim();
    if (name) {
      out.push({
        name,
        source: "guest",
        confidence: Number(row?.confidence) || 0,
        guest: true,
      });
    }
  }
  return out;
}

function asPeople(directory) {
  if (Array.isArray(directory?.people) && directory.people.length) {
    return directory.people.map((p) => ({
      name: String(p.name || "").trim(),
      officer_count: Number(p.officer_count) || 0,
      attendee_count: Number(p.attendee_count) || 0,
      identified_count: Number(p.identified_count) || 0,
      last_title: p.last_title || "",
      last_seen: p.last_seen || null,
      sources: Array.isArray(p.sources) ? p.sources : [],
    })).filter((p) => p.name);
  }
  const names = [
    ...(directory?.presiding_officers || []).map((n) => ({
      name: n,
      officer_count: 1,
      attendee_count: 0,
      sources: ["officer"],
    })),
    ...(directory?.attendees || []).map((n) => ({
      name: n,
      officer_count: 0,
      attendee_count: 1,
      sources: ["attendee"],
    })),
  ];
  const byKey = new Map();
  for (const row of names) {
    const key = row.name.toLowerCase();
    const prev = byKey.get(key);
    if (!prev) byKey.set(key, { identified_count: 0, last_title: "", ...row });
    else {
      prev.officer_count += row.officer_count;
      prev.attendee_count += row.attendee_count;
    }
  }
  return [...byKey.values()];
}

function queryScore(query, name) {
  const q = query.trim().toLowerCase();
  const n = name.toLowerCase();
  if (!q) return 1;
  if (n === q) return 100;
  if (n.startsWith(q)) return 80;
  if (n.includes(q)) return 55;
  const qTokens = q.split(/\s+/).filter(Boolean);
  const nTokens = n.split(/\s+/).filter(Boolean);
  let nick = 0;
  for (const qt of qTokens) {
    const nicks = nickSet(qt);
    if (nTokens.some((nt) => nicks.has(nt) || nt.startsWith(qt))) nick += 40;
  }
  if (nick) return nick;
  const qLast = qTokens[qTokens.length - 1] || "";
  const nLast = nTokens[nTokens.length - 1] || "";
  if (qLast.length > 2 && nLast.startsWith(qLast)) return 45;
  if (soundex(qLast) && soundex(qLast) === soundex(nLast) && qLast.length > 2) return 35;
  return 0;
}

/**
 * Rank directory + session + recent names for a combobox.
 */
export function rankNameSuggestions({
  query = "",
  directory = {},
  exclude = [],
  role = "attendee",
  recents = [],
  session = [],
  limit = 8,
} = {}) {
  const skip = new Set((exclude || []).map((n) => String(n).toLowerCase()));
  const q = String(query || "").trim();
  const people = asPeople(directory);
  const byKey = new Map();
  const take = (name, extra) => {
    const cleaned = String(name || "").trim();
    if (!cleaned) return;
    const key = cleaned.toLowerCase();
    if (skip.has(key)) return;
    const prev = byKey.get(key) || {
      name: cleaned,
      officer_count: 0,
      attendee_count: 0,
      identified_count: 0,
      last_title: "",
      sources: [],
      recent: false,
      sessionHit: false,
      guest: false,
      confidence: 0,
    };
    const sources = [...new Set([...(prev.sources || []), ...(extra.sources || [])])];
    byKey.set(key, {
      ...prev,
      ...extra,
      name: prev.name || cleaned,
      sources,
      recent: Boolean(prev.recent || extra.recent),
      sessionHit: Boolean(prev.sessionHit || extra.sessionHit),
      guest: Boolean(prev.guest || extra.guest),
      confidence: Math.max(prev.confidence || 0, extra.confidence || 0),
    });
  };

  for (const p of people) take(p.name, p);
  for (const name of recents || []) take(name, { recent: true, sources: ["recent"] });
  for (const hit of session || []) {
    take(hit.name, {
      sessionHit: true,
      guest: Boolean(hit.guest),
      confidence: hit.confidence || 0,
      sources: [hit.guest ? "guest" : "identified"],
      identified_count: 1,
    });
  }

  const ranked = [];
  for (const row of byKey.values()) {
    const match = queryScore(q, row.name);
    if (q && match <= 0) continue;
    let score = match;
    if (row.recent) score += 25;
    if (row.sessionHit) score += 30 + Math.round((row.confidence || 0) * 10);
    if (role === "officer") score += Math.min(20, row.officer_count * 4);
    else score += Math.min(16, row.attendee_count * 3);
    if (row.identified_count) score += 8;
    ranked.push({ ...row, score });
  }
  ranked.sort((a, b) => b.score - a.score || a.name.localeCompare(b.name));
  return ranked.slice(0, Math.max(1, limit));
}
