/** Normalize a meeting payload so the UI never reads null attendees/segments. */
export function normalizeMeeting(detail) {
  if (!detail || typeof detail !== "object") return null;
  const attendees = Array.isArray(detail.attendees)
    ? detail.attendees.map((n) => String(n || "").trim()).filter(Boolean)
    : typeof detail.attendees === "string"
      ? detail.attendees
          .split(/[,;\n]+/)
          .map((n) => n.trim())
          .filter(Boolean)
      : [];
  const segments = Array.isArray(detail.segments) ? detail.segments : [];
  return {
    ...detail,
    id: detail.id,
    title: detail.title ?? "",
    venue: detail.venue ?? "",
    presiding_officer: detail.presiding_officer ?? "",
    attendees,
    segments,
    final_transcript: detail.final_transcript ?? "",
    summary: detail.summary ?? "",
    summary_format: detail.summary_format ?? "bullets",
    translation: detail.translation ?? "",
    translation_language: detail.translation_language ?? "",
    status: detail.status ?? "recording",
    attendance: detail.attendance && typeof detail.attendance === "object" ? detail.attendance : null,
  };
}

export function normalizeMeetingList(list) {
  if (!Array.isArray(list)) return [];
  return list.map((row) => normalizeMeeting(row)).filter((row) => row?.id);
}
