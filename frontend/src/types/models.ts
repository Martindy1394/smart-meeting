/**
 * Canonical Smart Meeting data shapes shared across API, WS, and UI.
 *
 * Wire/finalize historically used `start`/`end`; DB/REST use `start_time`/
 * `end_time`. Canonical payloads include both. Attendees are always `string[]`
 * on the wire; the DB stores JSON text via a TypeDecorator.
 */

/** Faithfulness soft-warn report from /ai/summarize (also persisted on Meeting). */
export interface FaithfulnessLine {
  section: string;
  line: string;
  overlap: number;
}

export interface FaithfulnessReport {
  status: "ok" | "warn" | "skipped" | string;
  untraced: FaithfulnessLine[];
  checked: number;
}

/**
 * Transcript fragment. Prefer `start_time`/`end_time` (DB/API).
 * `start`/`end` are wire aliases with identical values.
 */
export interface TranscriptSegment {
  id?: string;
  kind?: "live" | "final" | string;
  text: string;
  start_time: number;
  end_time: number;
  /** Wire/WS/finalize alias for start_time */
  start?: number;
  /** Wire/WS/finalize alias for end_time */
  end?: number;
  seq: number;
  /** Whisper avg log-probability for the segment (lower = less confident). */
  avg_logprob?: number | null;
  no_speech_prob?: number | null;
  /** Soft flag — UI may underline / badge low-confidence captions. */
  low_confidence?: boolean;
}

/** Structured action item from BART Action Items (or persisted JSON). */
export interface ActionItem {
  text: string;
  owner?: string;
  action?: string;
  due_date?: string;
}

export interface LanguageDetectionInfo {
  language?: string | null;
  confidence?: number | null;
  detected_by?: string;
}

/** Lightweight list/dashboard meeting. */
export interface MeetingSummary {
  id: string;
  title: string;
  status: string;
  language: string;
  language_detection?: LanguageDetectionInfo | null;
  venue?: string;
  meeting_date?: string | null;
  duration_seconds: number;
  created_at: string;
  updated_at: string;
  summary?: string;
  summary_format?: string;
  translation?: string;
  translation_language?: string;
  extractive_fallback?: boolean;
  has_summary?: boolean;
  has_translation?: boolean;
  has_audio?: boolean;
  has_transcript?: boolean;
}

/** Full meeting detail from GET /api/meetings/{id}. */
export interface MeetingDetail {
  id: string;
  title: string;
  status: string;
  language: string;
  language_detection?: LanguageDetectionInfo | null;
  venue: string;
  meeting_date?: string | null;
  /** Always a string array on the API (DB stores JSON text). */
  attendees: string[];
  final_transcript: string;
  summary: string;
  summary_format: string;
  translation: string;
  translation_language: string;
  /** Persisted — survives reload after summarize. */
  extractive_fallback: boolean;
  /** Persisted faithfulness report — survives reload. */
  faithfulness?: FaithfulnessReport | null;
  /** Proper nouns / terms for Whisper initial_prompt (JSON list or newlines). */
  custom_vocab?: string;
  /** Do-not-translate glossary JSON for NLLB/mBART. */
  translation_glossary_json?: string;
  /** Persisted action items JSON (or parsed array below). */
  action_items_json?: string;
  action_items?: ActionItem[];
  /** Session ASR language lock (detect once, reuse). */
  language_locked?: boolean;
  /** Transcript↔translation faithfulness (Tier 2). */
  translation_faithfulness?: FaithfulnessReport | null;
  duration_seconds: number;
  created_at: string;
  updated_at: string;
  has_audio: boolean;
  segments: TranscriptSegment[];
}

export interface MeetingCreate {
  title?: string;
  language?: string;
  venue?: string;
  meeting_date?: string | null;
  attendees?: string[];
  custom_vocab?: string;
  translation_glossary_json?: string;
  action_items_json?: string;
  action_items?: ActionItem[];
  language_locked?: boolean;
  translation_faithfulness?: FaithfulnessReport | null;
}

export interface MeetingUpdate {
  title?: string | null;
  venue?: string | null;
  meeting_date?: string | null;
  attendees?: string[] | null;
  language?: string | null;
  custom_vocab?: string;
  translation_glossary_json?: string;
  action_items_json?: string;
  action_items?: ActionItem[];
  language_locked?: boolean;
  translation_faithfulness?: FaithfulnessReport | null;
}

export interface SummarizeResponse {
  summary: string;
  output_format: string;
  engine: string;
  translation: string;
  translation_language: string;
  extractive_fallback: boolean;
  faithfulness?: FaithfulnessReport | null;
}

/** WS live_segment / final_transcript segment payload. */
export interface WireSegment {
  text: string;
  start: number;
  end: number;
  start_time: number;
  end_time: number;
  seq?: number;
  kind?: string;
  avg_logprob?: number | null;
  no_speech_prob?: number | null;
  low_confidence?: boolean;
}
