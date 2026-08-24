import { useEffect, useId, useRef } from "react";

/**
 * Modal confirmation dialog for destructive / important actions.
 *
 * Props:
 *  open, title, message, confirmLabel, cancelLabel, tone ("danger"|"default"),
 *  busy, onConfirm, onCancel
 */
export default function ConfirmDialog({
  open,
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  tone = "default",
  busy = false,
  onConfirm,
  onCancel,
}) {
  const titleId = useId();
  const descId = useId();
  const confirmRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const prev = document.activeElement;
    const t = setTimeout(() => confirmRef.current?.focus(), 0);
    const onKey = (e) => {
      if (e.key === "Escape" && !busy) onCancel?.();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      clearTimeout(t);
      window.removeEventListener("keydown", onKey);
      if (prev && typeof prev.focus === "function") prev.focus();
    };
  }, [open, busy, onCancel]);

  if (!open) return null;

  return (
    <div
      className="confirm-overlay"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !busy) onCancel?.();
      }}
    >
      <div
        className={`confirm-dialog ${tone === "danger" ? "danger" : ""}`}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descId}
      >
        <h3 id={titleId} className="confirm-title">
          {title}
        </h3>
        <p id={descId} className="confirm-message">
          {message}
        </p>
        <div className="confirm-actions">
          <button
            type="button"
            className="btn secondary"
            onClick={onCancel}
            disabled={busy}
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            type="button"
            className={`btn ${tone === "danger" ? "danger" : ""}`}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? <span className="spinner" /> : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
