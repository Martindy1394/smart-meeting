import { useEffect, useId, useMemo, useRef, useState } from "react";

/** Combobox of past meeting titles or venues (history), filtered as you type. */
export default function HistorySuggestField({
  id,
  value,
  onChange,
  options = [],
  placeholder = "",
  disabled = false,
}) {
  const listId = useId();
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const blurTimer = useRef(null);

  const items = useMemo(() => {
    const q = String(value || "").trim().toLowerCase();
    const seen = new Set();
    const out = [];
    for (const raw of options) {
      const name = String(raw || "").trim();
      if (!name) continue;
      const key = name.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      if (q && !key.includes(q)) continue;
      out.push(name);
      if (out.length >= 8) break;
    }
    return out;
  }, [value, options]);

  useEffect(() => {
    setActive(0);
  }, [value, items.length]);

  function pick(name) {
    if (!name) return;
    onChange(name);
    setOpen(false);
  }

  function onKeyDown(e) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.min(items.length - 1, i + 1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.max(0, i - 1));
      return;
    }
    if (e.key === "Escape") {
      setOpen(false);
      return;
    }
    if (e.key === "Enter" && open && items[active]) {
      e.preventDefault();
      pick(items[active]);
    }
  }

  const show = open && items.length > 0;

  return (
    <div className="name-combobox">
      <input
        id={id}
        type="text"
        role="combobox"
        aria-expanded={show}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={show && items[active] ? `${listId}-${active}` : undefined}
        placeholder={placeholder}
        value={value}
        disabled={disabled}
        autoComplete="off"
        onChange={(e) => {
          onChange(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          blurTimer.current = setTimeout(() => setOpen(false), 120);
        }}
        onKeyDown={onKeyDown}
      />
      {show ? (
        <ul
          id={listId}
          className="name-suggest-list"
          role="listbox"
          onMouseDown={(e) => e.preventDefault()}
        >
          {items.map((name, i) => (
            <li
              id={`${listId}-${i}`}
              key={`${name}-${i}`}
              role="option"
              aria-selected={i === active}
              className={
                i === active ? "name-suggest-item is-active" : "name-suggest-item"
              }
              onMouseEnter={() => setActive(i)}
              onClick={() => pick(name)}
            >
              <span className="name-suggest-name">{name}</span>
              <span className="name-suggest-meta">
                <span className="name-flag">History</span>
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
