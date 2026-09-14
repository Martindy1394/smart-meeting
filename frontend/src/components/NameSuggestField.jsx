import { useEffect, useId, useMemo, useRef, useState } from "react";
import { rankNameSuggestions } from "../lib/nameSuggestions.js";

export default function NameSuggestField({
  id,
  value,
  onChange,
  onSelect,
  directory,
  exclude = [],
  recents = [],
  session = [],
  role = "attendee",
  placeholder = "",
  disabled = false,
  onEnterWithoutHighlight,
}) {
  const listId = useId();
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const blurTimer = useRef(null);

  const items = useMemo(
    () =>
      rankNameSuggestions({
        query: value,
        directory,
        exclude,
        role,
        recents,
        session,
        limit: 8,
      }),
    [value, directory, exclude, role, recents, session]
  );

  useEffect(() => {
    setActive(0);
  }, [value, items.length]);

  function pick(row) {
    if (!row?.name) return;
    onSelect(row.name, row);
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
    if (e.key === "Enter") {
      if (open && items[active]) {
        e.preventDefault();
        pick(items[active]);
        return;
      }
      if (onEnterWithoutHighlight) {
        e.preventDefault();
        onEnterWithoutHighlight();
      }
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
          {items.map((row, i) => (
            <li
              id={`${listId}-${i}`}
              key={`${row.name}-${i}`}
              role="option"
              aria-selected={i === active}
              className={
                i === active ? "name-suggest-item is-active" : "name-suggest-item"
              }
              onMouseEnter={() => setActive(i)}
              onClick={() => pick(row)}
            >
              <span className="name-suggest-name">{row.name}</span>
              <span className="name-suggest-meta">
                {row.recent ? <span className="name-flag">Recent</span> : null}
                {row.sessionHit && !row.guest ? (
                  <span className="name-flag is-present">Identified</span>
                ) : null}
                {row.guest ? <span className="name-flag is-guest">Guest</span> : null}
                {row.sources?.includes("officer") ? (
                  <span className="name-flag">Officer</span>
                ) : null}
                {role === "officer" && row.officer_count > 1 ? (
                  <span className="name-flag">{row.officer_count}× chaired</span>
                ) : null}
                {role === "attendee" && row.attendee_count > 1 ? (
                  <span className="name-flag">{row.attendee_count}× attended</span>
                ) : null}
                {row.last_title ? (
                  <span className="name-suggest-title">{row.last_title}</span>
                ) : null}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
