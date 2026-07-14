import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

const PASSWORD_RE = /^(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$/;
const USERNAME_RE = /^[A-Za-z0-9._-]{3,64}$/;

const EMPTY = {
  first_name: "",
  last_name: "",
  position: "",
  workplace: "",
  email: "",
  username: "",
  password: "",
};

export default function Signup() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState(EMPTY);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const passwordValid = PASSWORD_RE.test(form.password);
  const usernameValid = USERNAME_RE.test(form.username.trim());

  function setField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    if (!usernameValid) {
      setError(
        "Username must be 3–64 characters (letters, numbers, dots, underscores, hyphens)."
      );
      return;
    }
    if (!passwordValid) {
      setError(
        "Password must be at least 8 characters and include a number and a special character."
      );
      return;
    }
    setBusy(true);
    try {
      await signup({
        first_name: form.first_name.trim(),
        last_name: form.last_name.trim(),
        position: form.position.trim(),
        workplace: form.workplace.trim(),
        email: form.email.trim(),
        username: form.username.trim(),
        password: form.password,
      });
      navigate("/", { replace: true });
    } catch (err) {
      setError(err.message || "Signup failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-wrap">
      <form className="auth-card auth-card-wide" onSubmit={onSubmit}>
        <div className="brand">
          <span className="dot" /> Smart Meeting
        </div>
        <p className="auth-sub">Create your account to start capturing minutes.</p>

        {error && <div className="error-banner">{error}</div>}

        <div className="auth-grid">
          <div className="field">
            <label htmlFor="first_name">First name</label>
            <input
              id="first_name"
              type="text"
              value={form.first_name}
              onChange={(e) => setField("first_name", e.target.value)}
              required
            />
          </div>
          <div className="field">
            <label htmlFor="last_name">Last name</label>
            <input
              id="last_name"
              type="text"
              value={form.last_name}
              onChange={(e) => setField("last_name", e.target.value)}
              required
            />
          </div>
          <div className="field">
            <label htmlFor="position">Position</label>
            <input
              id="position"
              type="text"
              value={form.position}
              onChange={(e) => setField("position", e.target.value)}
              required
            />
          </div>
          <div className="field">
            <label htmlFor="workplace">Workplace</label>
            <input
              id="workplace"
              type="text"
              value={form.workplace}
              onChange={(e) => setField("workplace", e.target.value)}
              required
            />
          </div>
          <div className="field auth-grid-full">
            <label htmlFor="email">Working email address</label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              value={form.email}
              onChange={(e) => setField("email", e.target.value)}
              required
            />
          </div>
          <div className="field">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              autoComplete="username"
              value={form.username}
              onChange={(e) => setField("username", e.target.value)}
              required
            />
            <div className="hint">Used together with your password to sign in.</div>
          </div>
          <div className="field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              autoComplete="new-password"
              value={form.password}
              onChange={(e) => setField("password", e.target.value)}
              required
            />
            <div
              className="hint"
              style={{ color: form.password && !passwordValid ? "#fca5a5" : undefined }}
            >
              Min 8 characters, at least one number and one special character.
            </div>
          </div>
        </div>

        <button className="btn" style={{ width: "100%", marginTop: 8 }} disabled={busy}>
          {busy ? <span className="spinner" /> : "Create account"}
        </button>

        <div className="auth-footer">
          Already have an account? <Link to="/login">Sign in</Link>
        </div>
      </form>
    </div>
  );
}
