import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

const PASSWORD_RE = /^(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$/;

export default function Signup() {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const passwordValid = PASSWORD_RE.test(password);

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    if (!passwordValid) {
      setError(
        "Password must be at least 8 characters and include a number and a special character."
      );
      return;
    }
    setBusy(true);
    try {
      await signup(email.trim(), password, fullName.trim());
      navigate("/", { replace: true });
    } catch (err) {
      setError(err.message || "Signup failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-wrap">
      <form className="auth-card" onSubmit={onSubmit}>
        <div className="brand">
          <span className="dot" /> Smart Meeting
        </div>
        <p className="auth-sub">Create your account to start capturing minutes.</p>

        {error && <div className="error-banner">{error}</div>}

        <div className="field">
          <label htmlFor="name">Full name</label>
          <input
            id="name"
            type="text"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
          />
        </div>
        <div className="field">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <div
            className="hint"
            style={{ color: password && !passwordValid ? "#fca5a5" : undefined }}
          >
            Min 8 characters, at least one number and one special character.
          </div>
        </div>

        <button className="btn" style={{ width: "100%" }} disabled={busy}>
          {busy ? <span className="spinner" /> : "Create account"}
        </button>

        <div className="auth-footer">
          Already have an account? <Link to="/login">Sign in</Link>
        </div>
      </form>
    </div>
  );
}
