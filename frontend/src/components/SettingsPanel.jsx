import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext.jsx";

const PASSWORD_RE = /^(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$/;
const USERNAME_RE = /^[A-Za-z0-9._-]{3,64}$/;

export default function SettingsPanel() {
  const { user, updateProfile } = useAuth();
  const [form, setForm] = useState({
    first_name: "",
    last_name: "",
    position: "",
    workplace: "",
    email: "",
    username: "",
    password: "",
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!user) return;
    setForm({
      first_name: user.first_name || "",
      last_name: user.last_name || "",
      position: user.position || "",
      workplace: user.workplace || "",
      email: user.email || "",
      username: user.username || "",
      password: "",
    });
  }, [user]);

  function setField(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
    setSuccess("");
  }

  async function onSubmit(e) {
    e.preventDefault();
    setError("");
    setSuccess("");
    if (!USERNAME_RE.test(form.username.trim())) {
      setError(
        "Username must be 3–64 characters (letters, numbers, dots, underscores, hyphens)."
      );
      return;
    }
    if (form.password && !PASSWORD_RE.test(form.password)) {
      setError(
        "Password must be at least 8 characters and include a number and a special character."
      );
      return;
    }
    setBusy(true);
    try {
      const payload = {
        first_name: form.first_name.trim(),
        last_name: form.last_name.trim(),
        position: form.position.trim(),
        workplace: form.workplace.trim(),
        email: form.email.trim(),
        username: form.username.trim(),
      };
      if (form.password) payload.password = form.password;
      await updateProfile(payload);
      setForm((prev) => ({ ...prev, password: "" }));
      setSuccess("Profile saved.");
    } catch (err) {
      setError(err.message || "Could not save profile.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="content settings-panel">
      <div className="card settings-card">
        <div className="card-head">
          <h3>Account settings</h3>
          <span className="card-tag">Profile</span>
        </div>
        <div className="card-body">
          <p className="settings-intro">
            Personal information for the account owner. Username and password are
            used to sign in.
          </p>

          {error && <div className="error-banner">{error}</div>}
          {success && <div className="success-banner">{success}</div>}

          <form className="settings-form" onSubmit={onSubmit}>
            <div className="auth-grid">
              <div className="field">
                <label htmlFor="settings-first">First name</label>
                <input
                  id="settings-first"
                  value={form.first_name}
                  onChange={(e) => setField("first_name", e.target.value)}
                  required
                />
              </div>
              <div className="field">
                <label htmlFor="settings-last">Last name</label>
                <input
                  id="settings-last"
                  value={form.last_name}
                  onChange={(e) => setField("last_name", e.target.value)}
                  required
                />
              </div>
              <div className="field">
                <label htmlFor="settings-position">Position</label>
                <input
                  id="settings-position"
                  value={form.position}
                  onChange={(e) => setField("position", e.target.value)}
                  required
                />
              </div>
              <div className="field">
                <label htmlFor="settings-workplace">Workplace</label>
                <input
                  id="settings-workplace"
                  value={form.workplace}
                  onChange={(e) => setField("workplace", e.target.value)}
                  required
                />
              </div>
              <div className="field auth-grid-full">
                <label htmlFor="settings-email">Working email address</label>
                <input
                  id="settings-email"
                  type="email"
                  value={form.email}
                  onChange={(e) => setField("email", e.target.value)}
                  required
                />
              </div>
              <div className="field">
                <label htmlFor="settings-username">Username</label>
                <input
                  id="settings-username"
                  value={form.username}
                  onChange={(e) => setField("username", e.target.value)}
                  required
                />
              </div>
              <div className="field">
                <label htmlFor="settings-password">New password</label>
                <input
                  id="settings-password"
                  type="password"
                  autoComplete="new-password"
                  value={form.password}
                  onChange={(e) => setField("password", e.target.value)}
                  placeholder="Leave blank to keep current"
                />
              </div>
            </div>

            <div className="settings-actions">
              <button className="btn" type="submit" disabled={busy}>
                {busy ? <span className="spinner" /> : "Save profile"}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
