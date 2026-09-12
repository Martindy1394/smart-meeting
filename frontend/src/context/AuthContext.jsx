import { createContext, useCallback, useContext, useEffect, useState } from "react";
import {
  api,
  clearSessionTokens,
  getToken,
  setSessionTokens,
} from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const AUTH_BOOTSTRAP_MS = 12_000;
    async function bootstrap() {
      if (!getToken()) {
        setLoading(false);
        return;
      }
      try {
        const me = await Promise.race([
          api.me(),
          new Promise((_, reject) => {
            setTimeout(
              () => reject(new Error("Timed out checking the signed-in session.")),
              AUTH_BOOTSTRAP_MS
            );
          }),
        ]);
        if (!cancelled) setUser(me);
      } catch (err) {
        console.error("Auth bootstrap failed", err);
        // Keep the session on network blips; only drop tokens on auth rejection.
        if (err?.status === 401 || err?.status === 403) {
          clearSessionTokens();
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    bootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (username, password) => {
    const res = await api.login({ username, password });
    setSessionTokens(res);
    setUser(res.user);
    return res.user;
  }, []);

  const signup = useCallback(async (payload) => {
    const res = await api.signup(payload);
    setSessionTokens(res);
    setUser(res.user);
    return res.user;
  }, []);

  const updateProfile = useCallback(async (payload) => {
    const user = await api.updateProfile(payload);
    setUser(user);
    return user;
  }, []);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      clearSessionTokens();
    }
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{ user, loading, login, signup, updateProfile, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
