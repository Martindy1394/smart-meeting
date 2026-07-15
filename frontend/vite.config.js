import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api and /ws to the FastAPI backend during development so the frontend
// can use same-origin relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
        // Do not idle-out long live-transcription sessions in dev.
        timeout: 0,
        proxyTimeout: 0,
      },
    },
  },
});
