// Web server: serves the page in public/ and forwards /api requests
// to the Python backend (backend/app.py).

import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";

const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

const app = express();

app.use(
  createProxyMiddleware({
    target: BACKEND_URL,
    pathFilter: "/api",
    proxyTimeout: 30 * 60 * 1000, // long recordings can take minutes on CPU
  }),
);

app.use(express.static("public"));

app.listen(PORT, () => {
  console.log(`Open http://localhost:${PORT}  (backend: ${BACKEND_URL})`);
});
