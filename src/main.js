// Dataset Reviewer — static entry point. Boots straight into the dataset board
// (newest-first browse that becomes a semantic search the moment you type).
import "./style.css";
import { mountApp } from "./app.js";

// Theme toggle: no data-theme attribute = follow the system; clicking pins a
// choice (read before paint in index.html).
document.getElementById("theme-toggle")?.addEventListener("click", () => {
  const current = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch {}
});

mountApp(document.getElementById("app"));
