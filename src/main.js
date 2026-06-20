// Dataset Reviewer — static entry point. Boots straight into the dataset board
// (newest-first browse that becomes a semantic search the moment you type).
import "./style.css";
import { mountApp } from "./app.js";

// Theme toggle: no data-theme attribute = follow the system; clicking pins a
// choice. The pin is written to a cookie on .misclaw.app so the light/dark
// choice stays in sync across every *.misclaw.app site (localStorage is the
// same-origin fallback). The pre-paint reader in index.html applies it.
function setTheme(next) {
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch {}
  try {
    let c = "mc-theme=" + next + ";path=/;max-age=31536000;samesite=lax";
    if (location.hostname === "misclaw.app" || location.hostname.endsWith(".misclaw.app")) {
      c += ";domain=.misclaw.app";
    }
    if (location.protocol === "https:") c += ";secure";
    document.cookie = c;
  } catch {}
}

document.getElementById("theme-toggle")?.addEventListener("click", () => {
  const current = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  setTheme(current === "dark" ? "light" : "dark");
});

mountApp(document.getElementById("app"));
