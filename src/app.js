// Dataset Reviewer — dataset-first board. Browse newest dataset/benchmark papers
// by default; the moment you type, the list re-ranks by meaning (client-side
// bge-small). Facets (modality / venue / year / language / license / task)
// filter the list and show only when the corpus actually has the values.
import { loadIndex, browse, search, facetOptions, stats, isLoaded } from "./search.js";

const FACET_LABELS = {
  modality: "Modality", task: "Task", language: "Language",
  license: "License", venue: "Venue", year: "Year",
};
const FACET_ORDER = ["modality", "task", "language", "license", "venue", "year"];

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(kid));
  return n;
}

const state = {
  query: "",
  filters: { modality: new Set(), task: new Set(), language: new Set(), license: new Set(), venue: new Set(), year: new Set(), minConf: null },
  busy: false,
  expansion: [],
  lastTotal: 0,
};

let refs = {};

export function mountApp(root) {
  refs.root = root;
  root.innerHTML = "";

  refs.input = el("input", {
    type: "search", id: "q", class: "q",
    placeholder: "Search datasets & benchmarks — e.g. multilingual toxicity, speech emotion, tabular fraud…",
    autocomplete: "off", spellcheck: "false",
  });
  refs.input.addEventListener("input", onInput);

  refs.confToggle = el("label", { class: "conf-toggle hidden" },
    el("input", { type: "checkbox", onchange: onConfToggle }),
    el("span", {}, "High-confidence only"));

  refs.facets = el("div", { class: "facets" });
  refs.summary = el("div", { class: "summary" });
  refs.list = el("div", { class: "list" });

  root.append(
    el("div", { class: "searchbar" }, refs.input),
    refs.confToggle,
    refs.facets,
    refs.summary,
    refs.list,
  );

  refs.summary.textContent = "Loading the dataset index…";
  loadIndex((step) => { if (!isLoaded()) refs.summary.textContent = step; })
    .then(() => { renderFacets(); rerun(); })
    .catch((e) => { refs.summary.textContent = "Failed to load index: " + e.message; });
}

let inputTimer = null;
function onInput(e) {
  state.query = e.target.value.trim();
  clearTimeout(inputTimer);
  inputTimer = setTimeout(rerun, 220);
}
function onConfToggle(e) {
  state.filters.minConf = e.target.checked ? 0.5 : null;
  rerun();
}

function toggleFacet(key, value) {
  const set = state.filters[key];
  if (set.has(value)) set.delete(value); else set.add(value);
  renderFacets();
  rerun();
}

function renderFacets() {
  const opts = facetOptions(state.filters);
  refs.facets.innerHTML = "";
  for (const key of FACET_ORDER) {
    const values = opts[key];
    if (!values) continue;
    const chips = values.slice(0, 14).map(([val, n]) => {
      const on = state.filters[key].has(val);
      return el("button", { class: "chip" + (on ? " on" : ""), onclick: () => toggleFacet(key, val) },
        `${val}`, el("span", { class: "n" }, String(n)));
    });
    refs.facets.append(el("div", { class: "facet" },
      el("span", { class: "facet-label" }, FACET_LABELS[key]), ...chips));
  }
  // confidence toggle visible only once classifier-scored papers exist
  const s = stats();
  refs.confToggle.classList.toggle("hidden", !s || !s.hasScored);
}

async function rerun() {
  if (!isLoaded() || state.busy) return;
  state.busy = true;
  const q = state.query;
  try {
    let res;
    if (!q) {
      res = browse({ ...state.filters, top: 60 });
    } else {
      refs.summary.textContent = "Searching…";
      res = await search(q, { ...state.filters, top: 60 });
    }
    if (q !== state.query) { state.busy = false; return rerun(); } // query changed mid-flight
    state.expansion = res.expansion || [];
    state.lastTotal = res.total;
    renderList(res.papers, !!q);
    renderSummary(res, !!q);
    renderFacets();
  } finally {
    state.busy = false;
  }
}

function renderSummary(res, searching) {
  const s = stats();
  const parts = [];
  parts.push(`${res.total.toLocaleString()} ${searching ? "match" + (res.total === 1 ? "" : "es") : "dataset paper" + (res.total === 1 ? "" : "s")}`);
  if (!searching && s) parts.push(`of ${s.total.toLocaleString()} in the corpus`);
  if (res.papers.length < res.total) parts.push(`showing top ${res.papers.length}`);
  refs.summary.innerHTML = "";
  refs.summary.append(el("span", {}, parts.join(" · ")));
  if (state.expansion.length) {
    refs.summary.append(el("span", { class: "expansion" }, ` + also matched: ${state.expansion.join(", ")}`));
  }
}

const TRUNC = 280;
function renderList(papers, searching) {
  refs.list.innerHTML = "";
  if (!papers.length) {
    refs.list.append(el("div", { class: "empty" }, "No dataset papers match. Try fewer filters or a broader query."));
    return;
  }
  for (const p of papers) refs.list.append(card(p, searching));
}

function card(p, searching) {
  const head = el("div", { class: "card-head" });
  if (searching && p.rel != null) head.append(el("span", { class: "rel", title: "semantic similarity" }, p.rel.toFixed(2)));
  head.append(el("a", { class: "title", href: p.url || "#", target: "_blank", rel: "noopener" }, p.title || "(untitled)"));

  const authors = (p.authors || []).slice(0, 4).join(", ") + ((p.authors || []).length > 4 ? ", et al." : "");
  const metaBits = [authors, p.venue, p.year ? String(p.year) : null].filter(Boolean);
  const meta = el("div", { class: "card-meta" }, metaBits.join("  ·  "));

  const chips = el("div", { class: "card-chips" });
  if (p.modality && p.modality !== "other") chips.append(el("span", { class: "tag modality" }, p.modality));
  for (const t of (p.tasks || []).slice(0, 3)) chips.append(el("span", { class: "tag task" }, t));
  for (const l of (p.languages || []).slice(0, 4)) chips.append(el("span", { class: "tag lang" }, l));
  if (p.license) chips.append(el("span", { class: "tag license" }, p.license));

  const links = el("div", { class: "card-links" });
  const link = (href, label) => href && links.append(el("a", { href, target: "_blank", rel: "noopener" }, label));
  if (p.arxiv_id) link(`https://arxiv.org/abs/${p.arxiv_id}`, "arXiv");
  if (p.doi) link(`https://doi.org/${String(p.doi).replace(/^https?:\/\/doi\.org\//, "")}`, "DOI");
  if (p.hf) link(p.hf, "🤗 dataset");
  if (p.url) link(p.url, "page ↗");
  if (p.added) links.append(el("span", { class: "added" }, `added ${p.added}`));

  const body = el("div", { class: "card-body" });
  if (p.abstract) {
    const short = p.abstract.length > TRUNC;
    const abs = el("p", { class: "abstract" }, short ? p.abstract.slice(0, TRUNC).trimEnd() + "… " : p.abstract);
    if (short) {
      const more = el("button", { class: "more", onclick: () => { abs.textContent = p.abstract; more.remove(); } }, "more");
      abs.append(more);
    }
    body.append(abs);
  }

  return el("article", { class: "card" }, head, meta, chips.children.length ? chips : null, body, links);
}
