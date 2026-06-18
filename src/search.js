// In-browser semantic search over the dataset/benchmark corpus.
//
// Loads the shipped int8 bge-small embeddings + slim metadata once, embeds the
// query with the SAME model (so vectors match), and ranks the single result
// list by cosine similarity. Facets filter the list. All client-side — no
// server, nothing leaves the browser.
import { pipeline } from "@huggingface/transformers";

const MODEL = "Xenova/bge-small-en-v1.5";

let extractorP = null;
let DATA = null;
let indexP = null;

export function isLoaded() {
  return !!DATA;
}

export function loadModel() {
  if (!extractorP) extractorP = pipeline("feature-extraction", MODEL, { dtype: "q8" });
  return extractorP;
}

// Big data files ship split into <25 MiB parts (Cloudflare Pages limit):
// <name>.part0, .part1, … listed in meta.json's "files" manifest. Fetch all in
// parallel and concatenate in order (gzip bytes reassemble byte-identically).
async function fetchChunked(base, name, manifest) {
  const parts = manifest?.parts || 1;
  const urls = [];
  for (let i = 0; i < parts; i++) urls.push(`${base}data/${name}.part${i}`);
  const bufs = await Promise.all(urls.map(async (u) => {
    const res = await fetch(u);
    if (!res.ok) throw new Error(`fetch ${u}: HTTP ${res.status}`);
    return res.arrayBuffer();
  }));
  const total = bufs.reduce((n, b) => n + b.byteLength, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const b of bufs) { out.set(new Uint8Array(b), off); off += b.byteLength; }
  return out;
}

export function loadIndex(onStep) {
  if (!indexP) indexP = _loadIndex(onStep);
  return indexP;
}

async function _loadIndex(onStep) {
  if (DATA) return DATA;
  const base = import.meta.env.BASE_URL || "./";
  onStep?.("Loading index…");
  const meta = await (await fetch(base + "data/meta.json")).json();

  onStep?.("Downloading datasets…");
  const [embBytes, gzBytes] = await Promise.all([
    fetchChunked(base, "emb_int8.bin", meta.files?.["emb_int8.bin"]),
    fetchChunked(base, "papers.slim.jsonl.gz", meta.files?.["papers.slim.jsonl.gz"]),
  ]);
  const emb = new Int8Array(embBytes.buffer, embBytes.byteOffset, embBytes.byteLength);

  let text;
  if (gzBytes[0] === 0x1f && gzBytes[1] === 0x8b) {
    const stream = new Response(gzBytes).body.pipeThrough(new DecompressionStream("gzip"));
    text = await new Response(stream).text();
  } else {
    text = new TextDecoder().decode(gzBytes);
  }

  onStep?.("Parsing…");
  const papers = [];
  let maxYear = 0;
  for (const line of text.split("\n")) {
    if (!line) continue;
    const p = JSON.parse(line);
    papers.push(p);
    if (p.year && p.year > maxYear) maxYear = p.year;
  }
  if (papers.length !== meta.count) {
    console.warn(`paper/embedding count mismatch: ${papers.length} vs ${meta.count}`);
  }
  DATA = { meta, emb, papers, dim: meta.dim, count: papers.length, currentYear: maxYear || 2026 };

  onStep?.("Warming up the model…");
  loadModel(); // background; first query awaits it
  return DATA;
}

// ---- query-side synonym / acronym expansion (multi-vector MAX) ----
// bge-small doesn't reliably treat "LLM" == "large language model"; expand the
// query into variants (substituting synonyms in place) and score each paper by
// its best-matching variant. Cheap, client-side, no re-embed.
const SYNONYM_GROUPS = [
  ["LLM", "LLMs", "large language model", "large language models"],
  ["NLP", "natural language processing"],
  ["VQA", "visual question answering"],
  ["ASR", "automatic speech recognition", "speech recognition"],
  ["NER", "named entity recognition"],
  ["RAG", "retrieval-augmented generation", "retrieval augmented generation"],
  ["QA", "question answering"],
  ["MT", "machine translation"],
];
const MAX_VARIANTS = 4;
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function queryVariants(q) {
  const variants = [q];
  const added = [];
  for (const group of SYNONYM_GROUPS) {
    const present = group.find((t) => new RegExp("\\b" + escapeRe(t) + "\\b", "i").test(q));
    if (!present) continue;
    const re = new RegExp("\\b" + escapeRe(present) + "\\b", "gi");
    for (const alt of group) {
      if (group.some((g) => g.toLowerCase() === alt.toLowerCase() && new RegExp("\\b" + escapeRe(g) + "\\b", "i").test(q))) continue;
      if (variants.length >= MAX_VARIANTS) break;
      variants.push(q.replace(re, alt));
      added.push(alt);
    }
  }
  return { variants: [...new Set(variants)], added: [...new Set(added)] };
}

async function embedText(text) {
  const ex = await loadModel();
  const out = await ex((DATA.meta.query_prefix || "") + text, { pooling: "mean", normalize: true });
  return out.data; // Float32Array, unit length
}

// ---- facets ----
// Multi-valued facets (tasks/languages) match if ANY value is allowed.
const FACET_GETTERS = {
  modality: (p) => (p.modality ? [p.modality] : []),
  venue: (p) => (p.venue ? [p.venue] : []),
  year: (p) => (p.year ? [String(p.year)] : []),
  language: (p) => p.languages || [],
  license: (p) => (p.license ? [p.license] : []),
  task: (p) => p.tasks || [],
};

function passesFacets(p, filters) {
  for (const [key, getter] of Object.entries(FACET_GETTERS)) {
    const allowed = filters[key];
    if (!allowed || !allowed.size) continue;
    const vals = getter(p);
    if (!vals.some((v) => allowed.has(v))) return false;
  }
  if (filters.minConf != null && !(p.conf == null || p.conf >= filters.minConf)) return false;
  return true;
}

// Distinct facet values + counts over the papers passing the OTHER active
// filters (so counts reflect what selecting them would yield). Only facets with
// ≥2 distinct present values are returned — empty facets stay hidden.
export function facetOptions(filters = {}) {
  if (!DATA) return {};
  const out = {};
  for (const key of Object.keys(FACET_GETTERS)) {
    const counts = new Map();
    const others = { ...filters, [key]: null };
    for (const p of DATA.papers) {
      if (!passesFacets(p, others)) continue;
      for (const v of FACET_GETTERS[key](p)) counts.set(v, (counts.get(v) || 0) + 1);
    }
    if (counts.size >= 2) {
      out[key] = [...counts.entries()].sort((a, b) => b[1] - a[1] || String(a[0]).localeCompare(String(b[0])));
    }
  }
  return out;
}

export function stats() {
  if (!DATA) return null;
  const hasScored = DATA.papers.some((p) => p.conf != null && p.conf < 0.999);
  return { total: DATA.count, currentYear: DATA.currentYear, hasScored };
}

// No query: browse the corpus filtered + sorted (newest first).
export function browse(filters = {}) {
  if (!DATA) throw new Error("index not loaded");
  const out = DATA.papers.filter((p) => passesFacets(p, filters));
  out.sort((a, b) => (b.added || "").localeCompare(a.added || "") || (b.year || 0) - (a.year || 0));
  return { papers: out.slice(0, filters.top || 60), total: out.length, expansion: [] };
}

// Query: rank the filtered list by best-variant cosine similarity.
export async function search(query, filters = {}) {
  if (!DATA) throw new Error("index not loaded");
  const { variants, added } = queryVariants(query);
  const qvecs = [];
  for (const v of variants) qvecs.push(await embedText(v));
  const { dim, count, papers, emb } = DATA;
  const scored = [];
  for (let r = 0; r < count; r++) {
    const p = papers[r];
    if (!passesFacets(p, filters)) continue;
    const baseIdx = r * dim;
    let best = -Infinity;
    for (const q of qvecs) {
      let s = 0;
      for (let i = 0; i < dim; i++) s += q[i] * emb[baseIdx + i];
      if (s > best) best = s;
    }
    scored.push([Math.max(0, best / 127), r]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  const top = filters.top || 60;
  return {
    papers: scored.slice(0, top).map(([rel, r]) => ({ ...papers[r], rel })),
    total: scored.length,
    expansion: added,
  };
}
