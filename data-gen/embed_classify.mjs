// embed_classify.mjs — stage 2 of the dataset pipeline.
//
// 1. Embeds every corpus paper (title+abstract) with bge-small int8 into a
//    resumable id-keyed cache (data-gen/emb_cache.jsonl). Each paper embedded
//    once; reused for classification, the shipped base index, and the daily
//    overlay.
// 2. Scores papers that still need it (dataset_score == null — i.e. arXiv
//    candidates from the recall gate) against the prototype set:
//       dataset_score = max cos(paper, positive) − max cos(paper, negative)
//    and sets is_dataset_paper = dataset_score >= DS_THRESHOLD. Venue-sourced
//    papers (dataset_score already 1.0) are left untouched.
// 3. Writes the updated scores back to the corpus atomically.
//
// Run:  node data-gen/embed_classify.mjs            (embed missing + score null)
//       node data-gen/embed_classify.mjs --rescore  (re-score all arXiv papers)
//       node data-gen/embed_classify.mjs --force     (re-embed everything)
import fs from "fs";
import os from "os";
import path from "path";
import { embedDocs, quantizeInt8, dequantize, cosine, docText, int8ToB64, b64ToInt8, DIM } from "./bge.mjs";

const CORPUS = path.join(os.homedir(), "research", "dataset-corpus", "papers.jsonl");
const CACHE = new URL("emb_cache.jsonl", import.meta.url).pathname;
const PROTO = new URL("prototypes.json", import.meta.url).pathname;
// Inclusion rule for arXiv candidates (tuned on a 50-paper hand-labeled sample,
// ~100% precision / ~90% recall): a dataset/benchmark token in the TITLE plus a
// non-negative prototype margin, OR a high prototype margin on its own. The
// title token is the dominant signal; the prototype score (kept as the displayed
// confidence) acts as a sign check and catches the title-less high scorers.
const TITLE_FLOOR = parseFloat(process.env.DS_TITLE_FLOOR || "0.0");
const SCORE_HIGH = parseFloat(process.env.DS_SCORE_HIGH || "0.08");
const TITLE_WORD = /\b(?:bench(?:mark(?:s|ing)?)?|datasets?|corpus|corpora|treebank|databank|testset|test set|evaluation suite)\b/i;
const TITLE_SUFFIX = /[\w-](?:Bench|Benchmark|Gym|Eval|Suite|Corpus|Dataset)\b/; // CamelCase: RedactionBench, SkillChain-Gym
const titleToken = (t) => TITLE_WORD.test(t || "") || TITLE_SUFFIX.test(t || "");
const BATCH = 64;
const rescore = process.argv.includes("--rescore");
const force = process.argv.includes("--force");

function readCorpus() {
  return fs.readFileSync(CORPUS, "utf8").split("\n").filter((l) => l.length).map((l) => JSON.parse(l));
}
function writeCorpus(papers) {
  const tmp = CORPUS + ".tmp";
  fs.writeFileSync(tmp, papers.map((p) => JSON.stringify(p)).join("\n") + "\n");
  fs.renameSync(tmp, CORPUS);
}
function loadCache() {
  const m = new Map();
  if (!force && fs.existsSync(CACHE)) {
    for (const line of fs.readFileSync(CACHE, "utf8").split("\n")) {
      if (!line) continue;
      const { id, v } = JSON.parse(line);
      m.set(id, b64ToInt8(v));
    }
  }
  return m;
}
const isVenue = (p) => (p.dataset_signals || []).some((s) => s.startsWith("venue:"));

const papers = readCorpus();
const cache = loadCache();
if (force && fs.existsSync(CACHE)) fs.unlinkSync(CACHE);
const cacheOut = fs.createWriteStream(CACHE, { flags: force ? "w" : "a" });

// ---- 1. embed everything not cached ----
const todo = papers.filter((p) => !cache.has(p.id));
console.log(`corpus ${papers.length} · cached ${cache.size} · to embed ${todo.length}`);
const t0 = Date.now();
for (let i = 0; i < todo.length; i += BATCH) {
  const slice = todo.slice(i, i + BATCH);
  const vecs = await embedDocs(slice.map(docText));
  for (let j = 0; j < slice.length; j++) {
    const q = quantizeInt8(vecs[j]);
    cache.set(slice[j].id, q);
    cacheOut.write(JSON.stringify({ id: slice[j].id, v: int8ToB64(q) }) + "\n");
  }
  if ((i + slice.length) % 512 === 0 || i + slice.length === todo.length) {
    const rate = ((i + slice.length) / ((Date.now() - t0) / 1000)).toFixed(0);
    process.stdout.write(`  embedded ${i + slice.length}/${todo.length} (${rate}/s)\r`);
  }
}
await new Promise((res) => cacheOut.end(res));
if (todo.length) console.log();

// ---- 2. score candidates ----
const proto = JSON.parse(fs.readFileSync(PROTO, "utf8"));
const posF = await embedDocs(proto.positive);
const negF = await embedDocs(proto.negative);
const maxCos = (v, set) => set.reduce((m, p) => Math.max(m, cosine(v, p)), -Infinity);

let scored = 0, kept = 0;
for (const p of papers) {
  if (isVenue(p)) continue;                       // venue papers stay is_dataset_paper=true, score 1.0
  if (!rescore && p.dataset_score != null) continue;
  const int8 = cache.get(p.id);
  if (!int8) continue;
  const v = dequantize(int8);
  const score = maxCos(v, posF) - maxCos(v, negF);
  p.dataset_score = Math.round(score * 1e4) / 1e4;
  p.is_dataset_paper = (titleToken(p.title) && score >= TITLE_FLOOR) || score >= SCORE_HIGH;
  scored++;
  if (p.is_dataset_paper) kept++;
}
writeCorpus(papers);

const venue = papers.filter(isVenue).length;
const datasetPapers = papers.filter((p) => p.is_dataset_paper).length;
console.log(`scored ${scored} candidates (title-token & score>=${TITLE_FLOOR}, or score>=${SCORE_HIGH}; kept ${kept})`);
console.log(`corpus: ${papers.length} papers · ${venue} venue · ${datasetPapers} is_dataset_paper=true`);
