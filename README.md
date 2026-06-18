# Dataset Reviewer

A public workbench for **finding newly-proposed ML & NLP datasets and
benchmarks** by meaning. Type a free-text description of the data you need
("multilingual toxicity benchmark", "speech emotion dataset", "tabular fraud
detection") and the corpus of dataset/benchmark papers re-ranks by semantic
similarity — **entirely in your browser**. Updated daily.

🔎 No server, no accounts: query embedding (bge-small via transformers.js) and
cosine ranking happen client-side over shipped int8 embeddings. Sibling to
[mis-lit-reviewer](https://mis-lit-reviewer.misclaw.app).

## What counts as a "dataset paper"

A paper that **contributes a dataset or benchmark as its primary artifact** — not
one that merely uses existing data. Papers enter the corpus two ways:

- **High-precision venues** (auto-included): every accepted paper of the
  **NeurIPS Datasets & Benchmarks Track** (2022– via the OpenReview API) and ACL
  Anthology resource/LREC papers.
- **arXiv** (daily, broad ML/AI: `cs.CL, cs.CV, cs.LG, cs.AI, cs.IR, cs.SD,
  eess.AS`) filtered by a **hybrid classifier**: a keyword/regex recall gate
  (`corpus.py`) finds candidates; a bge-small **prototype-similarity** score
  (`data-gen/prototypes.json`) supplies precision. Each paper keeps its
  `dataset_score` and the signals that fired.

Best-effort facets (task / language / license) come from the **Hugging Face Hub**
when a dataset there links the paper's arXiv id; modality is derived from the
arXiv category. Facets show only when present.

## Stack

- **Frontend:** Vite + vanilla JS (`src/`), Cloudflare Pages at the domain root.
- **Search:** `@huggingface/transformers` with `Xenova/bge-small-en-v1.5`
  (384-dim) over `public/data/emb_int8.bin.part*` (int8) +
  `public/data/papers.slim.jsonl.gz.part*`. Files are split into <25 MiB chunks
  (Cloudflare Pages per-file limit); `public/data/meta.json` lists the manifest
  and the loader reassembles them at runtime.

## Develop

```bash
npm install
npm run dev      # http://localhost:5173
```

## Data pipeline

The corpus lives locally at `~/research/dataset-corpus/papers.jsonl`; the crawler
lives at `~/research/dataset-crawler/` (not deployed). To rebuild the shipped
index after the corpus grows:

```bash
node   data-gen/embed_classify.mjs   # embed (resumable cache) + score candidates
python3 data-gen/build_slim.py        # corpus -> papers.slim.jsonl + emb_int8.bin
python3 data-gen/ship_chunks.py       # gzip + split + meta.json manifest
git add public/data && git commit && git push   # CI deploys
```

`node data-gen/_smoketest.mjs` replays the browser's query path offline to spot-check
ranking quality.

## Deploy

Push to `main` → GitHub Actions builds (`npm run build`) and deploys `dist/` to
Cloudflare Pages → live at **dataset-reviewer.misclaw.app** in ~1 minute.
