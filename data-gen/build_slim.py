#!/usr/bin/env python3
"""build_slim.py — turn the local corpus into the shipped search index.

Reads ~/research/dataset-corpus/papers.jsonl + the embedding cache produced by
embed_classify.mjs, keeps the is_dataset_paper=True rows, and writes — in ONE
shared order:

  data-gen/papers.slim.jsonl      slim records the browser renders
  public/data/emb_int8.bin        their int8 bge-small vectors (384-dim)
  public/data/meta.json           base manifest (ship_chunks.py adds count+files)

Run AFTER `node data-gen/embed_classify.mjs`. Then `python3 data-gen/ship_chunks.py`.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(os.path.expanduser("~/research/dataset-corpus/papers.jsonl"))
CACHE = ROOT / "data-gen" / "emb_cache.jsonl"
SLIM = ROOT / "data-gen" / "papers.slim.jsonl"
DATA = ROOT / "public" / "data"
DIM = 384
ABSTRACT_CAP = 1500

MODEL = "Xenova/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_cache() -> dict[str, bytes]:
    cache: dict[str, bytes] = {}
    if not CACHE.exists():
        raise SystemExit(f"embedding cache not found: {CACHE} — run embed_classify.mjs first")
    with open(CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[obj["id"]] = base64.b64decode(obj["v"])
    return cache


def slim(rec: dict) -> dict:
    abstract = (rec.get("abstract") or "").strip() or None
    if abstract and len(abstract) > ABSTRACT_CAP:
        abstract = abstract[:ABSTRACT_CAP].rsplit(" ", 1)[0] + " …"
    out = {
        "id": rec["id"],
        "title": rec.get("title"),
        "authors": [a.get("name") for a in (rec.get("authors") or []) if a.get("name")],
        "year": rec.get("year"),
        "venue": rec.get("venue"),
        "abstract": abstract,
        "url": rec.get("url"),
        "doi": rec.get("doi"),
        "arxiv_id": rec.get("arxiv_id"),
        "modality": rec.get("modality"),
        "tasks": rec.get("tasks") or [],
        "languages": rec.get("languages") or [],
        "license": rec.get("license"),
        "hf": rec.get("hf_dataset_url"),
        "conf": rec.get("dataset_score"),
        "added": (rec.get("crawled_at") or "")[:10] or None,
    }
    return out


def main() -> int:
    cache = load_cache()
    DATA.mkdir(parents=True, exist_ok=True)
    kept = 0
    missing = 0
    with open(SLIM, "w", encoding="utf-8") as fs, open(DATA / "emb_int8.bin", "wb") as fb:
        with open(CORPUS, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("is_dataset_paper"):
                    continue
                vec = cache.get(rec["id"])
                if vec is None or len(vec) != DIM:
                    missing += 1
                    continue  # keep slim and emb aligned: only ship embedded rows
                fs.write(json.dumps(slim(rec), ensure_ascii=False) + "\n")
                fb.write(vec)
                kept += 1

    meta = {
        "model": MODEL,
        "dim": DIM,
        "count": kept,
        "quant": "int8/127",
        "normalized": True,
        "query_prefix": QUERY_PREFIX,
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"slim: {kept} dataset papers written ({missing} skipped for missing embedding)")
    print(f"  {SLIM}")
    print(f"  {DATA/'emb_int8.bin'} ({kept*DIM:,} bytes)")
    print("next: python3 data-gen/ship_chunks.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
