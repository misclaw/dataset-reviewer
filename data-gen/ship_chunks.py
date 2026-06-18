#!/usr/bin/env python3
"""Package the dataset for Cloudflare Pages: gzip the slim records, split both
big files into ≤20 MiB parts, and write the full meta.json manifest.

Run AFTER `python3 data-gen/build_slim.py` (which writes papers.slim.jsonl, a
single emb_int8.bin, and a meta.json without the files manifest). Verifies
records ↔ vectors alignment before touching anything.

Usage:  python3 data-gen/ship_chunks.py
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLIM = ROOT / "data-gen" / "papers.slim.jsonl"
DATA = ROOT / "public" / "data"
DIM = 384
PART = 20 * 1024 * 1024


def split(blob: bytes, name: str) -> dict:
    for old in DATA.glob(f"{name}.part*"):
        old.unlink()
    sizes = []
    for i in range(0, len(blob), PART):
        chunk = blob[i:i + PART]
        (DATA / f"{name}.part{len(sizes)}").write_bytes(chunk)
        sizes.append(len(chunk))
    return {"parts": len(sizes), "size": len(blob), "part_sizes": sizes}


def main() -> int:
    n = sum(1 for line in open(SLIM, encoding="utf-8") if line.strip())

    emb = DATA / "emb_int8.bin"
    if not emb.exists():
        sys.exit("emb_int8.bin not found — run `python3 data-gen/build_slim.py` first")
    vecs = emb.stat().st_size // DIM
    if vecs != n or emb.stat().st_size % DIM:
        sys.exit(f"alignment error: {n:,} records vs {vecs:,} vectors — refusing to ship")

    print(f"{n:,} records · gzipping…", flush=True)
    gz = SLIM.with_suffix(".jsonl.gz")
    # Deterministic gzip (mtime=0, no embedded filename) so an unchanged corpus
    # produces byte-identical output — the daily job's git diff stays empty and
    # we don't accrue no-op commits.
    raw = SLIM.read_bytes()
    with open(gz, "wb") as out, gzip.GzipFile(fileobj=out, mode="wb", compresslevel=9, mtime=0) as gzf:
        gzf.write(raw)

    meta_path = DATA / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["count"] = n
    meta["files"] = {
        "papers.slim.jsonl.gz": split(gz.read_bytes(), "papers.slim.jsonl.gz"),
        "emb_int8.bin": split(emb.read_bytes(), "emb_int8.bin"),
    }
    gz.unlink()
    emb.unlink()  # shipped as parts; build_slim regenerates it from the emb cache
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for f, info in meta["files"].items():
        print(f"  {f}: {info['parts']} part(s), {info['size']:,} bytes")
    print("meta.json updated — ready to commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
