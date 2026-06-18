// Shared bge-small embedding helpers for the data-gen pipeline. The SAME model
// (Xenova/bge-small-en-v1.5, 384-dim, q8) the browser loads via transformers.js,
// so corpus vectors computed here are directly comparable to query vectors the
// client computes. Documents are embedded WITHOUT the query prefix (matching how
// the browser scores the corpus); only queries get the prefix, in the client.
import { pipeline } from "@huggingface/transformers";

export const MODEL = "Xenova/bge-small-en-v1.5";
export const DIM = 384;
export const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";

let extractorP = null;
export function loadExtractor() {
  if (!extractorP) extractorP = pipeline("feature-extraction", MODEL, { dtype: "q8" });
  return extractorP;
}

// Embed an array of document texts → array of unit-norm Float32 vectors.
export async function embedDocs(texts) {
  const ex = await loadExtractor();
  const o = await ex(texts, { pooling: "mean", normalize: true });
  return o.tolist();
}

export function quantizeInt8(vec) {
  const b = new Int8Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    let q = Math.round(vec[i] * 127);
    b[i] = q > 127 ? 127 : q < -128 ? -128 : q;
  }
  return b;
}

export function dequantize(int8) {
  const f = new Float32Array(int8.length);
  for (let i = 0; i < int8.length; i++) f[i] = int8[i] / 127;
  return f;
}

// Cosine of two equal-length vectors (inputs are unit-norm float, or int8/127).
export function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// The text we embed for a paper: title + blank line + abstract, capped (bge-small
// truncates at 512 tokens anyway; the char cap keeps batches cheap).
export function docText(p) {
  return ((p.title || "") + "\n\n" + (p.abstract || "")).slice(0, 2000);
}

export const int8ToB64 = (int8) => Buffer.from(int8.buffer, int8.byteOffset, int8.byteLength).toString("base64");
export function b64ToInt8(s) {
  const buf = Buffer.from(s, "base64");
  return new Int8Array(buf.buffer, buf.byteOffset, buf.length);
}
