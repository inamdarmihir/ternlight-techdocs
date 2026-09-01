"""
eval_retrieval.py

Real code-search retrieval benchmark: given a docstring, find the function
it documents among 500 real, distinct CodeSearchNet Python functions (see
build_test_set.py — held-out `test` split, never touched during training).

Three real embedders, three real Qdrant collections, one query set:

  1. ours       — our own techdocs-distilled ternlight checkpoint (1.58-bit
                  ternary, ~9.5M params, this repo's training run)
  2. generic    — the actual published @ternlight/base package (1.58-bit
                  ternary, general-purpose distillation, not tuned for code)
  3. fastembed  — BAAI/bge-small-en-v1.5 via FastEmbed (33M params, fp32,
                  the "normal-sized" baseline everyone already uses)

Same 384-dim output for all three, so the comparison isn't confounded by
embedding size. Reports recall@{1,5,10} (does the correct function appear in
the top-k for its own docstring, against 499 real distractors) and per-query
embed latency, since "production ready" is a latency/size question as much
as a quality one.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from qdrant_client import QdrantClient, models

TERNLIGHT_TRAINING_DIR = Path(__file__).parent.parent / "upstream" / "training" / "distill"
BRIDGE_DIR = Path(__file__).parent.parent / "ternlight-bridge"
TEST_SET_PATH = Path(__file__).parent / "test_set.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def load_test_set() -> list[dict]:
    return json.loads(TEST_SET_PATH.read_text())["pairs"]


# ── Embedders ──────────────────────────────────────────────────────────────

def embed_ours(texts: list[str], ckpt_path: Path) -> tuple[list[list[float]], float]:
    """Our techdocs-distilled checkpoint, via ternlight's own real
    load_for_eval (ternary embedding table, BitLinear lambda=1, exactly the
    forward pass that gets shipped), not a reimplementation."""
    sys.path.insert(0, str(TERNLIGHT_TRAINING_DIR))
    from evaluation import load_for_eval
    from transformers import AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    em = load_for_eval(ckpt_path, device, embedding_format="ternary")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    t0 = time.perf_counter()
    vecs: list[list[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            toks = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            out = em.model(toks["input_ids"].to(device), toks["attention_mask"].to(device))
            vecs.extend(out.cpu().tolist())
    elapsed = time.perf_counter() - t0
    return vecs, elapsed


def embed_generic_ternlight(texts: list[str]) -> tuple[list[list[float]], float]:
    """The actual published @ternlight/base npm package, via the Node
    bridge in ../ternlight-bridge. Real inference, not a Python port."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["node", "embed_batch.mjs"],
        input=json.dumps(texts), capture_output=True, text=True, cwd=BRIDGE_DIR, timeout=600,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"ternlight bridge failed: {proc.stderr}")
    return json.loads(proc.stdout), elapsed


def embed_fastembed(texts: list[str]) -> tuple[list[list[float]], float]:
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    t0 = time.perf_counter()
    vecs = [v.tolist() for v in model.embed(texts)]
    elapsed = time.perf_counter() - t0
    return vecs, elapsed


# ── Qdrant retrieval + scoring ───────────────────────────────────────────────

def build_collection(client: QdrantClient, name: str, vectors: list[list[float]]) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    client.upsert(
        collection_name=name,
        points=models.Batch(ids=list(range(len(vectors))), vectors=vectors, payloads=[{}] * len(vectors)),
    )


def score_recall(client: QdrantClient, name: str, query_vectors: list[list[float]]) -> dict:
    ks = (1, 5, 10)
    hits = {k: 0 for k in ks}
    for qid, qvec in enumerate(query_vectors):
        results = client.query_points(collection_name=name, query=qvec, limit=max(ks)).points
        retrieved_ids = [r.id for r in results]
        for k in ks:
            if qid in retrieved_ids[:k]:
                hits[k] += 1
    n = len(query_vectors)
    return {f"recall@{k}": round(hits[k] / n, 4) for k in ks}


def run_embedder(name: str, embed_fn, documents: list[str], queries: list[str], client: QdrantClient) -> dict:
    print(f"\n=== {name} ===")
    doc_vecs, doc_secs = embed_fn(documents)
    print(f"  embedded {len(documents)} documents in {doc_secs:.2f}s "
          f"({len(documents) / doc_secs:.1f} docs/sec)")
    build_collection(client, f"techdocs_eval_{name}", doc_vecs)

    query_vecs, query_secs = embed_fn(queries)
    print(f"  embedded {len(queries)} queries in {query_secs:.2f}s "
          f"({query_secs / len(queries) * 1000:.2f} ms/query)")

    recall = score_recall(client, f"techdocs_eval_{name}", query_vecs)
    print(f"  {recall}")

    return {
        "n_documents": len(documents),
        "n_queries": len(queries),
        "document_embed_seconds": round(doc_secs, 2),
        "document_embed_per_sec": round(len(documents) / doc_secs, 1),
        "query_embed_ms_per_query": round(query_secs / len(queries) * 1000, 3),
        **recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to our trained QAT checkpoint (.pt)")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--skip-ours", action="store_true", help="Validate the harness with generic+fastembed only")
    args = parser.parse_args()

    pairs = load_test_set()
    documents = [p["document"] for p in pairs]
    queries = [p["query"] for p in pairs]
    print(f"Real held-out CodeSearchNet test set: {len(pairs)} query/document pairs "
          f"(same id in both lists = the correct match)")

    client = QdrantClient(url=args.qdrant_url)
    results = {}

    results["fastembed_bge_small"] = run_embedder("fastembed_bge_small", embed_fastembed, documents, queries, client)
    results["generic_ternlight_base"] = run_embedder(
        "generic_ternlight_base",
        lambda t: embed_generic_ternlight(t),
        documents, queries, client,
    )
    if not args.skip_ours:
        results["ours_techdocs_ternlight"] = run_embedder(
            "ours_techdocs_ternlight",
            lambda t: embed_ours(t, args.ckpt),
            documents, queries, client,
        )

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
