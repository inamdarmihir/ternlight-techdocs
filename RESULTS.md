# Results

Standalone summary of this repo's benchmark. See the main
[README](README.md) for method, context, and the full "what the results
actually show" discussion — this file exists so the numbers and their
immediate caveats can be read (or linked to) without the rest of the repo.

Full logs and machine-readable output are in [`eval/`](eval/):
[`eval/clean_eval_run.log`](eval/clean_eval_run.log) (raw stdout) and
[`eval/results.json`](eval/results.json) (structured). Exact commands to
regenerate these numbers on your own hardware are in
[`eval/README.md`](eval/README.md).

## Setup

Real 3-way retrieval benchmark, 500 held-out CodeSearchNet Python
query/document pairs (seed 42, training never touches this split), real
Qdrant collections (`qdrant/qdrant:v1.19.0`, cosine distance, 384-dim
vectors for all three embedders), one clean run with no other CPU-heavy
process running. MPS for our checkpoint, CPU for the other two.

## Table

| Embedder | Params | Weights | recall@1 | recall@5 | recall@10 | Doc embed | Query latency |
|---|---:|---|---:|---:|---:|---:|---:|
| **Ours** (techdocs QAT) | 9.5M | ternary | 0.872 | 0.948 | 0.964 | 357.5 docs/sec | 1.64 ms/query |
| Generic `@ternlight/base` | — | ternary | 0.902 | 0.966 | 0.974 | 19.3 docs/sec | 20.34 ms/query |
| FastEmbed `bge-small-en-v1.5` | 33M | fp32 | 0.982 | 0.992 | 0.996 | 1.7 docs/sec | 410.66 ms/query |

## The two findings

1. **Ternary quantization is a real middle ground, not just an extreme.**
   Both ternary rows land within 8-11 points of recall@1 (2-5 points at
   recall@10) of the fp32 row, at a large, real latency advantage
   (25x-200x faster per query). A
   [published Qdrant benchmark](https://github.com/Dylancouzon/static-embeddings)
   of static (no-transformer) embeddings on a similar code-search task
   scored 0.2887 NDCG@10 — worse than BM25 (0.2955) — a materially larger
   quality gap than either ternary model shows here. The two protocols
   (recall@k on a matching task vs. NDCG@10 on BEIR-style exact search) are
   not point-for-point comparable, but the qualitative gap size difference
   is the point: keeping the transformer and quantizing it lands much
   closer to full precision than removing the transformer entirely does.

2. **⚠️ Domain-specific distillation did not beat the generic checkpoint
   here (negative result, reported as-is).** Our techdocs-tuned model is
   18x higher document throughput and 12x lower query latency than the
   generic Ternlight release, but scores 2-3 points *lower* on
   recall@1/5/10 — not higher, which was the hypothesis going in. At this
   data scale (25k domain-specific samples, 30 epochs) and against an
   unknown-scale generic release, specialization traded some of the generic
   model's quality for speed rather than trading generic quality for domain
   quality. This is a single run at this specific scale — see the main
   README's [Limitations](README.md#limitations) for what this does and
   doesn't imply.

## Caveats that apply to every number above

- **Latency/throughput is hardware- and load-specific**, not a general
  benchmark of any of the three tools. Our checkpoint ran on Apple
  Silicon's `mps`; FastEmbed and the generic Ternlight package ran on CPU.
  See the README's [Apple Silicon vs. CPU](README.md#apple-silicon-mps-vs-cpu)
  section.
- **FastEmbed's absolute numbers here (1.7 docs/sec) are slower than commonly
  published `bge-small` benchmarks** (typically tens of docs/sec on CPU).
  Default settings, no thread-count tuning, one machine — a lower bound on
  a tuned deployment, not a FastEmbed-in-general number.
- **Recall numbers are from a single 500-pair sample**, one dataset
  (CodeSearchNet Python), one query style (docstring → function). No
  confidence intervals, no repeated runs, no other languages or corpora
  tested.

Full narrative, method, and reproduction instructions: [README.md](README.md).
