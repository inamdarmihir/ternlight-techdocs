# eval/ — regenerating the benchmark

This directory holds the actual 3-way retrieval benchmark referenced in the
main [README](../README.md) and [RESULTS.md](../RESULTS.md). Everything here
is a real script that talks to a real Qdrant server — there's no synthetic
data and no hand-written numbers.

| File | What it is |
|---|---|
| [`build_test_set.py`](build_test_set.py) | Builds the held-out query/document pairs from CodeSearchNet Python's own `test` split |
| [`test_set.json`](test_set.json) | Committed output of the above (seed 42, 500 pairs) — regenerate only if you want a different sample |
| [`eval_retrieval.py`](eval_retrieval.py) | The actual 3-way benchmark: embeds, indexes into Qdrant, scores recall@{1,5,10} |
| [`results.json`](results.json) | Committed machine-readable output of the last clean run |
| [`clean_eval_run.log`](clean_eval_run.log) | Committed raw stdout of that same run |

## Prerequisites

See the main [README's Prerequisites section](../README.md#prerequisites) for
exact version pins (Python 3.10-3.12, Node ≥ 18, `qdrant/qdrant:v1.19.0`).
Specifically for this directory:

- A running Qdrant server, reachable at `http://localhost:6333` by default:

  ```bash
  docker run -p 6333:6333 qdrant/qdrant:v1.19.0
  ```

- This repo's own Python dependencies (torch, transformers, datasets,
  qdrant-client, fastembed, bitlinear — see [`../requirements.txt`](../requirements.txt)):

  ```bash
  pip install -r ../requirements.txt
  ```

- The Node bridge's dependencies installed **before** running
  `eval_retrieval.py` — it shells out to `../ternlight-bridge/embed_batch.mjs`
  for the generic-Ternlight leg:

  ```bash
  cd ../ternlight-bridge && npm ci && cd -
  ```

- To evaluate *your own* checkpoint (the `--ckpt` flag below), a trained QAT
  checkpoint from the main README's [Quickstart](../README.md#quickstart-reproducing-this-end-to-end).
  Not required for `--skip-ours` runs.

## Regenerating the held-out test set

Optional — [`test_set.json`](test_set.json) is already committed with a fixed
seed, so this only matters if you want a different sample size or a
different seed. This downloads and shuffles the full CodeSearchNet Python
`test` split (~22k rows) from the Hugging Face Hub:

```bash
python3 build_test_set.py
```

Edit `N_SAMPLES` / `SEED` at the top of the script to change the sample.
Every downstream script reads `test_set.json` as-is — there's no re-drawing
inside `eval_retrieval.py`.

## Running the benchmark

All commands below assume you're in this directory (`cd eval`) with Qdrant
already running.

**Fast: validate the harness without a trained checkpoint (~10 minutes).**
Runs the FastEmbed and generic-Ternlight legs only — everything except the
step that needs your own checkpoint:

```bash
python3 eval_retrieval.py --skip-ours
```

**Full: all three embedders (~10-15 minutes on Apple Silicon for "ours" +
CPU for the other two; FastEmbed dominates the wall-clock, see below):**

```bash
python3 eval_retrieval.py --ckpt ../upstream/training/distill/runs/techdocs-qat-<run-id>/checkpoint_ep30.pt
```

**Custom Qdrant location:**

```bash
python3 eval_retrieval.py --ckpt <path> --qdrant-url http://your-host:6333
```

### CLI flags

| Flag | Required | Meaning |
|---|---|---|
| `--ckpt PATH` | Yes, unless `--skip-ours` | Path to a trained QAT checkpoint (`.pt`), loaded via Ternlight's `load_for_eval` |
| `--qdrant-url URL` | No (default `http://localhost:6333`) | Qdrant REST endpoint |
| `--skip-ours` | No | Skip the "ours" leg entirely — useful for validating the harness before training anything |

### What each run produces

- Overwrites [`results.json`](results.json) with the new numbers.
- Prints the same summary to stdout that [`clean_eval_run.log`](clean_eval_run.log)
  captured — redirect to a file if you want to keep your own log:
  `python3 eval_retrieval.py --ckpt <path> 2>&1 | tee my_run.log`.
- Creates/replaces three Qdrant collections named `techdocs_eval_<name>` —
  safe to re-run repeatedly, each run deletes and rebuilds its own
  collections.

## Expected runtime, per embedder

From the committed [`clean_eval_run.log`](clean_eval_run.log), on the machine
this was originally run on (Apple Silicon for "ours", CPU for the other two,
500 docs + 500 queries each):

| Embedder | Doc embed | Query embed | Total |
|---|---|---|---|
| FastEmbed `bge-small-en-v1.5` | ~297s | ~205s | **~8-9 minutes** — dominates total wall-clock |
| Generic `@ternlight/base` | ~26s | ~10s | ~36s |
| Ours (techdocs QAT) | ~1.4s | ~0.8s | ~2s |

If your run takes much longer than this on comparable hardware, something is
likely fighting the process for CPU — see the caveat below before trusting
the numbers.

## The one thing that will silently corrupt your results

**Run this with nothing else CPU-heavy in the background — most importantly,
not concurrently with a training job on the same machine.** FastEmbed's ONNX
runtime is CPU-bound and has no priority protection; an earlier validation
pass that ran this harness while QAT training was still running produced
recall numbers that matched the clean run almost exactly, but a FastEmbed
throughput number that was ~175x worse (0.3 docs/sec and 2,516 ms/query,
instead of the clean run's 1.7 docs/sec and 410 ms). The recall columns are
comparatively robust to this; the latency/throughput columns are not.

Before trusting a run's latency numbers, confirm nothing else CPU-heavy is
running — `ps aux`, not just checking that a training log looks finished (a
training process can still be exiting or checkpointing after its last log
line).

## Regenerating `RESULTS.md` / the README table

Neither is auto-generated from `results.json` — after a run you're satisfied
with, update the tables in [`../README.md`](../README.md#results) and
[`../RESULTS.md`](../RESULTS.md) by hand, and note in your commit/PR that the
numbers came from your own re-run (with your hardware and date), not the
original one, so readers don't mistake a re-run's numbers for the original
run's.
