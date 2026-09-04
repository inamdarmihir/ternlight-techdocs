# ternlight-techdocs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10--3.12](https://img.shields.io/badge/python-3.10--3.12-blue)](.python-version)
[![Node >=18](https://img.shields.io/badge/node-%3E%3D18-339933)](ternlight-bridge/package.json)
[![Qdrant v1.19](https://img.shields.io/badge/qdrant-v1.19.0-dc244c)](https://github.com/qdrant/qdrant/releases/tag/v1.19.0)

Ternary quantization (BitNet-style, weights in `{-1, 0, +1}`) as a real middle
ground between static embeddings and full fp32 transformers — measured on a
real code-search task, indexed and queried through Qdrant, with a
domain-specialized [Ternlight](https://github.com/soycaporal/ternlight)
distillation benchmarked against the generic release and an fp32 baseline.

## Two findings, up front

This repo runs a real, 3-way retrieval benchmark (own techdocs-distilled
checkpoint vs. the published `@ternlight/base` npm package vs. FastEmbed's
`bge-small-en-v1.5`, all through real Qdrant collections). Two things came out
of it, and the second one is a negative result, reported as-is rather than
buried at the bottom:

1. **Ternary quantization is a real middle ground, not just an extreme.**
   Both ternary models land within 8-11 points of recall@1 (2-5 points at
   recall@10) of the full fp32 model — far closer than a
   [published Qdrant benchmark](https://github.com/Dylancouzon/static-embeddings)
   suggests a no-transformer, static-embedding model gets (0.2887 NDCG@10 on
   the same kind of code-search task, worse than plain BM25).
2. **⚠️ Domain-specific distillation did not beat the generic checkpoint
   here.** Our techdocs-tuned model is 18x higher document throughput and 12x
   lower query latency than the generic Ternlight release, but scores
   2-3 points *lower* on recall@1/5/10 — not higher, which was the actual
   hypothesis going in. Full explanation in
   ["What the results actually show"](#what-the-results-actually-show) below.

## The spectrum this fills in

| | Static embeddings | Ternary quantization (this repo) | Full fp32 transformer |
|---|---|---|---|
| Keeps the transformer (attention, etc.)? | No | **Yes** — same architecture, weights quantized to 3 values | Yes, full precision |
| Tested here as | *(reference only — from the static-embeddings benchmark, not re-run in this repo)* | `@ternlight/base` (generic) **and** our techdocs-QAT checkpoint | `bge-small-en-v1.5` via FastEmbed |
| Quality (code search) | 0.2887 NDCG@10 — worse than BM25's 0.2955 | recall@1 0.872-0.902 | recall@1 0.982 |
| Speed | ~9,100 docs/sec (model2vec-style) | 19.3-357.5 docs/sec | 1.7 docs/sec |

A [published Qdrant benchmark](https://github.com/Dylancouzon/static-embeddings)
tested the two ends of this spectrum — full fp32 transformer, and no
transformer at all — and found static embeddings win on speed but lose on
quality everywhere tested, including code search. It didn't test the middle:
a model that keeps the transformer architecture but quantizes its weights to
three values, which is what ternary quantization (BitNet-style, as used in
Ternlight) actually is, and a different tradeoff than removing the
transformer entirely. This repo builds and measures that middle point for
one concrete, practical use case: given a docstring, find the function it
documents.

(The static-embeddings NDCG@10 numbers above are on a different protocol —
BEIR-style exact search against qrels — not directly comparable point-for-point
to this repo's recall@k on a held-out matching task. The qualitative
comparison, keeping the transformer vs. removing it, still holds; see
["What the results actually show"](#what-the-results-actually-show) for the
full caveat.)

## What's in this repo

Three original contributions, layered on top of the real, public Ternlight
training code (not vendored here — see [Prerequisites](#prerequisites)):

- [`configs/`](configs/) — three YAML configs (`techdocs.yaml`,
  `techdocs-fp32.yaml`, `techdocs-qat.yaml`) that point Ternlight's own
  distillation pipeline at CodeSearchNet Python instead of its default
  general-purpose corpus.
- [`eval/`](eval/) — a real 3-way retrieval benchmark: our checkpoint vs. the
  actual published `@ternlight/base` npm package vs. FastEmbed's
  `bge-small-en-v1.5`, all indexed and queried through real Qdrant
  collections. See [`eval/README.md`](eval/README.md) for exact regeneration
  commands.
- [`ternlight-bridge/`](ternlight-bridge/) — a small Node.js bridge
  (`embed_batch.mjs`) so the eval script can call the real, published
  Ternlight package directly, not a Python reimplementation of it.
- [`Makefile`](Makefile) — thin wrappers around every command in this README,
  for a one-command-per-step reproduction path (`make help` for the list).

## Method

**Data.** [CodeSearchNet Python](https://huggingface.co/datasets/code-search-net/code_search_net),
25,000 samples (docstring → function pairs), split 22,215 train / 1,234 val /
1,234 test. This is a deliberate scope-down from Ternlight's own "rigorous"
1M-sample training tier to a realistic single-session wall-clock budget — see
[Limitations](#limitations).

**Architecture.** 2 transformer layers, 4 attention heads, `d_model=256`,
`ffn_dim=1024`, 384-dim output, 9,490,816 parameters. Same architecture for
both training phases below; only the weight representation differs.

**Training.** Two phases, both 30 epochs, batch size 64, teacher =
`sentence-transformers/all-MiniLM-L6-v2`, on Apple Silicon (`mps`):

1. **fp32 baseline** (`techdocs-fp32.yaml`) — standard distillation,
   full-precision weights. Final epoch: `loss=0.1692 cos=0.8308
   val_spearman=0.6989`.
2. **QAT** (`techdocs-qat.yaml`) — quantization-aware training, `BitLinear`
   layers (ternary weights, `lambda=1`), 5-epoch QAT warmup, contrastive loss
   weight 0.15. Final epoch: `loss=0.1953 cos=0.8087 val_spearman=0.6894`.

Full per-epoch logs for both phases are in
[`full_training_log.txt`](full_training_log.txt).

**Evaluation.** A real held-out retrieval task
([`eval/build_test_set.py`](eval/build_test_set.py)): 500 query/document pairs
sampled (seed 42) from CodeSearchNet Python's own `test` split, which training
never touches. Query = docstring, document = function body, 500 real
distractors per query. Each of the three embedders gets its own Qdrant
collection (cosine distance, all vectors 384-dim so the comparison isn't
confounded by size); recall@{1,5,10} measures whether the correct function
lands in the top-k for its own docstring.

## Results

Clean run, no other CPU-heavy process running (see
["Verified, not just written"](#verified-not-just-written) for why that
qualifier matters), MPS for our checkpoint, CPU for the other two, real Qdrant
server (`qdrant/qdrant:v1.19.0`, Docker, default settings):

| Embedder | Params | Weights | recall@1 | recall@5 | recall@10 | Doc embed | Query latency |
|---|---:|---|---:|---:|---:|---:|---:|
| **Ours** (techdocs QAT) | 9.5M | ternary | 0.872 | 0.948 | 0.964 | 357.5 docs/sec | 1.64 ms/query |
| Generic `@ternlight/base` | — | ternary | 0.902 | 0.966 | 0.974 | 19.3 docs/sec | 20.34 ms/query |
| FastEmbed `bge-small-en-v1.5` | 33M | fp32 | 0.982 | 0.992 | 0.996 | 1.7 docs/sec | 410.66 ms/query |

Raw run output: [`eval/clean_eval_run.log`](eval/clean_eval_run.log).
Machine-readable results: [`eval/results.json`](eval/results.json). A
standalone summary of this table plus the narrative below lives in
[`RESULTS.md`](RESULTS.md); regeneration commands are in
[`eval/README.md`](eval/README.md).

## What the results actually show

Two findings, and the second one wasn't the expected outcome, so it's stated
plainly rather than smoothed over.

**1. Ternary quantization is a real middle ground, not just an extreme.**
Both ternary models land far closer to the full fp32 model's recall (within
8-11 points at k=1, within 2-5 points at k=10) than the static-embeddings
benchmark's numbers suggest a no-transformer model would. FastEmbed's
`bge-small` is the clear quality leader here, but at roughly 200x the
per-query latency of our checkpoint and 25x the latency of the generic
ternary model. If a use case can tolerate an 8-11 point recall@1 gap, ternary
quantization buys a large, real speed advantage that removing the transformer
entirely does not have to pay a BM25-losing quality price for. (Note: the
static-embeddings numbers above are NDCG@10 on a different protocol —
BEIR-style exact search against qrels — not directly comparable point for
point to this repo's recall@k on a held-out matching task. The qualitative
comparison, keeping the transformer vs. removing it, still holds.)

**2. Domain-specific distillation did not beat the generic checkpoint here.**
Our techdocs-tuned model is meaningfully faster than the generic Ternlight
package (18x higher document throughput, 12x lower query latency) but scores
2-3 points lower on recall@1/5/10, not higher. Ternary quantization plus a
much smaller training run (25k domain-specific samples vs. whatever the
generic release was trained on, architecture and corpus scale both unknown to
us) evidently trades some of the generic model's quality for speed, rather
than trading generic quality for domain quality as hoped going in. This is
the honest result of this specific run, not the intended headline finding,
and it's a real limitation to disclose rather than a negative result to hide:
at this data and epoch scale, specialization did not close the gap to the
generic model.

## Verified, not just written

Every number above came from an actual run against a real, named dataset and
a real local Qdrant server — nothing here is estimated or reconstructed from
a formula. Reproduction commands are in [Quickstart](#quickstart-reproducing-this-end-to-end)
and [`eval/README.md`](eval/README.md).

**A resource-contention bug is part of the record, not hidden from it.** An
earlier validation pass ran the eval harness concurrently with the QAT
training job on the same machine, to confirm the eval pipeline itself worked
correctly before training finished. It did (recall numbers matched the clean
run almost exactly), but the FastEmbed timing from that run was unusable: 0.3
docs/sec and 2,516 ms/query, roughly 175x slower than the clean run's 1.7
docs/sec and 410 ms. FastEmbed's ONNX runtime was fighting the training
process for CPU. The results table above comes only from a run made after
training's process had fully exited (`ps aux` checked, not just log output),
with nothing else CPU-heavy running alongside it. Anyone reproducing this
should do the same, or their FastEmbed column will look far worse than it
actually is.

**FastEmbed's absolute numbers are still slower than commonly published
benchmarks for `bge-small`** (typically tens of docs/sec on CPU, not 1.7).
This run used FastEmbed's default settings with no explicit thread-count
tuning, on one specific machine, with no other optimization pass applied.
That's reported as-is rather than adjusted toward an expected number: it's
this run's real result, on this hardware, with this configuration, and it
should be treated as a lower bound on what a tuned FastEmbed deployment could
do, not as a FastEmbed-in-general number.

**What's verified vs. estimated, at a glance:**

| Claim | Status |
|---|---|
| recall@{1,5,10} for all three embedders | Verified — real Qdrant queries against a real held-out test set |
| Doc embed throughput / query latency, this run | Verified, but hardware- and load-specific (see above) — not a general benchmark of any of the three embedders |
| "Ternary is a real middle ground" (qualitative) | Verified on this task; expected to generalize based on the mechanism (transformer kept vs. removed), not separately tested on other tasks |
| "Specialization didn't win here" | Verified for this run, this data/epoch scale — explicitly *not* claimed to generalize (see [Limitations](#limitations)) |
| Absolute FastEmbed/generic-Ternlight numbers as general benchmarks of those tools | **Not** verified as general numbers — this run's numbers on this machine, nothing more |
| Recall/latency on hardware other than the one this ran on | Not measured — see [Apple Silicon vs. CPU](#apple-silicon-mps-vs-cpu) |

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10-3.12 (3.11 recommended, see [`.python-version`](.python-version)) | matches the `torch>=2.0` / `transformers>=4.40` support matrix |
| Node.js | ≥ 18 | required by [`@ternlight/base`](https://www.npmjs.com/package/@ternlight/base); tested with Node 20/22 |
| npm | bundled with Node | the committed [`package-lock.json`](ternlight-bridge/package-lock.json) pins `@ternlight/base` to an exact resolved version — use `npm ci`, not `npm install`, for a deterministic bridge install |
| Qdrant | [`qdrant/qdrant:v1.19.0`](https://github.com/qdrant/qdrant/releases/tag/v1.19.0) (Docker) | pinned to match `qdrant-client>=1.19` in [`requirements.txt`](requirements.txt); the eval harness talks to it over `localhost:6333` |
| Git | any recent version | to clone the upstream Ternlight training code |
| Disk | ~1-2 GB | HF dataset cache + two ~109 MB checkpoints + venv/node_modules, none of which are committed (see [What's not included](#whats-not-included-and-why)) |

### Apple Silicon (MPS) vs. CPU

- Every latency/throughput number in the results table above was produced
  with our checkpoint on Apple Silicon's `mps` backend, and the two baseline
  embedders on CPU — see ["Verified, not just written"](#verified-not-just-written)
  for why FastEmbed's CPU numbers look the way they do.
- Ternlight's own `train.py` auto-selects the fastest available device, in
  this order: `cuda` → `mps` → `cpu` (`best_device()` in
  [`train.py`](https://github.com/soycaporal/ternlight/blob/main/training/distill/train.py)) —
  no flag needed. On a machine with neither an Apple GPU nor a CUDA GPU, both
  training phases will fall back to CPU and take substantially longer than
  the ~50 minutes reported here.
- [`eval/eval_retrieval.py`](eval/eval_retrieval.py)'s `embed_ours()` picks
  `mps` if available, else `cpu` — it does not check for CUDA. Reproducing on
  an NVIDIA machine will give CPU-speed numbers for the "ours" row unless you
  adapt that one line yourself.
- None of this changes recall — only wall-clock throughput/latency. Treat
  every latency/throughput number in this repo as *this run's hardware*
  numbers, not universal ones.

## Quickstart: reproducing this end to end

Two tracks: a **fast path** that validates the whole harness in a few minutes
without training anything, and the **full path** that reproduces the actual
results table (~50 minutes of training + ~10 minutes of eval on Apple
Silicon, longer on CPU-only machines — see
[Apple Silicon vs. CPU](#apple-silicon-mps-vs-cpu)).

### 0. One-time setup

```bash
git clone https://github.com/inamdarmihir/ternlight-techdocs
cd ternlight-techdocs
make setup        # clones upstream/, creates venv/, installs Python + Node deps
```

`make setup` = the manual steps below; use whichever you prefer.
`make help` lists every target, and [`eval/README.md`](eval/README.md) has
the full flag reference.

Manual equivalent:

1. Clone the upstream Ternlight training code (not vendored into this repo —
   it's a large, actively developed project with its own license; better to
   clone it fresh than ship a stale copy):

   ```bash
   git clone https://github.com/soycaporal/ternlight upstream
   ```

2. Create a virtualenv and install both requirement files (this repo's own
   eval dependencies, plus Ternlight's training dependencies):

   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r upstream/training/distill/requirements.txt
   pip install -r requirements.txt
   ```

3. Install the Node bridge's exact, lockfile-pinned dependency. **Do this
   before running any eval step** — `eval_retrieval.py` shells out to this
   directory for the generic-Ternlight leg of the benchmark, and skipping
   this step is the single most common way to get a confusing subprocess
   failure:

   ```bash
   cd ternlight-bridge && npm ci && cd ..
   ```

4. Start a pinned Qdrant server:

   ```bash
   docker run -p 6333:6333 qdrant/qdrant:v1.19.0
   ```

### Fast path: validate the harness without training anything (~10 minutes)

5. Run the eval harness with `--skip-ours`. This exercises the real Qdrant
   collections, the real generic `@ternlight/base` bridge, and the real
   FastEmbed baseline — skipping only the leg that needs a checkpoint you
   haven't trained yet:

   ```bash
   cd eval
   python3 eval_retrieval.py --skip-ours
   # or: make eval-quick
   ```

   This is the fastest way to confirm your environment (Qdrant reachable,
   Node bridge installed, FastEmbed downloading its ONNX model correctly)
   before committing to a full training run.

### Full path: reproduce the results table (~60 minutes total)

6. Copy this repo's configs into the upstream training pipeline and prep the
   data:

   ```bash
   cp configs/techdocs*.yaml upstream/training/distill/configs/
   cd upstream/training/distill
   python3 prep/prepare.py --config configs/techdocs.yaml
   ```

7. Train both phases — fp32 baseline, then QAT. Each is ~25 minutes on
   Apple Silicon; longer on CPU-only machines (see
   [Apple Silicon vs. CPU](#apple-silicon-mps-vs-cpu)):

   ```bash
   python3 train.py --config configs/techdocs-fp32.yaml
   python3 train.py --config configs/techdocs-qat.yaml
   ```

8. Run the full eval, pointing at the QAT checkpoint the run above produced.
   The `<run-id>` segment is the short git commit hash of your `upstream`
   clone at train time (see `git_commit()` in `train.py`), so it will differ
   from `23da804` in the original run — check `runs/` for the exact
   directory name. **Wait for the training process to fully exit first**
   (`ps aux`, not just the log — see
   ["Verified, not just written"](#verified-not-just-written) for why
   running the eval concurrently with training corrupts the FastEmbed timing
   column):

   ```bash
   cd ../../../eval
   python3 eval_retrieval.py --ckpt ../upstream/training/distill/runs/techdocs-qat-<run-id>/checkpoint_ep30.pt
   # or: make eval   (auto-discovers the checkpoint path above)
   ```

   This overwrites `eval/results.json` and `eval/clean_eval_run.log` with
   *your* run's numbers — expect them to differ somewhat from the ones in
   this README (recall should be close on the same data/seed; latency will
   depend entirely on your hardware, per
   [Apple Silicon vs. CPU](#apple-silicon-mps-vs-cpu)).

See [`eval/README.md`](eval/README.md) for every CLI flag, expected per-embedder
runtime, and troubleshooting notes.

## Using the trained checkpoint directly

Everything above trains and evaluates the model. To just embed your own text
with the resulting checkpoint, outside the eval harness, the whole call is
`load_for_eval` (Ternlight's own real loading path, `evaluation.py` in its
training repo) plus a standard `transformers` tokenizer, the exact pattern
[`eval/eval_retrieval.py`](eval/eval_retrieval.py)'s `embed_ours()` uses:

```python
import sys
sys.path.insert(0, "upstream/training/distill")  # after `git clone` per Quickstart above

import torch
from evaluation import load_for_eval
from transformers import AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT = "upstream/training/distill/runs/techdocs-qat-<run-id>/checkpoint_ep30.pt"

em = load_for_eval(CKPT, device, embedding_format="ternary")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = ["parses a JSON config file and returns a validated dict"]
toks = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
with torch.no_grad():
    vectors = em.model(toks["input_ids"].to(device), toks["attention_mask"].to(device))
# vectors: (len(texts), 384) tensor, ready to upsert into a Qdrant collection
# (cosine distance, size=384) the same way build_collection() in eval_retrieval.py does.
```

This needs a real trained checkpoint on disk first, either your own run of
the setup + training steps in [Quickstart](#quickstart-reproducing-this-end-to-end)
above (~50 minutes, no eval step required to just get a checkpoint), or the
generic, already-published `@ternlight/base`
package instead if you don't need the techdocs-specific tuning, callable
straight from Node via
[`ternlight-bridge/embed_batch.mjs`](ternlight-bridge/embed_batch.mjs)
without training anything.

## Adapting this to your own corpus

The three configs in [`configs/`](configs/) are the only place domain-specific
choices live; nothing else needs to change to point this pipeline at a
different corpus.

1. **Swap the data source in `techdocs.yaml`.** `data.sources` takes any
   dataset loadable via `datasets.load_dataset`, plus which field is the
   query text (`text_field`) — for a different code-docs task this might be
   docstrings from another language, API reference text, or your own
   internal doc corpus loaded from local files instead of the Hub.
2. **Keep the teacher fixed, or change it deliberately.**
   `teacher_id: sentence-transformers/all-MiniLM-L6-v2` sets the ceiling this
   student distills toward — swapping it changes what "good" means for your
   run, so treat it as a real experimental variable, not a default to leave
   alone.
3. **`total_samples`, `epochs`, and `qat_warmup_epochs` are wall-clock knobs,
   not correctness knobs.** This repo used 25k samples / 30 epochs to fit a
   realistic single-session budget (Ternlight's own "rigorous" tier is 1M
   samples — see [Limitations](#limitations)). Scale these up if you have the
   time budget; the qualitative comparison (ternary vs. fp32 vs. static)
   should hold, the exact recall numbers will move.
4. **Architecture (`d_model`, `n_layers`, `n_heads`, `ffn_dim`, `output_dim`)
   is a size/quality tradeoff independent of the corpus.** Leave it alone
   for a first run; revisit it once you have a baseline recall number on
   your own data to compare against.
5. **Rebuild the eval set from your corpus's own held-out split**, not
   training data — copy the pattern in
   [`eval/build_test_set.py`](eval/build_test_set.py) (query/document pairing,
   dedup on document text, fixed seed) against your own `test` split so the
   eval never touches what training saw.
6. **Everything downstream is corpus-agnostic**: `eval_retrieval.py`,
   `embed_batch.mjs`, and the Qdrant collection setup only assume a
   query/document pair list and a 384-dim output — no code changes needed
   there for a different corpus, only the configs and the held-out set.

## What's not included, and why

- **Trained checkpoints (`.pt` files, ~109 MB each).** Over GitHub's un-LFS'd
  file size limit, and fully reproducible in about 50 minutes total (prep +
  both training phases) from the [Quickstart](#quickstart-reproducing-this-end-to-end)
  commands. Shipping them would mean maintaining a second, binary source of
  truth alongside the reproducible one.
- **The vendored `upstream/` Ternlight clone.** It's a real, independently
  maintained, MIT-licensed project; this repo depends on it rather than
  republishing a snapshot of it.
- **`node_modules/`, `venv/`, training `cache/`, `runs/`.** Regenerated by
  the commands above (`make setup` / `make train`); committing them would
  just be committing other projects' build output.

## Limitations

- 25,000 training samples and 30 epochs per phase, chosen for realistic
  single-session wall-clock time, not Ternlight's own documented "rigorous"
  1M-sample tier. The qualitative comparison (ternary vs. fp32 vs. static)
  should hold at larger scale; the exact recall numbers likely would not.
- Single dataset (CodeSearchNet Python), single language, single query style
  (docstring-to-function). Generalization to other technical-doc retrieval
  tasks (API references, multi-language codebases, longer documents) is
  untested. Recall numbers are from a single 500-pair held-out sample; no
  confidence intervals or repeated runs.
- The specialization result above (finding 2) is a single run at this
  specific data/epoch scale. It should not be read as "distillation never
  helps," only as "it didn't help at this scale, on this task, in this run."
- Latency/throughput numbers are specific to the machine this ran on (Apple
  Silicon for our checkpoint, CPU for the two baselines) — see
  [Apple Silicon vs. CPU](#apple-silicon-mps-vs-cpu). They are not general
  benchmarks of any of the three tools involved.

## Citation

If you use this repo's configs, eval harness, or results, a link back to this
repository is appreciated. Ternlight itself should be cited separately:
[github.com/soycaporal/ternlight](https://github.com/soycaporal/ternlight).

## License

MIT. See [LICENSE](LICENSE).
