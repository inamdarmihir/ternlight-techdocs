# ternlight-techdocs

A domain-specialized distillation of [Ternlight](https://github.com/soycaporal/ternlight)
(a ternary-quantized, BitLinear transformer embedder, weights in `{-1, 0, +1}`) for
technical documentation retrieval, benchmarked against the generic Ternlight release
and a full fp32 transformer on a real, held-out code-search task, indexed and queried
through Qdrant.

## Why this exists

A published Qdrant benchmark ([static-embeddings](https://github.com/Dylancouzon/static-embeddings))
tested what happens when you remove the transformer entirely and use static,
model2vec-style embeddings instead. The finding was clear: static wins on speed
(about 9,100 docs/sec vs. 31 docs/sec for `bge-small` ONNX at batch 32) and loses on
quality everywhere it was tested, including code search, where it scored 0.2887
NDCG@10 against `bge-small`'s 0.6742 on CodeSearchNet Python — worse even than plain
BM25 (0.2955).

That benchmark tested two ends of a spectrum: full fp32 transformer, and no
transformer at all. It didn't test the middle: a model that keeps the transformer
architecture, attention and all, but quantizes its weights to three values. That's
what ternary quantization (BitNet-style, as used in Ternlight) actually is, and it's
a different tradeoff than removing the transformer. This repo builds and measures
that middle point for one concrete, practical use case: given a docstring, find the
function it documents.

## What's in this repo

Three original contributions, layered on top of the real, public Ternlight training
code (not vendored here — see "Reproducing" below):

- `configs/` — three YAML configs (`techdocs.yaml`, `techdocs-fp32.yaml`,
  `techdocs-qat.yaml`) that point Ternlight's own distillation pipeline at
  CodeSearchNet Python instead of its default general-purpose corpus.
- `eval/` — a real 3-way retrieval benchmark: our checkpoint vs. the actual published
  `@ternlight/base` npm package vs. FastEmbed's `bge-small-en-v1.5`, all indexed and
  queried through real Qdrant collections.
- `ternlight-bridge/` — a small Node.js bridge (`embed_batch.mjs`) so the eval script
  can call the real, published Ternlight package directly, not a Python
  reimplementation of it.

## Method

**Data.** [CodeSearchNet Python](https://huggingface.co/datasets/code-search-net/code_search_net),
25,000 samples (docstring → function pairs), split 22,215 train / 1,234 val / 1,234
test. This is a deliberate scope-down from Ternlight's own "rigorous" 1M-sample
training tier to a realistic single-session wall-clock budget — see Limitations.

**Architecture.** 2 transformer layers, 4 attention heads, `d_model=256`,
`ffn_dim=1024`, 384-dim output, 9,490,816 parameters. Same architecture for both
training phases below; only the weight representation differs.

**Training.** Two phases, both 30 epochs, batch size 64, teacher =
`sentence-transformers/all-MiniLM-L6-v2`, on Apple Silicon (`mps`):

1. **fp32 baseline** (`techdocs-fp32.yaml`) — standard distillation, full-precision
   weights. Final epoch: `loss=0.1692 cos=0.8308 val_spearman=0.6989`.
2. **QAT** (`techdocs-qat.yaml`) — quantization-aware training, `BitLinear` layers
   (ternary weights, `lambda=1`), 5-epoch QAT warmup, contrastive loss weight 0.15.
   Final epoch: `loss=0.1953 cos=0.8087 val_spearman=0.6894`.

Full per-epoch logs for both phases are in [`full_training_log.txt`](full_training_log.txt).

**Evaluation.** A real held-out retrieval task ([`eval/build_test_set.py`](eval/build_test_set.py)):
500 query/document pairs sampled (seed 42) from CodeSearchNet Python's own `test`
split, which training never touches. Query = docstring, document = function body,
500 real distractors per query. Each of the three embedders gets its own Qdrant
collection (cosine distance, all vectors 384-dim so the comparison isn't confounded
by size); recall@{1,5,10} measures whether the correct function lands in the top-k
for its own docstring.

## Results

Clean run, no other CPU-heavy process running (see "Verified, not just written"
below for why that qualifier matters), MPS for our checkpoint, CPU for the other
two, real Qdrant server (`qdrant/qdrant`, Docker, default settings):

| Embedder | Params | Weights | recall@1 | recall@5 | recall@10 | Doc embed | Query latency |
|---|---:|---|---:|---:|---:|---:|---:|
| **Ours** (techdocs QAT) | 9.5M | ternary | 0.872 | 0.948 | 0.964 | 357.5 docs/sec | 1.64 ms/query |
| Generic `@ternlight/base` | — | ternary | 0.902 | 0.966 | 0.974 | 19.3 docs/sec | 20.34 ms/query |
| FastEmbed `bge-small-en-v1.5` | 33M | fp32 | 0.982 | 0.992 | 0.996 | 1.7 docs/sec | 410.66 ms/query |

Raw run output: [`eval/clean_eval_run.log`](eval/clean_eval_run.log). Machine-readable
results: [`eval/results.json`](eval/results.json).

## What the results actually show

Two findings, and the second one wasn't the expected outcome, so it's stated
plainly rather than smoothed over.

**1. Ternary quantization is a real middle ground, not just an extreme.** Both
ternary models land far closer to the full fp32 model's recall (within 8-11 points
at k=1, within 2-5 points at k=10) than the static-embeddings benchmark's numbers
suggest a no-transformer model would. FastEmbed's `bge-small` is the clear
quality leader here, but at roughly 200x the per-query latency of our checkpoint
and 25x the latency of the generic ternary model. If a use case can tolerate an
8-11 point recall@1 gap, ternary quantization buys a large, real speed advantage
that removing the transformer entirely does not have to pay a BM25-losing quality
price for. (Note: the static-embeddings numbers above are NDCG@10 on a different
protocol — BEIR-style exact search against qrels — not directly comparable point
for point to this repo's recall@k on a held-out matching task. The qualitative
comparison, keeping the transformer vs. removing it, still holds.)

**2. Domain-specific distillation did not beat the generic checkpoint here.** Our
techdocs-tuned model is meaningfully faster than the generic Ternlight package
(18x higher document throughput, 12x lower query latency) but scores 2-3 points
lower on recall@1/5/10, not higher. Ternary quantization plus a much smaller
training run (25k domain-specific samples vs. whatever the generic release was
trained on, architecture and corpus scale both unknown to us) evidently trades
some of the generic model's quality for speed, rather than trading generic
quality for domain quality as hoped going in. This is the honest result of this
specific run, not the intended headline finding, and it's a real limitation to
disclose rather than a negative result to hide: at this data and epoch scale,
specialization did not close the gap to the generic model.

## Verified, not just written

Every number above came from an actual run against a real, named dataset and a
real local Qdrant server — nothing here is estimated or reconstructed from a
formula. Reproduction commands are below.

**A resource-contention bug is part of the record, not hidden from it.** An
earlier validation pass ran the eval harness concurrently with the QAT training
job on the same machine, to confirm the eval pipeline itself worked correctly
before training finished. It did (recall numbers matched the clean run almost
exactly), but the FastEmbed timing from that run was unusable: 0.3 docs/sec and
2,516 ms/query, roughly 175x slower than the clean run's 1.7 docs/sec and 410 ms.
FastEmbed's ONNX runtime was fighting the training process for CPU. The results
table above comes only from a run made after training's process had fully exited
(`ps aux` checked, not just log output), with nothing else CPU-heavy running
alongside it. Anyone reproducing this should do the same, or their FastEmbed
column will look far worse than it actually is.

**FastEmbed's absolute numbers are still slower than commonly published
benchmarks for `bge-small`** (typically tens of docs/sec on CPU, not 1.7). This
run used FastEmbed's default settings with no explicit thread-count tuning, on
one specific machine, with no other optimization pass applied. That's reported
as-is rather than adjusted toward an expected number: it's this run's real
result, on this hardware, with this configuration, and it should be treated as a
lower bound on what a tuned FastEmbed deployment could do, not as a
FastEmbed-in-general number.

## Reproducing this

Training and evaluation both depend on Ternlight's own real distillation code,
which is not vendored into this repo (it's a large, actively developed project
with its own license; better to clone it fresh than ship a stale copy):

```bash
git clone https://github.com/soycaporal/ternlight upstream
python3 -m venv venv && source venv/bin/activate
pip install -r upstream/training/distill/requirements.txt
pip install -r requirements.txt   # this repo's own eval dependencies

cp configs/techdocs*.yaml upstream/training/distill/configs/
cd upstream/training/distill

python3 prep/prepare.py --config configs/techdocs.yaml
python3 train.py --config configs/techdocs-fp32.yaml
python3 train.py --config configs/techdocs-qat.yaml
```

Then, with a local Qdrant server running (`docker run -p 6333:6333 qdrant/qdrant`)
and the training process fully exited:

```bash
cd ../../../eval
python3 eval_retrieval.py --ckpt ../upstream/training/distill/runs/techdocs-qat-<run-id>/checkpoint_ep30.pt
```

The generic-Ternlight leg of the eval calls the real npm package via
`ternlight-bridge/`:

```bash
cd ternlight-bridge && npm install
```

## Using the trained checkpoint directly

Everything above trains and evaluates the model. To just embed your own
text with the resulting checkpoint, outside the eval harness, the whole
call is `load_for_eval` (Ternlight's own real loading path, `evaluation.py`
in its training repo) plus a standard `transformers` tokenizer, the exact
pattern [`eval/eval_retrieval.py`](eval/eval_retrieval.py)'s `embed_ours()`
uses:

```python
import sys
sys.path.insert(0, "upstream/training/distill")  # after `git clone` per "Reproducing this" above

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
the "Reproducing this" commands above (~50 minutes total), or the generic,
already-published `@ternlight/base` package instead if you don't need the
techdocs-specific tuning, callable straight from Node via
[`ternlight-bridge/embed_batch.mjs`](ternlight-bridge/embed_batch.mjs)
without training anything.

## What's not included, and why

- **Trained checkpoints (`.pt` files, ~109 MB each).** Over GitHub's un-LFS'd file
  size limit, and fully reproducible in about 50 minutes total (prep + both
  training phases) from the commands above. Shipping them would mean maintaining
  a second, binary source of truth alongside the reproducible one.
- **The vendored `upstream/` Ternlight clone.** It's a real, independently
  maintained, MIT-licensed project; this repo depends on it rather than
  republishing a snapshot of it.
- **`node_modules/`, `venv/`, training `cache/`.** Regenerated by the commands
  above; committing them would just be committing other projects' build output.

## Limitations

- 25,000 training samples and 30 epochs per phase, chosen for realistic
  single-session wall-clock time, not Ternlight's own documented "rigorous"
  1M-sample tier. The qualitative comparison (ternary vs. fp32 vs. static) should
  hold at larger scale; the exact recall numbers likely would not.
- Single dataset (CodeSearchNet Python), single language, single query style
  (docstring-to-function). Generalization to other technical-doc retrieval tasks
  (API references, multi-language codebases, longer documents) is untested.
  Recall numbers are from a single 500-pair held-out sample; no confidence
  intervals or repeated runs.
- The specialization result above (finding 2) is a single run at this specific
  data/epoch scale. It should not be read as "distillation never helps," only as
  "it didn't help at this scale, on this task, in this run."

## Citation

If you use this repo's configs, eval harness, or results, a link back to this
repository is appreciated. Ternlight itself should be cited separately:
[github.com/soycaporal/ternlight](https://github.com/soycaporal/ternlight).

## License

MIT. See [LICENSE](LICENSE).
