"""
build_test_set.py

Pulls a real, held-out sample from CodeSearchNet Python's own `test` split
(22,176 examples), which training never touches, training only reads the
`train` split (see ../configs/techdocs.yaml). Each example becomes one
query/document pair:

    query    = func_documentation_string   (the docstring, natural language)
    document = func_code_string            (the function body, what a real
                                             code search tool would return)

This is a real code-search retrieval task: given a docstring, find the
function it documents among a pool of distractor functions. Written to
test_set.json so every downstream eval script reads the exact same sample,
same seed, no redrawing.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

N_SAMPLES = 500
SEED = 42
OUT_PATH = Path(__file__).parent / "test_set.json"


def main() -> None:
    ds = load_dataset("code-search-net/code_search_net", "python", split="test")
    print(f"Loaded real CodeSearchNet Python test split: {len(ds)} examples")

    rng = random.Random(SEED)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    pairs = []
    seen_docs = set()
    for i in indices:
        row = ds[i]
        query = (row["func_documentation_string"] or "").strip()
        doc = (row["func_code_string"] or "").strip()
        if not query or not doc:
            continue
        if doc in seen_docs:
            continue  # keep the retrieval pool free of exact-duplicate functions
        seen_docs.add(doc)
        pairs.append({
            "id": len(pairs),
            "query": query,
            "document": doc,
            "repo": row.get("repository_name"),
            "func_name": row.get("func_name"),
        })
        if len(pairs) >= N_SAMPLES:
            break

    OUT_PATH.write_text(json.dumps({
        "source": "code-search-net/code_search_net",
        "config": "python",
        "split": "test",
        "seed": SEED,
        "n_pairs": len(pairs),
        "pairs": pairs,
    }, indent=2))
    print(f"Wrote {len(pairs)} real query/document pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
