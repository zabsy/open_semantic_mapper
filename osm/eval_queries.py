"""Score text search against simulator ground truth.

Each test query lists the real objects that count as a correct answer
("something to sit on" accepts the chair or the sofa). We report
    top-1   share of queries whose best result is a correct object
    top-3   share with a correct object among the first three
    MRR     mean reciprocal rank: 1 for rank 1, 1/2 for rank 2, ... averaged

These numbers let us compare changes (cropping, scoring, swapping in a learned
segmenter) instead of eyeballing a few queries.

Usage:
    python -m osm.eval_queries --out outputs/room0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from osm.query import Searcher

# query -> real object names that count as correct
QUERIES = {
    # names
    "a sofa": ["sofa"],
    "a mug": ["mug"],
    "a red ball": ["red_ball"],
    "a floor lamp": ["lamp"],
    "a television": ["tv"],
    "a potted plant": ["plant"],
    "a bookshelf": ["bookshelf"],
    "a wooden table": ["table"],
    "a chair": ["chair"],
    # synonyms
    "a couch": ["sofa"],
    "a coffee cup": ["mug"],
    "a TV screen": ["tv"],
    # attributes and functions
    "something to sit on": ["chair", "sofa"],
    "something green": ["plant"],
    "a light source": ["lamp"],
    "a toy": ["red_ball"],
    "somewhere to put my coffee": ["table"],
    "something to watch movies on": ["tv"],
}


def evaluate(searcher: Searcher, labels: dict[int, str], verbose: bool = True) -> dict:
    top1 = top3 = mrr = 0.0
    for query, correct in QUERIES.items():
        ranked = [labels[oid] for oid, _, _ in searcher.rank(query)]
        rank = next(i for i, name in enumerate(ranked, 1) if name in correct)
        top1 += rank == 1
        top3 += rank <= 3
        mrr += 1 / rank
        if verbose:
            mark = "ok " if rank == 1 else f"#{rank:<2d}"
            print(f"  {mark} {query:30s} -> {', '.join(ranked[:3])}")
    n = len(QUERIES)
    return {"top1": top1 / n, "top3": top3 / n, "mrr": mrr / n, "queries": n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/room0"))
    args = parser.parse_args()

    labels = {o["id"]: o["gt_label"] for o in
              json.loads((args.out / "objects.json").read_text())["objects"]}
    scores = evaluate(Searcher.load(args.out), labels)
    print(f"\ntop-1 {scores['top1']:.0%}   top-3 {scores['top3']:.0%}   "
          f"MRR {scores['mrr']:.2f}   ({scores['queries']} queries)")


if __name__ == "__main__":
    main()
