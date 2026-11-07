"""Search the object map with text.

The query is embedded with CLIP's text encoder and compared with every
object's image embedding (cosine similarity, since all vectors are unit length).

Raw similarity has a weakness: big generic regions like walls look a bit like
everything, so they creep into the results of vague queries. We therefore rank
by relevancy (from the LERF paper): each object is also compared with a few
generic "canonical" phrases, and

    relevancy = min over canonical phrases c of
                exp(s_query) / (exp(s_query) + exp(s_c))

i.e. "does this object match the query better than it matches 'stuff'?".
An object that matches everything also matches "stuff", so it scores low.
0.5 means no preference; higher is a better match.

Usage:
    python -m osm.query "something to sit on"
    python -m osm.query "a mug" "a red ball" --top 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from osm.clip_features import Clip

# CLIP was trained on image captions, so a caption-like prompt works better
# than a bare word.
TEMPLATE = "a photo of {}."
CANONICAL = ["object", "things", "stuff", "texture"]


class Searcher:
    def __init__(self, ids: np.ndarray, features: np.ndarray, clip: Clip,
                 temperature: float = 10.0):
        self.ids = ids
        self.features = features
        self.clip = clip
        self.temperature = temperature
        canon = clip.text([TEMPLATE.format(c) for c in CANONICAL])
        self.canon_sims = features @ canon.T  # (objects, canonical phrases)

    def rank(self, query: str) -> list[tuple[int, float, float]]:
        """(object id, relevancy, raw similarity), best first."""
        text = self.clip.text([TEMPLATE.format(query)])[0]
        sims = self.features @ text
        q = np.exp(self.temperature * sims)[:, None]
        c = np.exp(self.temperature * self.canon_sims)
        relevancy = (q / (q + c)).min(axis=1)
        order = np.argsort(-relevancy)
        return [(int(self.ids[i]), float(relevancy[i]), float(sims[i])) for i in order]

    @classmethod
    def load(cls, out: Path) -> "Searcher":
        data = np.load(out / "features.npz")
        return cls(data["ids"], data["features"],
                   Clip(str(data["model"]), str(data["pretrained"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--out", type=Path, default=Path("outputs/room0"))
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    searcher = Searcher.load(args.out)
    objects = {o["id"]: o for o in
               json.loads((args.out / "objects.json").read_text())["objects"]}

    for q in args.queries:
        print(f'\n"{q}"')
        for oid, rel, sim in searcher.rank(q)[:args.top]:
            o = objects[oid]
            print(f"  relevancy {rel:.2f}  (sim {sim:.3f})  object {oid:2d}  at {o['center']}"
                  f"   truth: {o['gt_label']}")


if __name__ == "__main__":
    main()
