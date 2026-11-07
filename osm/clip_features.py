"""Give every mapped object a CLIP embedding.

CLIP maps images and text into the same vector space: a picture of a sofa and
the words "a sofa" end up as nearby vectors. So if every object has an image
embedding, a text query can be answered by finding the nearest objects.

For each object we
    1. pick its best few views (largest, least cut off by the image border),
    2. make two crops of each view:
         context crop  the region plus a margin of surroundings, as photographed
         masked crop   the same square, but everything outside the object's
                       mask painted gray
    3. embed both with CLIP and add them,
    4. average over views into one vector for the object.

Why two crops: the context crop of a table also shows the chair next to it,
so on its own it drifts toward "chair". The masked crop shows only the table
but looks unnatural (CLIP rarely saw gray cut-outs). Together they are more
reliable than either alone; on the test room, object names found the right
object 6/9 times with context crops, 7/9 masked, 8/9 combined.

Averaging several views makes the vector more stable: one view of a chair
from behind can be ambiguous, five views from different angles are not.

Usage (after python -m osm.objects):
    python -m osm.clip_features --data data/room0 --out outputs/room0
Writes
    features.npz   object ids + one unit-length 512-d vector per object
    crops.png      the crops each object was embedded from (for checking)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from osm.geometry import Sequence

MODEL = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


class Clip:
    """Small wrapper: images or text in, unit-length vectors out."""

    def __init__(self, model: str = MODEL, pretrained: str = PRETRAINED):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)

    @torch.no_grad()
    def images(self, crops: list[np.ndarray]) -> np.ndarray:
        batch = torch.stack([self.preprocess(Image.fromarray(c)) for c in crops])
        return _normalize(self.model.encode_image(batch).numpy())

    @torch.no_grad()
    def text(self, prompts: list[str]) -> np.ndarray:
        return _normalize(self.model.encode_text(self.tokenizer(prompts)).numpy())


def _normalize(x: np.ndarray) -> np.ndarray:
    """Unit length, so a dot product between two vectors is their cosine similarity."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def best_views(observations: list[dict], width: int, height: int, k: int) -> list[dict]:
    """The k most useful sightings of an object.

    Bigger is better (more detail), but a view cut off by the image border only
    shows part of the object, so those count for half.
    """
    def score(ob: dict) -> float:
        u0, v0, u1, v1 = ob["bbox"]
        cut = u0 <= 0 or v0 <= 0 or u1 >= width - 1 or v1 >= height - 1
        return ob["pixels"] * (0.5 if cut else 1.0)
    return sorted(observations, key=score, reverse=True)[:k]


def crop(rgb: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.2,
         min_size: int = 48) -> np.ndarray:
    """Square crop around a bounding box, enlarged by `margin` for context.

    CLIP expects a square image, and a little surrounding context (the table
    under a mug) helps it recognize small objects.
    """
    u0, v0, u1, v1 = bbox
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    side = max(u1 - u0, v1 - v0, min_size) * (1 + 2 * margin)
    h, w = rgb.shape[:2]
    a0, a1 = int(max(cu - side / 2, 0)), int(min(cu + side / 2, w))
    b0, b1 = int(max(cv - side / 2, 0)), int(min(cv + side / 2, h))
    return rgb[b0:b1, a0:a1]


def masked_image(rgb: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray,
                 fill: int = 128) -> np.ndarray:
    """Copy of the frame with everything outside the object painted gray."""
    u0, v0, u1, v1 = bbox
    out = np.full_like(rgb, fill)
    region = rgb[v0:v1 + 1, u0:u1 + 1]
    out[v0:v1 + 1, u0:u1 + 1] = np.where(mask[..., None], region, fill)
    return out


def embed_objects(seq: Sequence, objects_json: dict, masks: dict, clip: Clip, k: int = 5
                  ) -> tuple[np.ndarray, np.ndarray, dict[int, list[np.ndarray]]]:
    ids, feats, all_crops = [], [], {}
    for obj in objects_json["objects"]:
        observations = objects_json["observations"][str(obj["id"])]
        for i, ob in enumerate(observations):
            ob["index"] = i  # to find its mask after sorting
        views = best_views(observations, seq.K.width, seq.K.height, k)

        context, masked = [], []
        for ob in views:
            rgb = seq[ob["frame"]].rgb
            mask = masks[f"{obj['id']}/{ob['index']}"]
            context.append(crop(rgb, ob["bbox"]))
            masked.append(crop(masked_image(rgb, ob["bbox"], mask), ob["bbox"]))

        emb = clip.images(context + masked)
        vec = emb[:len(views)].mean(axis=0) + emb[len(views):].mean(axis=0)
        ids.append(obj["id"])
        feats.append(vec / np.linalg.norm(vec))
        all_crops[obj["id"]] = context + masked
    return np.array(ids), np.stack(feats), all_crops


def crop_sheet(objects: list[dict], crops: dict[int, list[np.ndarray]], tile: int = 96) -> np.ndarray:
    """One row per object: its context crops, then its masked crops."""
    k = max(len(c) for c in crops.values())
    sheet = np.full((tile * len(objects), tile * k, 3), 255, np.uint8)
    for r, obj in enumerate(objects):
        for c, img in enumerate(crops[obj["id"]]):
            small = np.array(Image.fromarray(img).resize((tile, tile)))
            sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = small
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/room0"))
    parser.add_argument("--out", type=Path, default=Path("outputs/room0"))
    parser.add_argument("--views", type=int, default=5, help="crops per object")
    args = parser.parse_args()

    seq = Sequence(args.data)
    objects_json = json.loads((args.out / "objects.json").read_text())
    masks = dict(np.load(args.out / "masks.npz"))
    ids, feats, crops = embed_objects(seq, objects_json, masks, Clip(), args.views)
    np.savez(args.out / "features.npz", ids=ids, features=feats,
             model=MODEL, pretrained=PRETRAINED)
    iio.imwrite(args.out / "crops.png", crop_sheet(objects_json["objects"], crops))
    print(f"Embedded {len(ids)} objects -> {args.out / 'features.npz'}")


if __name__ == "__main__":
    main()
