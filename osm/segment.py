"""Class-agnostic segmentation with SAM (Segment Anything).

SAM is promptable: give it a point and it returns masks of the thing under
that point. To segment a whole frame we prompt it with a grid of points.

    image ──encoder (once, ~2 s)──► embedding ──decoder (per point, ~70 ms)──► masks

The decoder runs once per point, so points are the cost. We use two passes:
    1. a coarse 8x8 grid,
    2. a finer 16x16 grid, but only points not already inside a pass-1 mask.
Big surfaces (floor, walls) are covered by pass 1, so pass 2 only spends
points where something is still unexplained, typically small objects.

Each point yields up to 3 candidate masks (SAM is unsure whether you meant a
part, the object, or a group); we keep the one SAM rates best, then drop
masks that are low quality, unstable, or duplicates. Finally the overlapping
masks are flattened into one label image, smaller masks painted on top, so a
mug stays a mug instead of being swallowed by the table mask under it.

The output has the same format as the simulator's ground-truth segmentation
(one int16 id per pixel, -1 = unlabeled), so the rest of the pipeline can use
either.

Usage:
    python -m osm.segment --data data/room0
Writes data/room0/sam/NNNN.npy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from osm.geometry import Sequence

MODEL = "Zigeng/SlimSAM-uniform-77"  # SAM distilled to ~1/13 the size, CPU friendly


class SamSegmenter:
    def __init__(self, model: str = MODEL, coarse: int = 8, fine: int = 16,
                 min_iou: float = 0.8, min_stability: float = 0.85,
                 nms_iou: float = 0.7, min_area: int = 100):
        """
        coarse, fine   point grid sizes of the two passes
        min_iou        SAM's own estimate of mask quality must reach this
        min_stability  mask must barely change when the threshold moves
                       (crisp edges, not a fuzzy blob)
        nms_iou        masks overlapping more than this are duplicates
        min_area       smaller final regions are dropped (pixels)
        """
        from transformers import SamModel, SamProcessor
        self.model = SamModel.from_pretrained(model).eval()
        self.processor = SamProcessor.from_pretrained(model)
        self.coarse, self.fine = coarse, fine
        self.min_iou, self.min_stability = min_iou, min_stability
        self.nms_iou, self.min_area = nms_iou, min_area

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """RGB (H, W, 3) -> label image (H, W) int16, -1 = unlabeled."""
        h, w = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt")
        embedding = self.model.get_image_embeddings(inputs["pixel_values"])

        masks, scores = self._decode(rgb, embedding, inputs, _grid(self.coarse, w, h))
        covered = np.any(masks, axis=0) if len(masks) else np.zeros((h, w), bool)
        fine = [p for p in _grid(self.fine, w, h) if not covered[int(p[1]), int(p[0])]]
        if fine:
            m2, s2 = self._decode(rgb, embedding, inputs, fine)
            masks, scores = np.concatenate([masks, m2]), np.concatenate([scores, s2])

        keep = _nms(masks, scores, self.nms_iou)
        return _flatten(masks[keep], self.min_area)

    def _decode(self, rgb, embedding, inputs, points) -> tuple[np.ndarray, np.ndarray]:
        """Prompt with each point; return its best surviving mask and score."""
        p = self.processor(images=rgb, input_points=[[[list(q)] for q in points]],
                           return_tensors="pt")
        out = self.model(image_embeddings=embedding, input_points=p["input_points"],
                         multimask_output=True)
        logits = out.pred_masks[0]          # (points, 3, 256, 256) low-res
        iou = out.iou_scores[0]             # (points, 3) SAM's quality estimate
        best = iou.argmax(dim=1)
        idx = torch.arange(len(points))
        logits, iou = logits[idx, best], iou[idx, best]

        # Stability: IoU between the mask thresholded a bit higher and lower.
        hi, lo = (logits > 1.0).sum((1, 2)), (logits > -1.0).sum((1, 2))
        stability = hi / lo.clamp(min=1)
        ok = (iou >= self.min_iou) & (stability >= self.min_stability)
        if not ok.any():
            return np.zeros((0, *rgb.shape[:2]), bool), np.zeros(0)

        full = self.processor.post_process_masks(
            [logits[ok][:, None]], inputs["original_sizes"], inputs["reshaped_input_sizes"])[0]
        return full[:, 0].numpy(), iou[ok].numpy()


def _grid(n: int, w: int, h: int) -> list[tuple[float, float]]:
    """n x n points at the centers of an even grid over the image."""
    c = (np.arange(n) + 0.5) / n
    return [(x * w, y * h) for y in c for x in c]


def _nms(masks: np.ndarray, scores: np.ndarray, max_iou: float) -> list[int]:
    """Keep the best-scoring mask among near-duplicates."""
    flat = masks.reshape(len(masks), -1).astype(np.float32)
    inter = flat @ flat.T
    area = flat.sum(1)
    iou = inter / (area[:, None] + area[None] - inter + 1e-6)
    keep = []
    for i in np.argsort(-scores):
        if all(iou[i, j] <= max_iou for j in keep):
            keep.append(int(i))
    return keep


def _flatten(masks: np.ndarray, min_area: int) -> np.ndarray:
    """Overlapping masks -> one label image. Small masks are painted last (on top)."""
    labels = np.full(masks.shape[1:], -1, np.int16)
    for i in np.argsort(-masks.reshape(len(masks), -1).sum(1)):
        labels[masks[i]] = i
    out = np.full_like(labels, -1)
    next_id = 0
    for i in np.unique(labels[labels >= 0]):
        region = labels == i
        if region.sum() >= min_area:
            out[region] = next_id
            next_id += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/room0"))
    args = parser.parse_args()

    seq = Sequence(args.data)
    seg = SamSegmenter()
    (args.data / "sam").mkdir(exist_ok=True)
    start = time.time()
    for i in range(len(seq)):
        labels = seg(seq[i].rgb)
        np.save(args.data / "sam" / f"{i:04d}.npy", labels)
        print(f"frame {i:3d}: {labels.max() + 1:2d} regions   "
              f"({(time.time() - start) / (i + 1):.1f} s/frame)", flush=True)


if __name__ == "__main__":
    main()
