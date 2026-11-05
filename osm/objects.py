"""Object-centric map: group 3D points into objects, frame by frame.

Each frame is split into masks (regions). For every mask we
    1. back-project its pixels to 3D world points,
    2. find the existing object it overlaps most in 3D (data association),
    3. merge into that object, or start a new object if nothing overlaps enough.

The mapper does NOT use the simulator's object ids to decide what belongs
together. A learned segmenter (e.g. SAM) produces anonymous masks whose ids
mean nothing across frames, so association has to come from geometry. The
ground-truth ids are carried along only to grade the result afterwards.

Usage:
    python -m osm.objects --data data/room0 --out outputs/room0
Writes
    objects.json        one entry per object: size, position, observations, grading
    objects.ply         point cloud colored by object
    objects_topdown.png bird's-eye view colored by object
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from osm.build_cloud import render_topdown, save_ply
from osm.geometry import Frame, Intrinsics, Sequence, backproject, voxel_downsample


@dataclass
class Observation:
    """One sighting of an object in one frame. CLIP will crop these later."""
    frame: int
    bbox: tuple[int, int, int, int]  # u_min, v_min, u_max, v_max in pixels
    pixels: int


@dataclass
class MapObject:
    id: int
    points: np.ndarray                      # (N, 3) world points, voxel-downsampled
    colors: np.ndarray                      # (N, 3) uint8
    keys: set = field(default_factory=set)  # coarse voxels it occupies, for association
    observations: list[Observation] = field(default_factory=list)
    gt_votes: Counter = field(default_factory=Counter)  # grading only

    def summary(self) -> dict:
        lo, hi = self.points.min(0), self.points.max(0)
        label, votes = self.gt_votes.most_common(1)[0]
        return {"id": self.id,
                "points": len(self.points),
                "center": np.round((lo + hi) / 2, 3).tolist(),
                "size": np.round(hi - lo, 3).tolist(),
                "observations": len(self.observations),
                "gt_label": label,
                "gt_purity": round(votes / sum(self.gt_votes.values()), 3)}


def voxel_keys(points: np.ndarray, size: float) -> set:
    """Set of voxel cells (as hashable tuples) that the points fall into."""
    cells = np.unique(np.floor(points / size).astype(np.int64), axis=0)
    return set(map(tuple, cells))


class ObjectMap:
    def __init__(self, K: Intrinsics, voxel: float = 0.02, assoc_voxel: float = 0.05,
                 min_overlap: float = 0.5, min_pixels: int = 100):
        """
        voxel        resolution of each object's stored points (meters)
        assoc_voxel  coarser grid used to compare a mask with objects; coarse cells
                     tolerate small depth or pose noise
        min_overlap  fraction of a mask's cells that must already belong to an
                     object for the mask to be merged into it
        min_pixels   masks smaller than this are ignored (too noisy to trust)
        """
        self.K = K
        self.voxel = voxel
        self.assoc_voxel = assoc_voxel
        self.min_overlap = min_overlap
        self.min_pixels = min_pixels
        self.objects: list[MapObject] = []

    def associate(self, keys: set) -> MapObject | None:
        """Existing object that overlaps these cells the most, if enough."""
        best, best_overlap = None, 0.0
        for obj in self.objects:
            overlap = len(keys & obj.keys) / len(keys)
            if overlap > best_overlap:
                best, best_overlap = obj, overlap
        return best if best_overlap >= self.min_overlap else None

    def integrate(self, frame: Frame, masks: list[np.ndarray],
                  gt_labels: list[str] | None = None) -> None:
        """Add one frame. `masks` are (H, W) bool arrays, one per region."""
        for i, mask in enumerate(masks):
            mask = mask & (frame.depth > 0)
            if mask.sum() < self.min_pixels:
                continue
            pts, valid = backproject(np.where(mask, frame.depth, 0), self.K, frame.T_wc)
            pts, cols = voxel_downsample(pts, frame.rgb[valid], self.voxel)
            keys = voxel_keys(pts, self.assoc_voxel)

            obj = self.associate(keys)
            if obj is None:
                obj = MapObject(id=len(self.objects), points=pts, colors=cols)
                self.objects.append(obj)
            else:
                obj.points, obj.colors = voxel_downsample(
                    np.concatenate([obj.points, pts]),
                    np.concatenate([obj.colors, cols]), self.voxel)
            obj.keys |= keys

            v, u = np.nonzero(mask)
            obj.observations.append(Observation(frame.index,
                                                (int(u.min()), int(v.min()), int(u.max()), int(v.max())),
                                                int(mask.sum())))
            if gt_labels is not None:
                obj.gt_votes[gt_labels[i]] += int(mask.sum())


def gt_masks(frame: Frame, labels: dict[int, str]) -> tuple[list[np.ndarray], list[str]]:
    """Split a ground-truth segmentation image into anonymous masks.

    The mapper only receives the masks; the label list is for grading.
    """
    ids = [i for i in np.unique(frame.seg) if i >= 0]
    return [frame.seg == i for i in ids], [labels[int(i)] for i in ids]


def grade(objects: list[MapObject], labels: dict[int, str]) -> str:
    """How well do mapped objects match the real ones?

    purity  share of an object's pixels that came from its majority real object
            (low = different things got merged together)
    splits  how many mapped objects a real object ended up as
            (more than 1 = one thing got broken into pieces)
    """
    splits = Counter(o.gt_votes.most_common(1)[0][0] for o in objects)
    lines = [f"{'real object':12s} {'mapped as':>9s} {'purity':>7s}"]
    for name in labels.values():
        mine = [o for o in objects if o.gt_votes.most_common(1)[0][0] == name]
        if not mine:
            lines.append(f"{name:12s} {'MISSED':>9s}")
            continue
        purity = min(o.summary()["gt_purity"] for o in mine)
        lines.append(f"{name:12s} {splits[name]:>9d} {purity:>7.2f}")
    return "\n".join(lines)


def object_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (rng.random((n, 3)) * 200 + 40).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/room0"))
    parser.add_argument("--out", type=Path, default=Path("outputs/room0"))
    args = parser.parse_args()

    seq = Sequence(args.data)
    omap = ObjectMap(seq.K)
    for i in range(len(seq)):
        frame = seq[i]
        masks, names = gt_masks(frame, seq.labels)
        omap.integrate(frame, masks, names)

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [o.summary() for o in omap.objects]
    (args.out / "objects.json").write_text(json.dumps(
        {"objects": summaries,
         "observations": {o.id: [vars(ob) for ob in o.observations] for o in omap.objects}},
        indent=1))

    pal = object_palette(len(omap.objects))
    pts = np.concatenate([o.points for o in omap.objects])
    cols = np.concatenate([np.tile(pal[o.id], (len(o.points), 1)) for o in omap.objects])
    save_ply(args.out / "objects.ply", pts, cols)
    iio.imwrite(args.out / "objects_topdown.png", render_topdown(pts, cols, omap.voxel))

    print(f"{len(omap.objects)} objects\n")
    print(f"{'id':>3s} {'gt_label':10s} {'points':>7s} {'views':>5s}  center              size")
    for s in summaries:
        print(f"{s['id']:>3d} {s['gt_label']:10s} {s['points']:>7d} {s['observations']:>5d}  "
              f"{str(s['center']):19s} {s['size']}")
    print("\n" + grade(omap.objects, seq.labels))


if __name__ == "__main__":
    main()
