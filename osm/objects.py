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
    python -m osm.objects --segmentation sam --out outputs/room0_sam
Writes
    objects.json        one entry per object: size, position, observations, grading
    objects.ply         point cloud colored by object
    objects_topdown.png bird's-eye view colored by object
    masks.npz           each observation's mask, cropped to its bounding box
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
    mask: np.ndarray                 # (v_max - v_min + 1, u_max - u_min + 1) bool, the region inside bbox

    def to_json(self) -> dict:
        return {"frame": self.frame, "bbox": list(self.bbox), "pixels": self.pixels}


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
        min_overlap  share of cells two sets must have in common (relative to the
                     smaller set) for a mask to be merged into an object
        min_pixels   masks smaller than this are ignored (too noisy to trust)
        """
        self.K = K
        self.voxel = voxel
        self.assoc_voxel = assoc_voxel
        self.min_overlap = min_overlap
        self.min_pixels = min_pixels
        self.objects: list[MapObject] = []
        self._next_id = 0

    def associate(self, keys: set) -> list[MapObject]:
        """All existing objects this mask overlaps enough, best match first.

        Overlap is measured against the SMALLER of the two cell sets. When an
        object slides into view, the first frame sees only an edge; the next
        frame's mask contains that edge plus much more. Relative to the smaller
        set (the edge) the overlap is high, so they still match.
        """
        scored = []
        for obj in self.objects:
            shared = len(keys & obj.keys)
            if shared:
                score = shared / min(len(keys), len(obj.keys))
                if score >= self.min_overlap:
                    scored.append((score, obj))
        scored.sort(key=lambda t: -t[0])
        return [obj for _, obj in scored]

    def merge(self, target: MapObject, other: MapObject) -> None:
        """Fold `other` into `target` and drop it from the map."""
        target.points, target.colors = voxel_downsample(
            np.concatenate([target.points, other.points]),
            np.concatenate([target.colors, other.colors]), self.voxel)
        target.keys |= other.keys
        target.observations += other.observations
        target.gt_votes += other.gt_votes
        self.objects.remove(other)

    def integrate(self, frame: Frame, masks: list[np.ndarray],
                  gt_names: dict[int, str] | None = None) -> None:
        """Add one frame. `masks` are (H, W) bool arrays, one per region.

        If `gt_names` is given, the frame's ground-truth segmentation is used to
        record which real objects each mask covered. That is for grading only;
        it never influences association.
        """
        for mask in masks:
            mask = mask & (frame.depth > 0)
            if mask.sum() < self.min_pixels:
                continue
            pts, valid = backproject(np.where(mask, frame.depth, 0), self.K, frame.T_wc)
            pts, cols = voxel_downsample(pts, frame.rgb[valid], self.voxel)
            keys = voxel_keys(pts, self.assoc_voxel)

            matches = self.associate(keys)
            if not matches:
                obj = MapObject(id=self._next_id, points=pts, colors=cols)
                self._next_id += 1
                self.objects.append(obj)
            else:
                # A mask touching several objects means they were pieces of one
                # thing seen from different angles: join them.
                obj = matches[0]
                for other in matches[1:]:
                    self.merge(obj, other)
                obj.points, obj.colors = voxel_downsample(
                    np.concatenate([obj.points, pts]),
                    np.concatenate([obj.colors, cols]), self.voxel)
            obj.keys |= keys

            v, u = np.nonzero(mask)
            u0, v0, u1, v1 = int(u.min()), int(v.min()), int(u.max()), int(v.max())
            obj.observations.append(Observation(frame.index, (u0, v0, u1, v1), int(mask.sum()),
                                                mask[v0:v1 + 1, u0:u1 + 1].copy()))
            if gt_names is not None:
                ids, counts = np.unique(frame.seg[mask], return_counts=True)
                for gid, n in zip(ids, counts):
                    if gid >= 0:
                        obj.gt_votes[gt_names[int(gid)]] += int(n)


def split_labels(labels: np.ndarray) -> list[np.ndarray]:
    """Label image (one id per pixel, -1 = none) -> one bool mask per id.

    The ids are thrown away: the mapper only gets anonymous regions, whether
    they came from the simulator or from SAM.
    """
    return [labels == i for i in np.unique(labels) if i >= 0]


def load_segmentation(data: Path, frame: Frame, source: str) -> np.ndarray:
    if source == "gt":
        return frame.seg
    return np.load(data / source / f"{frame.index:04d}.npy")


def grade(objects: list[MapObject], labels: dict[int, str]) -> str:
    """How well do mapped objects match the real ones?

    purity  share of an object's pixels that came from its majority real object
            (low = different things got merged together)
    splits  how many mapped objects a real object ended up as
            (more than 1 = one thing got broken into pieces)
    """
    splits = Counter(o.gt_votes.most_common(1)[0][0] for o in objects)
    lines = [f"{len(objects)} mapped objects for {len(labels)} real objects\n",
             f"{'real object':12s} {'mapped as':>9s} {'purity':>7s}"]
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
    parser.add_argument("--segmentation", default="gt",
                        help="'gt' for simulator masks, or a folder inside --data "
                             "with label images (e.g. 'sam' from osm.segment)")
    args = parser.parse_args()

    seq = Sequence(args.data)
    omap = ObjectMap(seq.K)
    for i in range(len(seq)):
        frame = seq[i]
        masks = split_labels(load_segmentation(args.data, frame, args.segmentation))
        omap.integrate(frame, masks, seq.labels)

    for new_id, obj in enumerate(omap.objects):  # merges leave gaps in ids
        obj.id = new_id

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [o.summary() for o in omap.objects]
    (args.out / "objects.json").write_text(json.dumps(
        {"objects": summaries,
         "observations": {o.id: [ob.to_json() for ob in o.observations] for o in omap.objects}},
        indent=1))
    # Keyed "<object id>/<observation index>", same order as in objects.json.
    np.savez_compressed(args.out / "masks.npz", **{
        f"{o.id}/{i}": ob.mask for o in omap.objects for i, ob in enumerate(o.observations)})

    pal = object_palette(len(omap.objects))
    pts = np.concatenate([o.points for o in omap.objects])
    cols = np.concatenate([np.tile(pal[o.id], (len(o.points), 1)) for o in omap.objects])
    save_ply(args.out / "objects.ply", pts, cols)
    iio.imwrite(args.out / "objects_topdown.png", render_topdown(pts, cols, omap.voxel))

    print(f"{'id':>3s} {'gt_label':10s} {'points':>7s} {'views':>5s}  center              size")
    for s in summaries:
        print(f"{s['id']:>3d} {s['gt_label']:10s} {s['points']:>7d} {s['observations']:>5d}  "
              f"{str(s['center']):19s} {s['size']}")
    print("\n" + grade(omap.objects, seq.labels))


if __name__ == "__main__":
    main()
