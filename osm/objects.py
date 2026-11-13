"""Object-centric map: group 3D points into objects, frame by frame.

Each frame is split into masks (regions). For every mask we
    1. back-project its pixels to 3D world points,
    2. find the existing objects it overlaps in 3D (data association),
    3. if several, pick the one that looks most alike (CLIP similarity),
    4. merge into it, or start a new object if it overlaps nothing.

Why both geometry and appearance: a learned segmenter is inconsistent. One
frame gives "sofa back" and "sofa seat", another gives the whole sofa; one
frame gives the mug alone, another gives "table with mug". In 3D both cases
look the same (one region covering two objects), but sofa parts look alike
while a mug and a table do not.

Appearance is not enough for everything: in this room most surfaces are brown
wood, and CLIP rates "floor" and "chair" as 0.93 alike. So pieces of flat
surfaces (a segmenter cuts a tiled floor into one region per tile) are joined
by geometry instead: every few frames, touching objects that are flat and lie
on the same plane are merged.

The mapper does NOT use the simulator's object ids to decide what belongs
together. A learned segmenter (e.g. SAM) produces anonymous masks whose ids
mean nothing across frames, so association has to come from the masks
themselves. The ground-truth ids are carried along only to grade the result.

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
from osm.clip_features import Clip, embed_masks
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
    feature_sum: np.ndarray | None = None   # sum of CLIP vectors of its masks

    @property
    def feature(self) -> np.ndarray | None:
        """Average appearance, unit length."""
        if self.feature_sum is None:
            return None
        return self.feature_sum / np.linalg.norm(self.feature_sum)

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


def erode(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Shrink a mask by `pixels` on every side (a pixel survives only if its
    whole 3x3 neighborhood is inside the mask, repeated `pixels` times)."""
    for _ in range(pixels):
        p = np.pad(mask, 1)
        mask = mask.copy()
        for dv in (0, 1, 2):
            for du in (0, 1, 2):
                mask &= p[dv:dv + mask.shape[0], du:du + mask.shape[1]]
    return mask


def voxel_keys(points: np.ndarray, size: float) -> set:
    """Set of voxel cells (as hashable tuples) that the points fall into."""
    cells = np.unique(np.floor(points / size).astype(np.int64), axis=0)
    return set(map(tuple, cells))


class ObjectMap:
    def __init__(self, K: Intrinsics, voxel: float = 0.02, assoc_voxel: float = 0.05,
                 min_overlap: float = 0.5, min_pixels: int = 100, erode_pixels: int = 2,
                 min_similarity: float = 0.85, plane_tolerance: float = 0.01):
        """
        voxel        resolution of each object's stored points (meters)
        assoc_voxel  coarser grid used to compare a mask with objects; coarse cells
                     tolerate small depth or pose noise
        min_overlap  share of cells two sets must have in common (relative to the
                     smaller set) for a mask to be merged into an object
        min_pixels   masks smaller than this are ignored (too noisy to trust)
        erode_pixels masks are shrunk by this much before their points are stored:
                     a segmenter's boundary often bleeds onto the neighbor, and
                     one stray pixel on the table edge puts a point 77 cm above
                     a floor piece (association still uses the full mask)
        min_similarity   CLIP similarity two objects need for one mask covering
                         both to merge them (sofa parts ~0.88; mug vs table ~0.68,
                         but up to 0.76 after an ambiguous mug+table mask)
        plane_tolerance  a point is on a plane if within this many meters of it
        """
        self.K = K
        self.voxel = voxel
        self.assoc_voxel = assoc_voxel
        self.min_overlap = min_overlap
        self.min_pixels = min_pixels
        self.erode_pixels = erode_pixels
        self.min_similarity = min_similarity
        self.plane_tolerance = plane_tolerance
        self.objects: list[MapObject] = []
        self._next_id = 0

    def associate(self, keys: set, feature: np.ndarray | None = None) -> list[MapObject]:
        """Existing objects this mask matches, best match first.

        Geometry: overlap is measured against the SMALLER of the two cell sets.
        When an object slides into view, the first frame sees only an edge; the
        next frame's mask contains that edge plus much more. Relative to the
        smaller set (the edge) the overlap is high, so they still match.

        Appearance (if a feature is given) only ranks the candidates: a mug
        mask overlaps both the mug and the table under it in 3D, and looks
        more like the mug. It never rejects the only candidate, because views
        of one object can look quite different (a table from two angles
        scored as low as 0.66, about as low as mug vs table).
        """
        scored = []
        for obj in self.objects:
            shared = len(keys & obj.keys)
            if shared:
                score = shared / min(len(keys), len(obj.keys))
                if score < self.min_overlap:
                    continue
                if feature is not None:
                    score = float(feature @ obj.feature)
                scored.append((score, obj))
        scored.sort(key=lambda t: -t[0])
        return [obj for _, obj in scored]

    def similar(self, a: MapObject, b: MapObject, threshold: float) -> bool:
        if a.feature is None or b.feature is None:
            return True  # no appearance information: geometry alone decides
        return float(a.feature @ b.feature) >= threshold

    def plane(self, obj: MapObject, min_inliers: float = 0.8
              ) -> tuple[np.ndarray, np.ndarray] | None:
        """(point on plane, unit normal) if the object is flat, else None.

        A best-fit plane passes through the centroid; its normal is the
        direction in which the points vary least (smallest singular vector).
        Segment edges are imperfect, so a floor piece often carries a few
        stray points from the table edge next to it. Those would tilt the fit,
        so we fit twice (the second time ignoring points far from the first
        plane) and call the object flat if most of its points lie on the plane.
        Measured on the test room: flat pieces 88-100%, 3D objects at most 30%.
        """
        if len(obj.points) < 10:
            return None
        pts = obj.points
        for keep in (np.ones(len(pts), bool), None):
            if keep is None:
                keep = np.abs((pts - center) @ normal) < 5 * self.plane_tolerance
                if keep.sum() < 10:
                    return None
            center = pts[keep].mean(axis=0)
            normal = np.linalg.svd(pts[keep] - center, full_matrices=False)[2][-1]
        inliers = np.abs((pts - center) @ normal) < self.plane_tolerance
        return (center, normal) if inliers.mean() >= min_inliers else None

    def merge_coplanar(self, max_angle_cos: float = 0.98, max_offset: float = 0.02) -> int:
        """Merge touching flat objects that lie on the same plane. Returns merges done.

        Two planes are the same if their normals are parallel and one plane's
        center lies within `max_offset` meters of the other plane.
        """
        merged = 0
        planes = {id(o): self.plane(o) for o in self.objects}
        changed = True
        while changed:
            changed = False
            flat = [o for o in self.objects if planes[id(o)] is not None]
            for i, a in enumerate(flat):
                ca, na = planes[id(a)]
                for b in flat[i + 1:]:
                    cb, nb = planes[id(b)]
                    if abs(na @ nb) >= max_angle_cos and abs(na @ (cb - ca)) <= max_offset \
                            and a.keys & b.keys:
                        self.merge(a, b)
                        planes[id(a)] = self.plane(a)
                        merged += 1
                        changed = True
                        break
                if changed:
                    break
        return merged

    def merge(self, target: MapObject, other: MapObject) -> None:
        """Fold `other` into `target` and drop it from the map."""
        target.points, target.colors = voxel_downsample(
            np.concatenate([target.points, other.points]),
            np.concatenate([target.colors, other.colors]), self.voxel)
        target.keys |= other.keys
        target.observations += other.observations
        target.gt_votes += other.gt_votes
        if other.feature_sum is not None:
            target.feature_sum = other.feature_sum if target.feature_sum is None \
                else target.feature_sum + other.feature_sum
        self.objects.remove(other)

    def integrate(self, frame: Frame, masks: list[np.ndarray],
                  features: np.ndarray | None = None,
                  gt_names: dict[int, str] | None = None) -> None:
        """Add one frame. `masks` are (H, W) bool arrays, one per region.

        `features` (optional) holds one CLIP vector per mask. Without it,
        association uses geometry only.

        If `gt_names` is given, the frame's ground-truth segmentation is used to
        record which real objects each mask covered. That is for grading only;
        it never influences association.
        """
        for i, mask in enumerate(masks):
            feature = None if features is None else features[i]
            mask = mask & (frame.depth > 0)
            if mask.sum() < self.min_pixels:
                continue
            # Two point sets from one mask:
            #   full mask    -> association cells. Generous edges are what make
            #                   neighboring pieces (floor tiles, sofa parts) touch.
            #   eroded core  -> the stored shape. Without the ragged edge, a
            #                   floor piece stays flat and a table keeps its size.
            # The observation below keeps the full mask for CLIP crops.
            full, _ = backproject(np.where(mask, frame.depth, 0), self.K, frame.T_wc)
            keys = voxel_keys(full, self.assoc_voxel)
            core = erode(mask, self.erode_pixels)
            if not core.any():
                core = mask
            pts, valid = backproject(np.where(core, frame.depth, 0), self.K, frame.T_wc)
            pts, cols = voxel_downsample(pts, frame.rgb[valid], self.voxel)

            matches = self.associate(keys, feature)
            if not matches:
                obj = MapObject(id=self._next_id, points=pts, colors=cols)
                self._next_id += 1
                self.objects.append(obj)
            else:
                # A mask covering several objects that look alike means they
                # were pieces of one thing seen from different angles: join them.
                obj = matches[0]
                for other in matches[1:]:
                    if self.similar(obj, other, self.min_similarity):
                        self.merge(obj, other)
                obj.points, obj.colors = voxel_downsample(
                    np.concatenate([obj.points, pts]),
                    np.concatenate([obj.colors, cols]), self.voxel)
            obj.keys |= keys
            if feature is not None:
                obj.feature_sum = feature.copy() if obj.feature_sum is None \
                    else obj.feature_sum + feature

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
    parser.add_argument("--geometry-only", action="store_true",
                        help="associate by 3D overlap alone, without CLIP")
    parser.add_argument("--merge-every", type=int, default=10,
                        help="merge touching coplanar objects every N frames")
    args = parser.parse_args()

    seq = Sequence(args.data)
    omap = ObjectMap(seq.K)
    clip = None if args.geometry_only else Clip()
    for i in range(len(seq)):
        frame = seq[i]
        masks = split_labels(load_segmentation(args.data, frame, args.segmentation))
        masks = [m for m in masks if (m & (frame.depth > 0)).sum() >= omap.min_pixels]
        features = None if clip is None or not masks else embed_masks(clip, frame.rgb, masks)
        omap.integrate(frame, masks, features, seq.labels)
        if (i + 1) % args.merge_every == 0:
            omap.merge_coplanar()
    omap.merge_coplanar()

    for new_id, obj in enumerate(omap.objects):  # merges leave gaps in ids
        obj.id = new_id

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [o.summary() for o in omap.objects]
    for s, o in zip(summaries, omap.objects):
        s["flat"] = omap.plane(o) is not None  # floor, wall, tabletop-like pieces
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
