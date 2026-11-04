"""Camera geometry: turn depth images into 3D points.

The pinhole camera model links a 3D point (X, Y, Z) in camera coordinates
(OpenCV axes: x right, y down, z forward) to a pixel (u, v):

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

Back-projection runs this in reverse. A depth image gives Z for every pixel,
so we can solve for X and Y:

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

Then the camera pose T_wc (camera-to-world) moves the point into the shared
world frame, which is what lets points from different frames line up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Frame:
    """Everything recorded at one instant."""
    index: int
    rgb: np.ndarray    # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, meters, 0 = invalid
    seg: np.ndarray    # (H, W) int16, object id, -1 = nothing
    T_wc: np.ndarray   # (4, 4) camera-to-world


def backproject(depth: np.ndarray, K: Intrinsics, T_wc: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Depth image -> 3D world points.

    Returns
        points: (N, 3) world coordinates of every pixel with valid depth
        mask:   (H, W) bool, which pixels those were (same order as points),
                so callers can pick matching colors or object ids with img[mask]
    """
    mask = depth > 0
    v, u = np.nonzero(mask)
    z = depth[mask]
    # +0.5: a pixel's center, not its top-left corner.
    x = (u + 0.5 - K.cx) * z / K.fx
    y = (v + 0.5 - K.cy) * z / K.fy
    points_cam = np.stack([x, y, z], axis=1)
    points_world = points_cam @ T_wc[:3, :3].T + T_wc[:3, 3]
    return points_world, mask


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Keep one averaged point per voxel (a small cube of side `voxel` meters).

    Neighboring frames see the same surfaces, so a raw merge has many
    near-duplicate points. Averaging per voxel removes them and keeps the
    density even.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    n = len(counts)
    out_pts = np.zeros((n, 3))
    out_col = np.zeros((n, 3))
    np.add.at(out_pts, inverse, points)
    np.add.at(out_col, inverse, colors.astype(np.float64))
    out_pts /= counts[:, None]
    out_col /= counts[:, None]
    return out_pts, out_col.astype(np.uint8)


class Sequence:
    """Read a capture folder written by osm.sim.capture."""

    def __init__(self, root: Path):
        self.root = Path(root)
        meta = json.loads((self.root / "camera.json").read_text())
        self.K = Intrinsics(**meta["intrinsics"])
        self.poses = [np.array(p) for p in meta["poses"]]
        self.labels = {int(k): v for k, v in
                       json.loads((self.root / "labels.json").read_text()).items()}

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, i: int) -> Frame:
        name = f"{i:04d}"
        return Frame(index=i,
                     rgb=iio.imread(self.root / "rgb" / f"{name}.png"),
                     depth=np.load(self.root / "depth" / f"{name}.npy"),
                     seg=np.load(self.root / "seg" / f"{name}.npy"),
                     T_wc=self.poses[i])
