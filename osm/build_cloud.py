"""Fuse every frame of a capture into one colored point cloud of the room.

This is mapping with known poses: each frame's depth is back-projected into
world coordinates using its pose, and all frames are merged. Because every
frame uses the same world frame, overlapping views land on top of each other.

Usage:
    python -m osm.build_cloud --data data/room0 --out outputs/room0
Writes
    cloud.ply      open in MeshLab, CloudCompare or Blender
    topdown.png    quick bird's-eye preview
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from osm.geometry import Sequence, backproject, voxel_downsample


def fuse(seq: Sequence, voxel: float, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    all_pts, all_col = [], []
    for i in range(0, len(seq), stride):
        frame = seq[i]
        pts, mask = backproject(frame.depth, seq.K, frame.T_wc)
        # Downsample per frame too, so memory stays small as frames pile up.
        pts, col = voxel_downsample(pts, frame.rgb[mask], voxel)
        all_pts.append(pts)
        all_col.append(col)
    return voxel_downsample(np.concatenate(all_pts), np.concatenate(all_col), voxel)


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary PLY with xyz + rgb, the most widely supported point cloud format."""
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(vertex.tobytes())


def render_topdown(points: np.ndarray, colors: np.ndarray, cell: float,
                   extent: float = 4.2, max_z: float = 1.9, scale: int = 2) -> np.ndarray:
    """Orthographic view from above. Higher points are drawn over lower ones.

    One image pixel covers one `cell` x `cell` square of floor (use the voxel
    size, so there are no gaps between points). The result is enlarged by
    `scale` for easier viewing. Points above max_z are skipped so tall walls
    don't hide the room.
    """
    size = int(round(2 * extent / cell))
    keep = points[:, 2] < max_z
    pts, col = points[keep], colors[keep]
    order = np.argsort(pts[:, 2])  # paint low to high
    pts, col = pts[order], col[order]
    px = ((pts[:, 0] + extent) / (2 * extent) * (size - 1)).astype(int)
    py = ((extent - pts[:, 1]) / (2 * extent) * (size - 1)).astype(int)  # +y is up in the image
    img = np.full((size, size, 3), 255, np.uint8)
    ok = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    img[py[ok], px[ok]] = col[ok]
    return img.repeat(scale, axis=0).repeat(scale, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/room0"))
    parser.add_argument("--out", type=Path, default=Path("outputs/room0"))
    parser.add_argument("--voxel", type=float, default=0.02, help="voxel size in meters")
    parser.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    args = parser.parse_args()

    seq = Sequence(args.data)
    points, colors = fuse(seq, args.voxel, args.stride)
    args.out.mkdir(parents=True, exist_ok=True)
    save_ply(args.out / "cloud.ply", points, colors)
    iio.imwrite(args.out / "topdown.png", render_topdown(points, colors, args.voxel))
    print(f"{len(points):,} points from {len(range(0, len(seq), args.stride))} frames "
          f"-> {args.out / 'cloud.ply'}")


if __name__ == "__main__":
    main()
