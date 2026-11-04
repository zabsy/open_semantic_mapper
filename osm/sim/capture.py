"""Simulated RGB-D camera: walk a camera through the MuJoCo room and save frames.

For every frame we save
    rgb/NNNN.png     color image (H, W, 3) uint8
    depth/NNNN.npy   depth in meters (H, W) float32, distance along the camera's
                     viewing axis (the "z" depth a real RGB-D sensor reports)
    seg/NNNN.npy     ground-truth object id per pixel (H, W) int16, -1 = nothing

plus, once per run,
    camera.json      intrinsics and one 4x4 pose per frame
    labels.json      object id -> object name (from the body names in room.xml)

Pose convention (important, it is a classic source of bugs):
    Poses are camera-to-world matrices T_wc in the OpenCV camera convention:
    x right, y down, z forward. MuJoCo/OpenGL cameras use x right, y up,
    looking down -z, so we flip the y and z axes when saving.
    A point p_cam in camera coordinates maps to the world as
        p_world = T_wc[:3, :3] @ p_cam + T_wc[:3, 3]

Usage:
    MUJOCO_GL=osmesa python -m osm.sim.capture --out data/room0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

SCENE = Path(__file__).with_name("room.xml")
CAMERA = "rgbd"
RIG = "cam_rig"

# Rotation from OpenGL camera axes (y up, looking -z) to OpenCV (y down, looking +z).
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation matrix (OpenGL camera convention) for a camera at `eye` facing `target`.

    Columns are the camera's x (right), y (up) and z (backward) axes in world
    coordinates. World up is +z.
    """
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, -forward], axis=1)


def intrinsics(model: mujoco.MjModel, width: int, height: int) -> dict:
    """Pinhole intrinsics from MuJoCo's vertical field of view.

    fovy is the full vertical angle, so half the image height covers fovy/2:
        tan(fovy / 2) = (height / 2) / fy
    Pixels are square, so fx = fy.
    """
    fovy = math.radians(model.cam(CAMERA).fovy[0])
    fy = (height / 2) / math.tan(fovy / 2)
    return {"width": width, "height": height,
            "fx": fy, "fy": fy, "cx": width / 2, "cy": height / 2}


def trajectory(n_frames: int, radius: float = 0.8, height: float = 1.4,
               pitch_deg: float = -25.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """A loop around the room center, looking outward and slightly down.

    Returns a list of (eye, target) pairs. Looking outward from near the center
    means one loop sees every wall and every object in the room.
    """
    poses = []
    for i in range(n_frames):
        theta = 2 * math.pi * i / n_frames
        eye = np.array([radius * math.cos(theta), radius * math.sin(theta), height])
        yaw = theta + math.radians(20)  # look a bit sideways, not straight out
        pitch = math.radians(pitch_deg)
        direction = np.array([math.cos(yaw) * math.cos(pitch),
                              math.sin(yaw) * math.cos(pitch),
                              math.sin(pitch)])
        poses.append((eye, eye + direction))
    return poses


def geom_to_object(model: mujoco.MjModel) -> tuple[np.ndarray, dict[int, str]]:
    """Map every geom id to an object id (= its body id) and name each object."""
    geom_obj = model.geom_bodyid.astype(np.int16)
    labels = {}
    for body_id in np.unique(geom_obj):
        name = model.body(int(body_id)).name
        labels[int(body_id)] = name
    return geom_obj, labels


def capture(out: Path, n_frames: int, width: int, height: int) -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    mocap_id = model.body(RIG).mocapid[0]
    cam_id = model.camera(CAMERA).id
    geom_obj, labels = geom_to_object(model)

    for sub in ("rgb", "depth", "seg"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    poses = []
    for i, (eye, target) in enumerate(trajectory(n_frames)):
        # Move the camera rig, then let MuJoCo recompute all positions.
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, look_at(eye, target).flatten())
        data.mocap_pos[mocap_id] = eye
        data.mocap_quat[mocap_id] = quat
        mujoco.mj_forward(model, data)

        # Render the three images from the same camera state.
        renderer.update_scene(data, camera=CAMERA)
        rgb = renderer.render()

        renderer.enable_depth_rendering()
        depth = renderer.render().astype(np.float32)
        renderer.disable_depth_rendering()

        renderer.enable_segmentation_rendering()
        seg_raw = renderer.render()  # [..., 0] = geom id, [..., 1] = object type
        renderer.disable_segmentation_rendering()

        is_geom = (seg_raw[..., 0] >= 0) & (seg_raw[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        seg = np.full((height, width), -1, dtype=np.int16)
        seg[is_geom] = geom_obj[seg_raw[..., 0][is_geom]]
        depth[seg < 0] = 0.0  # no surface hit -> invalid depth, like a real sensor

        # Read the pose back from MuJoCo (not from our own math) and convert to OpenCV.
        T_wc = np.eye(4)
        T_wc[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3) @ GL_TO_CV
        T_wc[:3, 3] = data.cam_xpos[cam_id]
        poses.append(T_wc.tolist())

        iio.imwrite(out / "rgb" / f"{i:04d}.png", rgb)
        np.save(out / "depth" / f"{i:04d}.npy", depth)
        np.save(out / "seg" / f"{i:04d}.npy", seg)

    meta = {"intrinsics": intrinsics(model, width, height),
            "pose_convention": "T_wc, camera-to-world, OpenCV axes (x right, y down, z forward)",
            "poses": poses}
    (out / "camera.json").write_text(json.dumps(meta, indent=1))
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    renderer.close()
    print(f"Saved {n_frames} frames to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/room0"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    capture(args.out, args.frames, args.width, args.height)


if __name__ == "__main__":
    main()
