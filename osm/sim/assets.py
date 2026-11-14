"""Download Poly Haven models and convert them for MuJoCo.

Poly Haven (polyhaven.com) publishes photo-textured 3D models under CC0, i.e.
public domain: free to use for anything, no attribution required.

Models come as glTF. MuJoCo cannot read glTF, but it can read OBJ meshes with
a PNG texture, one texture per mesh. So for each model we
    1. download the glTF with 1k textures,
    2. split it into one part per material (a sofa might be "fabric" + "legs"),
    3. rotate from glTF's y-up axes to our z-up axes,
    4. shift it so its footprint is centered on the origin and its base at z=0,
    5. write each part as an OBJ (with texture coordinates) plus its color texture.

Everything lands in data/assets/<model id>/ with a manifest.json describing
the parts and the model's size. Downloads are cached: a model is fetched once.

Usage:
    python -m osm.sim.assets Sofa_01 ArmChair_01
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

API = "https://api.polyhaven.com"
ASSETS = Path("data/assets")

# glTF is y-up; we are z-up. Rotate +90 degrees about x: (x, y, z) -> (x, -z, y).
# A model's front (+z in glTF) ends up facing -y.
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "open_semantic_mapper"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def download(model_id: str, resolution: str = "1k") -> Path:
    """Fetch the glTF and the files it references. Returns the .gltf path."""
    folder = ASSETS / model_id / "src"
    files = json.loads(_get(f"{API}/files/{model_id}"))
    gltf = files["gltf"][resolution]["gltf"]
    target = folder / Path(gltf["url"]).name
    if target.exists():
        return target
    for rel, info in gltf["include"].items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(info["url"]))
    target.write_bytes(_get(gltf["url"]))
    return target


def texture(texture_id: str, resolution: str = "1k") -> Path:
    """Download a Poly Haven surface texture (its color map) as PNG. Cached."""
    target = ASSETS / "textures" / f"{texture_id}.png"
    if not target.exists():
        files = json.loads(_get(f"{API}/files/{texture_id}"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_get(files["Diffuse"][resolution]["png"]["url"]))
    return target


def write_obj(path: Path, vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray) -> None:
    """Minimal OBJ with one texture coordinate per vertex.

    glTF puts the texture origin at the top-left and OBJ at the bottom-left,
    but MuJoCo's OBJ loader flips v itself, so glTF coordinates go in as-is
    (flipping them here scrambled the sofa's fabric and wood).
    """
    lines = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in vertices]
    lines += [f"vt {u:.5f} {v:.5f}" for u, v in uv]
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in faces + 1]
    path.write_text("\n".join(lines) + "\n")


def convert(model_id: str) -> dict:
    """Download (if needed) and convert one model. Returns its manifest."""
    out = ASSETS / model_id
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    import trimesh
    scene = trimesh.load(download(model_id), force="scene")
    meshes = [m for m in scene.dump() if isinstance(m, trimesh.Trimesh) and len(m.faces)]

    # Group submeshes by their color texture, so each MuJoCo mesh has one texture.
    groups: dict[int, list] = {}
    textures = {}
    for m in meshes:
        mat = getattr(m.visual, "material", None)
        image = getattr(mat, "baseColorTexture", None) if mat is not None else None
        key = id(image)
        textures[key] = (image, getattr(mat, "baseColorFactor", None))
        groups.setdefault(key, []).append(m)

    all_vertices = np.concatenate([m.vertices for m in meshes]) @ Y_UP_TO_Z_UP.T
    lo, hi = all_vertices.min(0), all_vertices.max(0)
    shift = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])

    parts = []
    for n, (key, group) in enumerate(groups.items()):
        verts, uvs, faces, offset = [], [], [], 0
        for m in group:
            uv = getattr(m.visual, "uv", None)
            verts.append(m.vertices @ Y_UP_TO_Z_UP.T + shift)
            uvs.append(uv if uv is not None else np.zeros((len(m.vertices), 2)))
            faces.append(m.faces + offset)
            offset += len(m.vertices)
        name = f"part{n}"
        write_obj(out / f"{name}.obj", np.concatenate(verts), np.concatenate(uvs),
                  np.concatenate(faces))
        image, factor = textures[key]
        part = {"mesh": f"{name}.obj"}
        if image is not None:
            image.convert("RGB").save(out / f"{name}.png")
            part["texture"] = f"{name}.png"
        else:
            rgba = np.array(factor if factor is not None else [200, 200, 200, 255], float)
            part["rgba"] = (rgba / (255 if rgba.max() > 1 else 1)).round(3).tolist()
        parts.append(part)

    manifest = {"id": model_id, "size": np.round(hi - lo, 3).tolist(), "parts": parts,
                "source": f"https://polyhaven.com/a/{model_id}", "license": "CC0"}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    for model_id in args.models:
        m = convert(model_id)
        print(f"{model_id:28s} size {m['size']}  parts {len(m['parts'])}")


if __name__ == "__main__":
    main()
