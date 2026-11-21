wip. An open vocab, object centric 3d mapper (inspired by the FindAnything paper from ICRA 2026) that builds a map from a camera and allows you to search it with text.

# open_semantic_mapper

A camera moves through a room. The mapper turns what it sees into a 3D map
made of **objects**, gives every object a vision-language embedding, and
answers free-text queries such as `"something to sit on"` or
`"somewhere to put my coffee"` by ranking the objects in the map.

Everything runs on CPU. Input comes from a MuJoCo simulation, which also
provides ground truth (depth, camera poses, per-pixel object labels) so every
stage can be measured.

```
 MuJoCo scene ──► RGB + depth + pose ──► segmentation ──► masks
                                              │
                    ┌─────────────────────────┘
                    ▼
      back-project to 3D ──► data association ──► object map
      (pinhole + pose)       (geometry + CLIP)       │
                                                     ▼
                       text query ──► CLIP ──► relevancy ranking ──► objects
```

## Pipeline

### 1. Simulated RGB-D camera (`osm/sim/`)

* **Scenes.** `room.xml` is a hand-built room of primitive shapes.
  `scenes.py` builds realistic rooms (`living_room`, `study`) from
  photo-textured [Poly Haven](https://polyhaven.com) models (CC0). Every
  object is one MuJoCo body whose name is its ground-truth label.
* **Assets.** `assets.py` downloads a model's glTF, groups its meshes by
  color texture (MuJoCo allows one texture per mesh), rotates glTF's y-up
  axes to z-up, puts the base at z = 0, and writes OBJ + PNG pairs. MuJoCo's
  OBJ loader flips texture v itself, so glTF coordinates are written as-is.
* **Capture.** `capture.py` moves a mocap-mounted camera around a 0.8 m
  circle at 1.4 m height, looking outward and 25° down, and saves 60 frames of
  RGB, metric depth and ground-truth segmentation (geom ids mapped to body
  ids), plus intrinsics and camera-to-world poses.
* **Conventions.** Poses are stored in the OpenCV camera convention
  (x right, y down, z forward). MuJoCo cameras use OpenGL axes, so the
  rotation is multiplied by `diag(1, -1, -1)` once at save time. Intrinsics
  come from the vertical field of view: `fy = (H/2) / tan(fovy/2)`, `fx = fy`.

### 2. Geometry (`osm/geometry.py`)

* **Back-projection:** `X = (u + 0.5 - cx) Z / fx`, `Y = (v + 0.5 - cy) Z / fy`,
  then `p_world = R p_cam + t` with the frame's pose.
* **Voxel downsampling:** points are averaged per 2 cm voxel, which removes
  duplicates from overlapping frames.
* `build_cloud.py` fuses all frames into one colored point cloud (`.ply`) as a
  sanity check of the geometry.

### 3. Segmentation (`osm/segment.py`)

Class-agnostic masks from **SlimSAM** (`Zigeng/SlimSAM-uniform-77`, SAM
distilled to about 1/13 of its size) via Hugging Face `transformers`.

* The image encoder runs once per frame (~2 s on CPU); the mask decoder runs
  once per prompt point (~70 ms each), so prompt points are the cost.
* **Two-pass prompting:** an 8×8 grid, then a 16×16 grid restricted to points
  not already inside a first-pass mask. Large surfaces are covered in pass 1,
  so pass 2 spends points on what is still unexplained (usually small
  objects). This brought segmentation from ~80 s/frame (stock pipeline) to
  ~8 s/frame.
* Per point, the best of SAM's three candidate masks is kept, then masks are
  filtered by predicted IoU (≥ 0.8) and stability (≥ 0.85), deduplicated with
  mask NMS (IoU > 0.7), and flattened into one label image with smaller masks
  painted on top (so a mug is not swallowed by the table mask under it).

The output has the same format as the simulator's ground-truth segmentation,
so the rest of the pipeline accepts either (`--segmentation gt|sam`).

### 4. Object map and data association (`osm/objects.py`)

Masks arrive anonymous: a mask's id in one frame means nothing in the next.
Deciding which masks are the same object is **data association**, and it is
where most of the difficulty is.

For each mask:

1. Back-project the **full mask** to 3D and compute the set of 5 cm cells it
   touches (used for association).
2. Back-project an **eroded mask** (2 px) to get the points that are stored.
   Segment boundaries bleed onto neighbors; one stray pixel on a table edge
   puts a point 77 cm above a floor piece. Association still uses the full
   mask, because generous edges are what make neighboring pieces touch.
3. **Candidates:** existing objects whose cells overlap the mask's cells by at
   least 50%, measured against the *smaller* of the two sets. An object
   sliding into view is first seen as an edge, later as a whole; relative to
   the smaller set (the edge) the overlap is still high.
4. **Pick by appearance:** each mask gets a CLIP embedding; among the
   candidates, the one with the most similar average embedding wins. A mug
   mask overlaps both the mug and the table below it in 3D, and looks like
   the mug. Appearance only ranks; it never rejects the only candidate,
   because two views of one table scored as low as 0.66.
5. **Bridging:** if one mask matches several objects, they were pieces of one
   thing (SAM splits a sofa into back, seat and arm in one frame and returns
   the whole sofa in the next). They are merged only if they look alike
   (cosine ≥ 0.85; sofa parts ~0.88, mug vs table 0.68 to 0.76) and the mask
   spans at least 50 cells (~0.1 m²). The size rule exists because a
   114-pixel, 9-cell sliver at the foot of a bookshelf also touched the floor
   and merged the two.
6. **Coplanar merging** (every 10 frames): CLIP rates most brown wooden
   surfaces as alike (floor vs chair 0.93), so pieces of flat surfaces are
   joined by geometry instead. An object is flat if at least 80% of its points
   lie within 1 cm of a plane fitted in two passes (the second ignoring points
   far from the first fit). Touching flat objects with parallel normals and
   planes within 2 cm are merged. This joins a tiled floor that SAM cuts into
   one region per tile.

Each object keeps its points, cells, summed CLIP embedding, and every
observation (frame, bounding box, cropped mask) for later embedding. Ground-
truth labels ride along only for grading (purity, splits); they never affect
association.

### 5. Object embeddings (`osm/clip_features.py`)

OpenCLIP **ViT-B/32** (`laion2b_s34b_b79k`).

* For each object, the 5 best views are used (largest pixel count; views cut
  off by the image border count half).
* Each view gives two square crops: a **context crop** (bounding box plus a
  20% margin) and a **masked crop** (same square, pixels outside the mask set
  to gray). The two embeddings are added. Context helps small objects; the
  masked crop stops an object absorbing its neighbors' appearance. On the
  primitive room, object names were found 6/9 times with context crops, 7/9
  masked, 8/9 combined.
* **Flat objects** (floor and wall pieces) use only the masked crop. A small
  floor patch next to the plant otherwise has the plant in its context crop
  and wins "a potted plant".
* Objects seen in **fewer than 3 frames** are left out of search. In every
  test room each such object was a fragment (for example a 50-pixel sliver of
  a chair at the image border) of something that also exists in full, and
  their mostly-background crops matched nearly every query.

### 6. Search (`osm/query.py`)

The query is embedded with the CLIP text encoder using the prompt
`"a photo of {query}."`. Raw cosine similarity lets generic regions (walls)
creep into vague queries, so objects are ranked by **relevancy** as in LERF:

```
relevancy = min over c in {object, things, stuff, texture, wall, floor} of
            exp(s_query) / (exp(s_query) + exp(s_c))        (temperature 10)
```

"Does this object match the query better than it matches 'stuff'?" An object
that matches everything also matches "stuff" and scores low; 0.5 means no
preference. "wall" and "floor" are added to LERF's four phrases: with SAM,
walls end up as several pieces, and a wall piece with furniture-shaped holes
cut out of its masked crop used to rank first for queries like "a sofa".
Walls and floors themselves cannot be searched for as a result.

### 7. Evaluation (`osm/eval_queries.py`)

Each scene has a set of queries with the objects that count as correct
(`"something to sit on"` accepts any seat). Reported: **top-1**, **top-3** and
**MRR** (mean reciprocal rank). An object missing from the map counts as a
miss.

## Results

60 frames per room, CPU only.

| Room | Masks | Objects (real) | Top-1 | Top-3 | MRR |
|---|---|---|---|---|---|
| Primitive room | ground truth | 11 (11) | 89% | 100% | 0.94 |
| Primitive room | SAM | 22 (11) | 78% | 94% | 0.87 |
| Living room | ground truth | 14 (12) | 88% | 100% | 0.93 |
| Living room | SAM | 25 (12) | 88% | 88% | 0.91 |
| Study | ground truth | 20 (17) | 84% | 95% | 0.90 |
| Study | SAM | 30 (17) | 63% | 79% | 0.73 |

Known weaknesses:

* **Study with SAM.** The beige walls, grey floor and wooden cabinet look
  alike to CLIP (0.85 to 0.96). At frame 29 one SAM mask spanning floor and
  wall bridges the two; every later wall mask then matches the combined
  object, which ends up holding the floor, all four walls, the shelf and the
  cabinet (half its pixels are not floor). The chest of drawers is therefore
  effectively missing, and "an office chair" confuses the other chairs.
* **Tried and rejected:** blocking bridging for flat objects (many furniture
  parts are flat too; objects 30 to 49 in the study), per-cell majority voting
  (objects 30 to 88), and blocking bridging for objects wider than 3 m. The
  last one does keep the study floor clean and raises study top-1 to 68%, but
  drops the primitive room to 56%, so it is a net loss on these rooms.
* Objects are over-split with SAM (about twice as many as real ones), mostly
  into large, clean parts.
* Segmentation plus embedding takes about 10 s per frame, so this runs
  offline, not in real time.

## Usage

Headless Linux needs CPU rendering: `apt-get install libosmesa6` and
`MUJOCO_GL=osmesa`.

```bash
pip install -r requirements.txt

# Realistic scene (downloads Poly Haven models on first use)
python -m osm.sim.scenes living_room
MUJOCO_GL=osmesa python -m osm.sim.capture --scene data/scenes/living_room/scene.xml --out data/living_room

# Segment, map, embed
python -m osm.segment --data data/living_room
python -m osm.objects --data data/living_room --segmentation sam --out outputs/living_room_sam
python -m osm.clip_features --data data/living_room --out outputs/living_room_sam

# Search and evaluate
python -m osm.query "something to sit on" --out outputs/living_room_sam
python -m osm.eval_queries --out outputs/living_room_sam --queries data/scenes/living_room/queries.json
```

The primitive room is the default everywhere: `python -m osm.sim.capture`
then the same steps with `--data data/room0`. Leaving out `--segmentation`
uses ground-truth masks. `python -m osm.build_cloud` writes a fused point
cloud of a capture.

Models download on first use: SlimSAM and CLIP from Hugging Face, scene
models and textures from Poly Haven.

## Layout

```
osm/
  sim/
    room.xml         primitive test room
    assets.py        Poly Haven download, glTF -> OBJ + PNG for MuJoCo
    scenes.py        realistic rooms and their test queries
    capture.py       simulated RGB-D camera
  geometry.py        intrinsics, poses, back-projection, voxel downsampling
  build_cloud.py     fused point cloud + top-down render
  segment.py         two-pass SAM segmentation
  objects.py         object map and data association
  clip_features.py   per-mask and per-object CLIP embeddings
  query.py           relevancy-ranked text search
  eval_queries.py    top-1 / top-3 / MRR against ground truth
data/                captures, downloaded assets, generated scenes (not committed)
outputs/             maps, embeddings, renders (not committed)
```

## Relation to FindAnything

Shared idea: open-vocabulary, object-centric mapping, with vision-language
features attached to segments and aggregated per object over time instead of
per pixel or per voxel, so a map can be queried with arbitrary text.

Not (yet) here: an occupancy map of free and unknown space, submaps, visual-
inertial odometry (poses come from the simulator), real-time operation, and
exploration planning.

## Credits

* 3D models and textures: [Poly Haven](https://polyhaven.com), CC0.
* Segmentation: SlimSAM (`Zigeng/SlimSAM-uniform-77`), a pruned Segment Anything.
* Embeddings: [OpenCLIP](https://github.com/mlfoundations/open_clip) ViT-B/32 trained on LAION-2B.
* Relevancy scoring: LERF (Kerr et al., ICCV 2023).
* Simulation: [MuJoCo](https://mujoco.org).
