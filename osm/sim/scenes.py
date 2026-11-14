"""Realistic test rooms built from Poly Haven models.

Each scene is an 8 x 8 m room (same size as the primitive test room, so the
same camera loop works) furnished with photo-textured models. Every placed
model becomes one MuJoCo body whose name is its ground-truth label, exactly
like room.xml, so capture, grading and evaluation work unchanged.

Each scene also lists test queries with the objects that count as correct
answers. Two different rooms let us tune on one and check on the other, so
improvements are less likely to be tricks that only fit one room.

Usage:
    python -m osm.sim.scenes living_room study
Writes data/scenes/<name>/scene.xml and queries.json, downloading models
on first use. Then for example:
    python -m osm.sim.capture --scene data/scenes/living_room/scene.xml --out data/living_room
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from osm.sim.assets import ASSETS, convert, texture

SCENES_DIR = Path("data/scenes")
HALF = 4.0  # room spans [-4, 4] in x and y


@dataclass
class Item:
    label: str                      # body name = ground-truth label
    model: str                      # Poly Haven model id
    x: float
    y: float
    z: float = 0.0                  # height of the model's base (e.g. on a table)
    face: tuple[float, float] = (0.0, 0.0)  # point the model's front turns toward


@dataclass
class Scene:
    floor: str                      # Poly Haven texture ids
    wall: str
    items: list[Item]
    queries: dict[str, list[str]] = field(default_factory=dict)


SCENES = {
    "living_room": Scene(
        floor="wood_floor", wall="painted_plaster_wall",
        items=[
            Item("sofa", "Sofa_01", -3.3, 0.0),
            Item("armchair", "ArmChair_01", -2.2, 2.6),
            Item("coffee_table", "CoffeeTable_01", -1.8, 0.0),
            Item("books", "book_encyclopedia_set_01", -1.8, 0.35, z=0.523),
            Item("vase", "ceramic_vase_01", -1.8, -0.45, z=0.523),
            Item("tv", "Television_01", 3.6, 0.0),  # vintage set on its own legs
            Item("plant", "potted_plant_01", 3.2, 3.2),
            Item("bookshelf", "wooden_bookshelf_worn", -0.5, 3.65),
            Item("football", "football", 1.0, -2.3),
            Item("basket", "wicker_basket_01", 2.4, -3.5),
        ],
        queries={
            "a sofa": ["sofa"],
            "an armchair": ["armchair"],
            "a coffee table": ["coffee_table"],
            "a set of books": ["books"],
            "a vase": ["vase"],
            "a television": ["tv"],
            "a potted plant": ["plant"],
            "a bookshelf": ["bookshelf", "books"],
            "a ball": ["football"],
            "a wicker basket": ["basket"],
            "a couch": ["sofa"],
            "something to sit on": ["sofa", "armchair"],
            "something to read": ["books", "bookshelf"],
            "something green": ["plant"],
            "something to play sports with": ["football"],
            "somewhere to put my coffee": ["coffee_table"],
            "something to watch the news on": ["tv"],
        }),
    "study": Scene(
        floor="laminate_floor_02", wall="beige_wall_001",
        items=[
            Item("desk", "metal_office_desk", 0.0, 3.45),
            Item("desk_lamp", "desk_lamp_arm_01", -0.7, 3.55, z=0.788, face=(0.0, 0.0)),
            Item("alarm_clock", "alarm_clock_01", 0.6, 3.5, z=0.788),
            Item("office_chair", "modern_arm_chair_01", 0.0, 2.4, face=(0.0, 4.0)),
            Item("dining_table", "dining_table", 1.9, -1.8),
            Item("dining_chair_1", "dining_chair_02", 1.9, -0.75, face=(1.9, -1.8)),
            Item("dining_chair_2", "dining_chair_02", 1.9, -2.85, face=(1.9, -1.8)),
            Item("cabinet", "drawer_cabinet", -3.65, -1.5),
            Item("shelf", "Shelf_01", -3.8, 1.6),
            Item("plant", "potted_plant_02", -3.2, -3.3),
            Item("tv_stool", "wooden_stool_01", 3.5, 2.5),
            Item("tv", "television_02", 3.5, 2.5, z=0.437),
            Item("bat", "baseball_bat", 3.6, -3.6),
            Item("vase", "brass_vase_01", -2.4, 3.6),
            Item("green_chair", "GreenChair_01", -1.5, -3.3),
        ],
        queries={
            "a desk": ["desk"],
            "a desk lamp": ["desk_lamp"],
            "an alarm clock": ["alarm_clock"],
            "an office chair": ["office_chair"],
            "a dining table": ["dining_table"],
            "a wooden chair": ["dining_chair_1", "dining_chair_2"],
            "a chest of drawers": ["cabinet"],
            "a shelf": ["shelf"],
            "a potted plant": ["plant"],
            "a television": ["tv"],
            "a baseball bat": ["bat"],
            "a brass vase": ["vase"],
            "a green armchair": ["green_chair"],
            "something to sit on": ["office_chair", "dining_chair_1", "dining_chair_2",
                                    "green_chair", "tv_stool"],
            "something that tells the time": ["alarm_clock"],
            "a light source": ["desk_lamp"],
            "somewhere to eat dinner": ["dining_table"],
            "somewhere to work": ["desk"],
            "something to store clothes in": ["cabinet"],
        }),
}


def yaw_to_face(x: float, y: float, target: tuple[float, float]) -> float:
    """Yaw (degrees) that turns a model's front (-y) toward `target`.

    Rotating (0, -1) by yaw gives (sin yaw, -cos yaw); set that equal to the
    direction from the model to the target.
    """
    dx, dy = target[0] - x, target[1] - y
    return math.degrees(math.atan2(dx, -dy))


def build(name: str) -> Path:
    scene = SCENES[name]
    out = SCENES_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    floor_tex = texture(scene.floor).resolve()
    wall_tex = texture(scene.wall).resolve()
    assets = [
        f'<texture name="floor_tex" type="2d" file="{floor_tex}"/>',
        '<material name="floor_mat" texture="floor_tex" texrepeat="4 4"/>',
        f'<texture name="wall_tex" type="2d" file="{wall_tex}"/>',
        '<material name="wall_mat" texture="wall_tex" texrepeat="3 1"/>',
    ]
    bodies = []
    for item in scene.items:
        manifest = convert(item.model)
        folder = (ASSETS / item.model).resolve()
        geoms = []
        for k, part in enumerate(manifest["parts"]):
            key = f"{item.label}_{k}"
            assets.append(f'<mesh name="{key}" file="{folder / part["mesh"]}" inertia="shell"/>')
            if "texture" in part:
                assets.append(f'<texture name="{key}" type="2d" file="{folder / part["texture"]}"/>')
                assets.append(f'<material name="{key}" texture="{key}"/>')
            else:
                rgba = " ".join(str(c) for c in part["rgba"])
                assets.append(f'<material name="{key}" rgba="{rgba}"/>')
            geoms.append(f'<geom type="mesh" mesh="{key}" material="{key}" '
                         f'contype="0" conaffinity="0"/>')
        yaw = yaw_to_face(item.x, item.y, item.face)
        bodies.append(f'<body name="{item.label}" pos="{item.x} {item.y} {item.z}" '
                      f'euler="0 0 {yaw:.1f}">\n      ' + "\n      ".join(geoms) + "\n    </body>")

    wall_geoms = "\n      ".join(
        f'<geom type="box" pos="{px} {py} 1.25" size="{sx} {sy} 1.25" material="wall_mat"/>'
        for px, py, sx, sy in [(0, HALF, HALF, 0.05), (0, -HALF, HALF, 0.05),
                               (HALF, 0, 0.05, HALF), (-HALF, 0, 0.05, HALF)])
    nl = "\n    "
    xml = f"""<!-- Generated by osm.sim.scenes from Poly Haven models (CC0). Do not edit. -->
<mujoco model="{name}">
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.5 0.5 0.5" specular="0 0 0"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    {nl.join(assets)}
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light pos="2 2 3" dir="-0.3 -0.3 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <body name="floor">
      <geom type="plane" size="{HALF} {HALF} 0.1" material="floor_mat"/>
    </body>
    <body name="wall">
      {wall_geoms}
    </body>
    {nl.join(bodies)}
    <body name="cam_rig" mocap="true" pos="0 0 1.4">
      <camera name="rgbd" fovy="60" pos="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""
    (out / "scene.xml").write_text(xml)
    (out / "queries.json").write_text(json.dumps(scene.queries, indent=1))
    return out / "scene.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenes", nargs="+", choices=sorted(SCENES))
    args = parser.parse_args()
    for name in args.scenes:
        print(f"{name}: {build(name)}")


if __name__ == "__main__":
    main()
