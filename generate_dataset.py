#!/usr/bin/env python3
"""
generate_dataset.py (Blender 3.6+)

Headless-friendly dataset render script:
- Renders RGB PNGs + Depth EXR (Z-pass) via compositor
- Generates per-frame JSON annotations with camera pose + intrinsics
- Animates camera in a zig-zag path across trellis wires

Key fixes included:
- Linux-safe paths (no Windows C:\)
- Eevee + lower resolution defaults to reduce OOM kills
- Depth compositor outputs EXR with FP16 + ZIP compression
- compute_camera_intrinsics() is its own function (NOT inside setup_depth_compositor)
- Helpful errors if objects are missing
- Output directory created automatically
"""

import bpy
import os
import json
import math
from math import radians
from mathutils import Vector
# Add after line 25 (after imports)
import gc

# Add a memory cleanup function after line 90
def cleanup_memory():
    """Force garbage collection and cleanup unused Blender data"""
    # Remove unused data blocks
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    # Force Python garbage collection
    gc.collect()

# Modify the main() function - add cleanup after line 285 (after trunk_points)
def main():
    scene = bpy.context.scene

    # Render stability settings
    scene.render.engine = RENDER_ENGINE
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = RES_PERCENT
    
    # ADD THESE MEMORY OPTIMIZATIONS:
    # Disable unnecessary features to save memory
    scene.eevee.taa_render_samples = 1  # Minimal samples
    scene.eevee.use_ssr = False  # Disable screen space reflections
    scene.eevee.use_bloom = False
    scene.eevee.use_gtao = False
    scene.eevee.use_soft_shadows = False
    
    # Limit texture size
    scene.render.image_settings.compression = 90  # Higher compression
    
    # Get required objects (throws helpful error if missing)
    tree_obj  = get_obj(TREE_OBJ_NAME)
    cam_obj   = get_obj(CAM_NAME)
    post0_obj = get_obj(POST0_NAME)
    post1_obj = get_obj(POST1_NAME)
    wire_objs = [get_obj(n) for n in WIRE_NAMES]

    # Camera basics
    cam_data = cam_obj.data
    cam_data.clip_start = 0.001
    cam_data.lens = 50  # mm

    rgb_dir, depth_dir, ann_dir = ensure_output_dirs(OUTPUT_DIR)

    trunk_points = sample_trunk_points(tree_obj, step=STEP)
    setup_depth_compositor(depth_dir)
    
    # ADD THIS: Cleanup after initial setup
    cleanup_memory()

    # ... rest of the function ...
    
    # In the render loop (after line 311), add cleanup every 10 frames:
    for f in range(1, total_frames + 1):
        scene.frame_set(f)

        rgb_path = os.path.join(rgb_dir, f"rgb_{f:04d}.png")
        scene.render.filepath = rgb_path

        # Render RGB + compositor (depth EXR)
        bpy.ops.render.render(write_still=True)

        intrinsics = compute_camera_intrinsics(cam_obj, scene)

        ann = {
            "frame": f,
            "rgb_path": os.path.abspath(rgb_path),
            "depth_file_pattern": os.path.abspath(os.path.join(depth_dir, "depth_####.exr")),
            "camera": {
                "location": [float(v) for v in cam_obj.location],
                "rotation_euler": [float(a) for a in cam_obj.rotation_euler],
                "intrinsics": intrinsics,
            },
            "points": trunk_points,
        }

        ann_path = os.path.join(ann_dir, f"frame_{f:04d}.json")
        with open(ann_path, "w") as f_json:
            json.dump(ann, f_json, indent=2)

        # ADD THIS: Periodic memory cleanup
        if f % 10 == 0:
            cleanup_memory()

        # Light progress message (helps you see it isn't frozen)
        if f == 1 or f % 10 == 0 or f == total_frames:
            print(f"  Frame {f}/{total_frames}")

    print("Done! Zig-zag dataset written to:", OUTPUT_DIR)
# ==============================
# CONFIG (EDIT THESE)
# ==============================
TREE_OBJ_NAME = "tree0_TRUNK"
CAM_NAME      = "Camera"

POST0_NAME = "post0"
POST1_NAME = "post1"

WIRE_NAMES = [
    "wire0_1",
    "wire0_2",
    "wire0_3",
    "wire0_4",
    "wire0_5",
    "wire0_6",
]

STEP = 0.1  # trunk sample step (scene units; 0.1 ~ 10cm if 1 unit = 1m)

# Zig-zag animation params
FRAMES_PER_SEGMENT = 25
DIST_FROM_SURFACE  = -1.85
TILT_RANGE         = radians(20)

# Render stability defaults (reduce Linux "Killed" / OOM)
RENDER_ENGINE = 'BLENDER_EEVEE'
RES_X = 640
RES_Y = 480
RES_PERCENT = 100

# Output directory (portable): next to this script by default
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "tree_dataset")

# Optional: override output dir via env var:
#   OUTPUT_DIR=/some/path blender -b ... -P generate_dataset.py
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", OUTPUT_DIR)


# ==============================
# HELPERS
# ==============================
def get_obj(name: str) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None:
        available = sorted([o.name for o in bpy.data.objects])
        preview = "\n  - " + "\n  - ".join(available[:40])
        more = "" if len(available) <= 40 else f"\n  ... (+{len(available)-40} more)"
        raise RuntimeError(
            f"Missing object '{name}'.\n"
            f"Fix by renaming the object in your .blend OR updating the CONFIG.\n"
            f"Objects found (first 40):{preview}{more}"
        )
    return obj


def ensure_output_dirs(base_dir: str):
    rgb_dir   = os.path.join(base_dir, "rgb")
    depth_dir = os.path.join(base_dir, "depth")
    ann_dir   = os.path.join(base_dir, "ann")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    return rgb_dir, depth_dir, ann_dir


def sample_trunk_points(tree_obj, step=0.1):
    """
    Sample points along the tree object's LOCAL Z axis (bounding box min_z->max_z).
    Stores world position + world direction of local +Z.
    """
    bbox = [Vector(corner) for corner in tree_obj.bound_box]
    min_z = min(v.z for v in bbox)
    max_z = max(v.z for v in bbox)

    local_dir = Vector((0.0, 0.0, 1.0))
    dir_world = (tree_obj.matrix_world.to_quaternion() @ local_dir).normalized()

    points = []
    pid = 0
    z = min_z
    while z <= max_z + 1e-9:
        local_pos = Vector((0.0, 0.0, z))
        world_pos = tree_obj.matrix_world @ local_pos

        points.append({
            "id": pid,
            "pos_world": [float(world_pos.x), float(world_pos.y), float(world_pos.z)],
            "dir_world": [float(dir_world.x), float(dir_world.y), float(dir_world.z)],
        })
        pid += 1
        z += step

    return points


def setup_depth_compositor(depth_dir: str):
    """
    Configure compositor to output Z (Depth) pass to OpenEXR files:
      depth_0001.exr, depth_0002.exr, ...
    Uses FP16 + ZIP compression to reduce memory and disk.
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree

    # Clear existing nodes
    tree.nodes.clear()

    rl = tree.nodes.new(type='CompositorNodeRLayers')
    rl.location = (0, 0)

    out = tree.nodes.new(type='CompositorNodeOutputFile')
    out.label = "Depth Output"
    out.base_path = depth_dir
    out.format.file_format = 'OPEN_EXR'
    out.format.color_depth = '16'  # FP16 (half float)
    out.format.exr_codec = 'ZIP'   # compression
    out.file_slots[0].path = "depth_"

    tree.links.new(rl.outputs['Depth'], out.inputs[0])

    # Enable Z pass
    bpy.context.view_layer.use_pass_z = True


def compute_camera_intrinsics(cam_obj, scene):
    """
    Approximate camera intrinsics K for Blender camera.
    Returns dict: {"width","height","K":[[fx,0,cx],[0,fy,cy],[0,0,1]]}
    """
    cam = cam_obj.data
    render = scene.render

    scale = render.resolution_percentage / 100.0
    width = render.resolution_x * scale
    height = render.resolution_y * scale

    sensor_width = cam.sensor_width
    sensor_height = cam.sensor_height

    # Horizontal fit is common; Blender supports AUTO/HORIZONTAL/VERTICAL.
    # We'll treat anything not VERTICAL as horizontal fit.
    if cam.sensor_fit == 'VERTICAL':
        sensor_size = sensor_height
    else:
        sensor_size = sensor_width

    pixel_aspect = render.pixel_aspect_x / render.pixel_aspect_y

    fx = cam.lens / sensor_size * width
    fy = cam.lens / sensor_size * height * pixel_aspect

    cx = width / 2.0
    cy = height / 2.0

    K = [
        [float(fx), 0.0,      float(cx)],
        [0.0,       float(fy), float(cy)],
        [0.0,       0.0,      1.0]
    ]

    return {"width": int(width), "height": int(height), "K": K}


def animate_zigzag_wires_camera(
    cam_obj,
    post0_obj,
    post1_obj,
    wire_objs,
    dist_from_surface=0.20,
    frames_per_segment=20,
    tilt_range=radians(20)
):
    """
    Snake path across wires:
    - For each wire: move from one post to the other.
    - Alternates direction each wire (zig-zag).
    - Offsets camera along world -Y by dist_from_surface (flip sign if needed).
    - Tilts slightly from bottom->top wire.
    """
    scene = bpy.context.scene

    p0 = post0_obj.matrix_world.translation
    p1 = post1_obj.matrix_world.translation

    left, right = (p0, p1) if p0.x <= p1.x else (p1, p0)

    normal_dir = Vector((0.0, -1.0, 0.0)).normalized()

    total_wires = len(wire_objs)
    frame = 1

    for idx, wire_obj in enumerate(wire_objs):
        bbox = [wire_obj.matrix_world @ Vector(corner) for corner in wire_obj.bound_box]
        wire_z = sum(v.z for v in bbox) / 8.0

        if idx % 2 == 0:
            start = Vector((right.x, right.y, wire_z))
            end   = Vector((left.x,  left.y,  wire_z))
        else:
            start = Vector((left.x,  left.y,  wire_z))
            end   = Vector((right.x, right.y, wire_z))

        if total_wires > 1:
            t_wire = idx / (total_wires - 1)
        else:
            t_wire = 0.5
        extra_tilt = (t_wire - 0.5) * tilt_range

        for s in range(frames_per_segment):
            t = 0.5 if frames_per_segment == 1 else s / (frames_per_segment - 1)

            target = start.lerp(end, t)
            cam_loc = target + normal_dir * dist_from_surface

            scene.frame_set(frame)
            cam_obj.location = cam_loc

            direction = (target - cam_loc).normalized()
            rot = direction.to_track_quat('-Z', 'Y').to_euler()
            rot.x += extra_tilt

            cam_obj.rotation_euler = rot
            cam_obj.keyframe_insert(data_path="location")
            cam_obj.keyframe_insert(data_path="rotation_euler")

            frame += 1

    print("Zig-zag animation created across wires.")


# ==============================
# MAIN
# ==============================
def main():
    scene = bpy.context.scene

    # Render stability settings
    scene.render.engine = RENDER_ENGINE
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = RES_PERCENT

    # Get required objects (throws helpful error if missing)
    tree_obj  = get_obj(TREE_OBJ_NAME)
    cam_obj   = get_obj(CAM_NAME)
    post0_obj = get_obj(POST0_NAME)
    post1_obj = get_obj(POST1_NAME)
    wire_objs = [get_obj(n) for n in WIRE_NAMES]

    # Camera basics
    cam_data = cam_obj.data
    cam_data.clip_start = 0.001
    cam_data.lens = 50  # mm

    rgb_dir, depth_dir, ann_dir = ensure_output_dirs(OUTPUT_DIR)

    trunk_points = sample_trunk_points(tree_obj, step=STEP)
    setup_depth_compositor(depth_dir)

    # Animate camera
    frames_per_segment = FRAMES_PER_SEGMENT
    animate_zigzag_wires_camera(
        cam_obj,
        post0_obj,
        post1_obj,
        wire_objs,
        dist_from_surface=DIST_FROM_SURFACE,
        frames_per_segment=frames_per_segment,
        tilt_range=TILT_RANGE
    )

    total_frames = frames_per_segment * len(wire_objs)
    print(f"Rendering {total_frames} frames to: {OUTPUT_DIR}")

    # Render loop
    for f in range(1, total_frames + 1):
        scene.frame_set(f)

        rgb_path = os.path.join(rgb_dir, f"rgb_{f:04d}.png")
        scene.render.filepath = rgb_path

        # Render RGB + compositor (depth EXR)
        bpy.ops.render.render(write_still=True)

        intrinsics = compute_camera_intrinsics(cam_obj, scene)

        ann = {
            "frame": f,
            "rgb_path": os.path.abspath(rgb_path),
            "depth_file_pattern": os.path.abspath(os.path.join(depth_dir, "depth_####.exr")),
            "camera": {
                "location": [float(v) for v in cam_obj.location],
                "rotation_euler": [float(a) for a in cam_obj.rotation_euler],
                "intrinsics": intrinsics,
            },
            "points": trunk_points,
        }

        ann_path = os.path.join(ann_dir, f"frame_{f:04d}.json")
        with open(ann_path, "w") as f_json:
            json.dump(ann, f_json, indent=2)

        # Light progress message (helps you see it isn't frozen)
        if f == 1 or f % 10 == 0 or f == total_frames:
            print(f"  Frame {f}/{total_frames}")

    print("Done! Zig-zag dataset written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
