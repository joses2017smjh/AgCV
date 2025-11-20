import os
import json
import math
import numpy as np
import open3d as o3d

# Enable EXR support in OpenCV BEFORE importing cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


# ==============================
# CONFIG
# ==============================

BASE_DIR = r"C:\Users\joses\Desktop\tree_dataset"
ANN_DIR = os.path.join(BASE_DIR, "ann")
DEPTH_DIR = os.path.join(BASE_DIR, "depth")
RGB_DIR = os.path.join(BASE_DIR, "rgb")

# Ignore crazy far depths (background / infinity)
MAX_DEPTH = 10.0   # tweak depending on your scene scale


# ==============================
# UTILS
# ==============================

def euler_xyz_to_matrix(rx, ry, rz):
    """Build rotation matrix from Blender-style XYZ Euler angles (radians)."""
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])

    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,   0,  1]])

    # Blender XYZ order: X then Y then Z -> Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    return R


def load_depth_exr(path):
    """Load EXR depth as float32 (single channel)."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth file: {path}")

    # If EXR has multiple channels, take the first
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    depth = depth.astype(np.float32)
    print(f"{os.path.basename(path)} -> shape={depth.shape}, "
          f"min={depth.min():.3f}, max={depth.max():.3f}")
    return depth


def backproject_depth_to_points(depth, K, R, t, stride=1):
    """Convert a depth map to 3D points in WORLD coordinates."""
    h, w = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)

    z = depth[vv, uu]

    # Keep only valid & not too far depths
    mask = (z > 0) & (z < MAX_DEPTH)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    uu = uu[mask].astype(np.float32)
    vv = vv[mask].astype(np.float32)
    z = z[mask].astype(np.float32)

    # Backproject to camera coordinates
    x_cam = (uu - cx) * z / fx
    y_cam = (vv - cy) * z / fy
    z_cam = z

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # Camera coords -> world coords: p_w = R * p_c + t
    pts_world = (R @ pts_cam.T).T + t.reshape(1, 3)

    return pts_world.astype(np.float32)


# ==============================
# ANALYSIS
# ==============================

def analyze_frame(frame_num, stride=8):
    """Analyze a single frame and return its point cloud."""
    ann_path = os.path.join(ANN_DIR, f"frame_{frame_num:04d}.json")
    depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_num:04d}.exr")
    rgb_path = os.path.join(RGB_DIR, f"rgb_{frame_num:04d}.png")
    
    print(f"\n{'='*60}")
    print(f"Analyzing Frame {frame_num}")
    print(f"{'='*60}")
    
    # Load annotation
    with open(ann_path, "r") as f:
        ann = json.load(f)
    
    # Load depth
    depth = load_depth_exr(depth_path)
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"Non-zero pixels: {np.sum(depth > 0)} / {depth.size}")
    
    # Camera info
    intr = ann["camera"]["intrinsics"]
    K = np.array(intr["K"], dtype=np.float32)
    loc = np.array(ann["camera"]["location"], dtype=np.float32)
    rot_euler = np.array(ann["camera"]["rotation_euler"], dtype=np.float32)
    R = euler_xyz_to_matrix(rot_euler[0], rot_euler[1], rot_euler[2])
    
    print(f"\nCamera location: {loc}")
    print(f"Camera rotation (euler): {rot_euler}")
    print(f"\nIntrinsics K:\n{K}")
    print(f"\nRotation matrix R:\n{R}")
    
    # Backproject to world
    pts_world = backproject_depth_to_points(depth, K, R, loc, stride=stride)
    print(f"\nBackprojected points: {pts_world.shape[0]}")
    
    if pts_world.shape[0] > 0:
        print("Point cloud bounds:")
        print(f"  X: [{pts_world[:, 0].min():.3f}, {pts_world[:, 0].max():.3f}]")
        print(f"  Y: [{pts_world[:, 1].min():.3f}, {pts_world[:, 1].max():.3f}]")
        print(f"  Z: [{pts_world[:, 2].min():.3f}, {pts_world[:, 2].max():.3f}]")
    
    # Create point cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts_world)
    
    # Add colors from RGB if available
    if os.path.exists(rgb_path) and pts_world.shape[0] > 0:
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        us = np.arange(0, depth.shape[1], stride)
        vs = np.arange(0, depth.shape[0], stride)
        uu, vv = np.meshgrid(us, vs)
        z = depth[vv, uu]
        mask = (z > 0) & (z < MAX_DEPTH)

        colors = rgb[vv[mask], uu[mask]] / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return cloud, ann


def main():
    # Analyze a few sample frames
    frames_to_check = [30, 50 ]
    
    clouds = []
    for frame_num in frames_to_check:
        ann_path = os.path.join(ANN_DIR, f"frame_{frame_num:04d}.json")
        if not os.path.exists(ann_path):
            print(f"Frame {frame_num} doesn't exist, skipping...")
            continue
        
        try:
            cloud, ann = analyze_frame(frame_num, stride=16)
            clouds.append(cloud)
        except Exception as e:
            print(f"Error processing frame {frame_num}: {e}")
    
    if clouds:
        print(f"\n{'='*60}")
        print(f"Visualizing {len(clouds)} frames together")
        print(f"Each frame will be shown in different positions")
        print(f"{'='*60}")
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        
        o3d.visualization.draw_geometries(
            clouds + [coord_frame],
            window_name="Multi-Frame Point Cloud",
            width=1200, height=800,
        )


if __name__ == "__main__":
    main()
