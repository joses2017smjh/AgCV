import os
import json
import math
import numpy as np
import open3d as o3d
import imageio
import cv2

BASE_DIR = r"C:\Users\joses\Desktop\tree_dataset"
ANN_DIR = os.path.join(BASE_DIR, "ann")
DEPTH_DIR = os.path.join(BASE_DIR, "depth")
RGB_DIR = os.path.join(BASE_DIR, "rgb")

# Change this to view different frames
FRAME_TO_VIEW = 1

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

    R = Rz @ Ry @ Rx
    return R


def load_depth_exr(path):
    """Load a single-channel EXR depth file as float32 numpy array."""
    depth = imageio.imread(path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.astype(np.float32)


def backproject_depth_to_points(depth, K, R, t, stride=1):
    """Convert a depth map to 3D points."""
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth[vv, uu]

    mask = z > 0
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    uu = uu[mask].astype(np.float32)
    vv = vv[mask].astype(np.float32)
    z = z[mask].astype(np.float32)

    # Backproject to camera coordinates
    x_cam = (uu - cx) * z / fx
    y_cam = (vv - cy) * z / fy
    z_cam = z

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # Transform to world coordinates
    pts_world = (R @ pts_cam.T).T + t.reshape(1, 3)

    return pts_world.astype(np.float32), pts_cam.astype(np.float32)


def visualize_frame(frame_num, stride=4):
    """Visualize a single frame."""
    ann_path = os.path.join(ANN_DIR, f"frame_{frame_num:04d}.json")
    depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_num:04d}.exr")
    rgb_path = os.path.join(RGB_DIR, f"rgb_{frame_num:04d}.png")
    
    print(f"Loading Frame {frame_num}...")
    
    # Load annotation
    with open(ann_path, "r") as f:
        ann = json.load(f)
    
    # Load depth
    depth = load_depth_exr(depth_path)
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    
    # Camera info
    K = np.array(ann["camera"]["intrinsics"]["K"], dtype=np.float32)
    loc = np.array(ann["camera"]["location"], dtype=np.float32)
    rot_euler = np.array(ann["camera"]["rotation_euler"], dtype=np.float32)
    R = euler_xyz_to_matrix(rot_euler[0], rot_euler[1], rot_euler[2])
    
    print(f"Camera location: {loc}")
    print(f"Camera rotation (degrees): {np.degrees(rot_euler)}")
    
    # Backproject
    pts_world, pts_cam = backproject_depth_to_points(depth, K, R, loc, stride=stride)
    print(f"Generated {pts_world.shape[0]} points")
    
    # Create world-space point cloud
    cloud_world = o3d.geometry.PointCloud()
    cloud_world.points = o3d.utility.Vector3dVector(pts_world)
    
    # Create camera-space point cloud
    cloud_cam = o3d.geometry.PointCloud()
    cloud_cam.points = o3d.utility.Vector3dVector(pts_cam)
    
    # Load RGB and assign colors
    if os.path.exists(rgb_path):
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        us = np.arange(0, depth.shape[1], stride)
        vs = np.arange(0, depth.shape[0], stride)
        uu, vv = np.meshgrid(us, vs)
        z = depth[vv, uu]
        mask = z > 0
        
        colors = rgb[vv[mask], uu[mask]] / 255.0
        cloud_world.colors = o3d.utility.Vector3dVector(colors)
        cloud_cam.colors = o3d.utility.Vector3dVector(colors)
    
    # Create coordinate frames
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=loc)
    camera_frame.rotate(R, center=loc)
    
    print(f"\nShowing point cloud in WORLD coordinates")
    print("Red=X, Green=Y, Blue=Z")
    o3d.visualization.draw_geometries([cloud_world, world_frame, camera_frame],
                                     window_name=f"Frame {frame_num} - World Space",
                                     width=1200, height=800)
    
    print(f"\nShowing point cloud in CAMERA coordinates")
    print("This should look like what the camera sees")
    cam_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([cloud_cam, cam_origin],
                                     window_name=f"Frame {frame_num} - Camera Space",
                                     width=1200, height=800)


if __name__ == "__main__":
    visualize_frame(FRAME_TO_VIEW, stride=4)

