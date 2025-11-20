import os
import json
import glob
import math

import numpy as np
import open3d as o3d
import networkx as nx

# Enable EXR support in OpenCV BEFORE importing cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


# ==============================
# CONFIG
# ==============================

BASE_DIR = r"C:\Users\joses\Desktop\tree_dataset"
ANN_DIR = os.path.join(BASE_DIR, "ann")
DEPTH_DIR = os.path.join(BASE_DIR, "depth")

# Only use every Nth pixel to avoid millions of points
PIXEL_STRIDE = 8        # 8 = keep 1/64 of pixels
MAX_FRAMES = 120        # set to None to use ALL frames

# Voxel size for downsampling (meters in your Blender units)
VOXEL_SIZE = 0.05       # 5 cm; tweak as needed

# Ignore crazy far depths (background / infinity)
MAX_DEPTH = 2      # adjust based on scene scale

# For kNN graph
KNN_K = 6

# Statistical outlier removal (currently unused, but kept for future)
REMOVE_OUTLIERS = True
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0


# ==============================
# UTILS
# ==============================

def euler_xyz_to_matrix(rx, ry, rz):
    """
    Build rotation matrix from Blender-style XYZ Euler angles (radians).
    """
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)

    # Rotation matrices for X, Y, Z
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]])

    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,   0,  1]])

    # Blender 'XYZ' means rotate X, then Y, then Z:
    # Combined matrix is R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def load_depth_exr(path):
    """
    Load EXR depth as float32 (single channel) using OpenCV.
    """
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth file: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    print(f"{os.path.basename(path)} -> min={depth.min():.3f}, max={depth.max():.3f}")
    return depth


def backproject_depth_to_points(depth, K, R, t, stride=1):
    """
    Convert a depth map + intrinsics + extrinsics into world-space 3D points.

    depth: HxW depth array (float, distance from camera)
    K: 3x3 intrinsic matrix
    R: 3x3 rotation (camera to world)
    t: 3-vector translation (camera location in world coords)
    stride: sample every 'stride' pixels to reduce density
    """
    h, w = depth.shape

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Create grid of pixel coordinates
    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)

    z = depth[vv, uu]  # sampled depth

    # Mask out invalid depths (zero, negative, or too far)
    mask = (z > 0) & (z < MAX_DEPTH)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    uu = uu[mask].astype(np.float32)
    vv = vv[mask].astype(np.float32)
    z = z[mask].astype(np.float32)

    # Backproject to camera coordinates (approx pinhole)
    x_cam = (uu - cx) * z / fx
    y_cam = (vv - cy) * z / fy
    z_cam = z

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)

    # Transform to world coordinates: p_world = R * p_cam + t
    pts_world = (R @ pts_cam.T).T + t.reshape(1, 3)

    return pts_world.astype(np.float32)


# ==============================
# STEP 1: Fuse all frames into a global point cloud
# ==============================

def build_global_point_cloud():
    ann_files = sorted(glob.glob(os.path.join(ANN_DIR, "frame_*.json")))
    print(f"DEBUG: Found {len(ann_files)} annotation files in {ANN_DIR}")
    print(f"DEBUG: DEPTH_DIR = {DEPTH_DIR}")
    if MAX_FRAMES is not None:
        ann_files = ann_files[:MAX_FRAMES]

    all_points = []

    for ann_path in ann_files:
        with open(ann_path, "r") as f:
            ann = json.load(f)

        frame = ann["frame"]

        # Depth file naming: depth_0001.exr, depth_0002.exr, ...
        depth_path = os.path.join(DEPTH_DIR, f"depth_{frame:04d}.exr")
        if not os.path.isfile(depth_path):
            print(f"[WARN] Missing depth for frame {frame}: {depth_path}")
            continue

        depth = load_depth_exr(depth_path)

        intr = ann["camera"]["intrinsics"]
        K = np.array(intr["K"], dtype=np.float32)

        loc = np.array(ann["camera"]["location"], dtype=np.float32)
        rot_euler = np.array(ann["camera"]["rotation_euler"], dtype=np.float32)
        R = euler_xyz_to_matrix(rot_euler[0], rot_euler[1], rot_euler[2])

        pts_world = backproject_depth_to_points(depth, K, R, loc, stride=PIXEL_STRIDE)
        if pts_world.shape[0] == 0:
            print(f"[INFO] Frame {frame}: no valid 3D points after masking")
            continue

        all_points.append(pts_world)
        print(f"Frame {frame}: added {pts_world.shape[0]} points")

    if not all_points:
        raise RuntimeError("No points loaded from any frame. Check depth & MAX_DEPTH!")

    all_points = np.vstack(all_points)
    print(f"Total fused points (before downsample): {all_points.shape[0]}")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(all_points)

    # Downsample
    print(f"Downsampling with voxel size {VOXEL_SIZE}...")
    cloud = cloud.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"Points after downsample: {np.asarray(cloud.points).shape[0]}")

    return cloud


# ==============================
# STEP 2: Build graph + MST skeleton
# ==============================

def build_knn_graph_from_cloud(cloud, k=6):
    pts = np.asarray(cloud.points)
    n = pts.shape[0]
    print(f"Building kNN graph with {n} points, k={k}...")

    kdtree = o3d.geometry.KDTreeFlann(cloud)

    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    for i in range(n):
        _, idxs, dists2 = kdtree.search_knn_vector_3d(cloud.points[i], k)
        for j, d2 in zip(idxs[1:], dists2[1:]):  # skip self at idx[0]
            dist = math.sqrt(d2)
            G.add_edge(i, j, weight=dist)

    print(f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def prune_mst_branches(T, pts, min_branch_length=0.3):
    """
    Prune short branches from the MST to get a cleaner skeleton.
    
    min_branch_length: minimum path length (in meters) to keep a branch
    """
    print(f"Pruning short branches (< {min_branch_length}m)...")
    
    pruned = True
    iterations = 0
    while pruned and iterations < 100:
        pruned = False
        iterations += 1
        
        # Find leaf nodes (degree 1)
        leaves = [n for n in T.nodes() if T.degree(n) == 1]
        
        for leaf in leaves:
            if leaf not in T:
                continue
                
            # Trace path from leaf until we hit a branch point (degree > 2)
            path = [leaf]
            current = leaf
            path_length = 0.0
            
            while T.degree(current) <= 2:
                neighbors = list(T.neighbors(current))
                if not neighbors:
                    break
                    
                # Get next node (not the previous one)
                next_node = None
                for n in neighbors:
                    if len(path) < 2 or n != path[-2]:
                        next_node = n
                        break
                
                if next_node is None:
                    break
                
                # Add edge length
                edge_data = T.get_edge_data(current, next_node)
                path_length += edge_data.get('weight', 0)
                path.append(next_node)
                current = next_node
                
                # Stop if we've reached a branch point
                if T.degree(current) > 2:
                    break
            
            # Remove this branch if it's too short
            if path_length < min_branch_length and len(path) > 1:
                T.remove_nodes_from(path[:-1])  # Keep the branch point
                pruned = True
    
    print(f"Pruned MST has {T.number_of_nodes()} nodes, {T.number_of_edges()} edges")
    return T


def compute_mst_skeleton(cloud, k=KNN_K):
    pts = np.asarray(cloud.points)
    if pts.shape[0] == 0:
        raise RuntimeError("Point cloud is empty before MST.")

    G = build_knn_graph_from_cloud(cloud, k=k)
    print("Computing Minimum Spanning Tree...")
    T = nx.minimum_spanning_tree(G, weight='weight')
    print(f"MST has {T.number_of_nodes()} nodes, {T.number_of_edges()} edges")
    
    # Prune small branches
    T = prune_mst_branches(T, pts, min_branch_length=0.3)

    # Build skeleton points: each MST node -> pos + dir (average of neighbor directions)
    skeleton_points = []
    for i in T.nodes:
        p = pts[i]
        neighbors = list(T.neighbors(i))
        if neighbors:
            dirs = []
            for j in neighbors:
                v = pts[j] - p
                norm = np.linalg.norm(v)
                if norm > 1e-6:
                    dirs.append(v / norm)
            if dirs:
                dir_vec = np.mean(dirs, axis=0)
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-6:
                    dir_vec = dir_vec / norm
                else:
                    dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            dir_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        skeleton_points.append({
            "id": int(i),
            "pos_world": [float(p[0]), float(p[1]), float(p[2])],
            "dir_world": [float(dir_vec[0]), float(dir_vec[1]), float(dir_vec[2])]
        })

    return skeleton_points, T


# ==============================
# STEP 3: Save skeleton to JSON
# ==============================

def save_skeleton_json(skeleton_points, mst, out_path):
    # Save edges for proper tree structure visualization
    edges = [[int(i), int(j)] for i, j in mst.edges()]
    
    data = {
        "num_points": len(skeleton_points),
        "num_edges": len(edges),
        "points": skeleton_points,
        "edges": edges
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Skeleton saved to {out_path}")


# ==============================
# MAIN
# ==============================

def main():
    # 1) Build global cloud
    cloud = build_global_point_cloud()

    # Visualize the fused cloud
    print("Showing fused point cloud (close window to continue)...")
    o3d.visualization.draw_geometries([cloud])

    # 2) Skeleton via MST
    skeleton_points, mst = compute_mst_skeleton(cloud, k=KNN_K)

    # 3) Save skeleton
    out_skel = os.path.join(BASE_DIR, "skeleton_mst.json")
    save_skeleton_json(skeleton_points, mst, out_skel)

    # Visualize skeleton as red points overlaid on gray cloud
    skel_pts = np.array([p["pos_world"] for p in skeleton_points], dtype=np.float32)
    skel_cloud = o3d.geometry.PointCloud()
    skel_cloud.points = o3d.utility.Vector3dVector(skel_pts)
    skel_cloud.paint_uniform_color([1.0, 0.0, 0.0])

    print("Showing fused cloud (gray) and MST skeleton (red)...")
    cloud.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([cloud, skel_cloud])


if __name__ == "__main__":
    main()
