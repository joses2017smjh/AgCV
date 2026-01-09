"""
Image-based data loader for multi-task learning.
Uses RGB images + camera info from generate_dataset.py output.
Integrates with tree metadata for ground truth.
"""
"""
Image-based data loader for multi-task learning.
Uses RGB images + camera info from generate_dataset.py output.
Integrates with tree metadata for ground truth.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
from typing import Dict, List, Tuple


def compute_camera_intrinsics_from_annotation(ann_data):
    """
    Extract camera intrinsics from annotation JSON (same format as generate_dataset.py).
    """
    intrinsics = ann_data['camera']['intrinsics']
    K = np.array(intrinsics['K'], dtype=np.float32)
    return K, intrinsics['width'], intrinsics['height']


def project_3d_to_2d(points_3d, camera_K, camera_location, camera_rotation):
    """
    Project 3D points to 2D image coordinates using camera parameters.
    
    Args:
        points_3d: (N, 3) array of 3D points in world coordinates
        camera_K: (3, 3) camera intrinsic matrix
        camera_location: (3,) camera position in world coordinates
        camera_rotation: (3,) Euler angles (rotation_euler from annotation)
    
    Returns:
        points_2d: (N, 2) array of 2D image coordinates
        depths: (N,) array of depths
    """
    try:
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('xyz', camera_rotation).as_matrix()
    except ImportError:
        # Fallback: simple rotation matrix from Euler angles
        rx, ry, rz = camera_rotation
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)
        
        R = np.array([
            [cos_ry*cos_rz, -cos_ry*sin_rz, sin_ry],
            [cos_rx*sin_rz + sin_rx*sin_ry*cos_rz, cos_rx*cos_rz - sin_rx*sin_ry*sin_rz, -sin_rx*cos_ry],
            [sin_rx*sin_rz - cos_rx*sin_ry*cos_rz, sin_rx*cos_rz + cos_rx*sin_ry*sin_rz, cos_rx*cos_ry]
        ])
    
    # Transform points to camera coordinates
    points_cam = (R @ (points_3d - camera_location).T).T
    
    # Project to image plane
    points_homogeneous = points_cam[:, :3]  # (N, 3)
    depths = points_homogeneous[:, 2]
    
    # Avoid division by zero
    valid = depths > 0
    points_2d = np.zeros((len(points_3d), 2))
    
    if valid.any():
        points_2d[valid] = (camera_K @ points_homogeneous[valid].T).T[:, :2]
        points_2d[valid] /= depths[valid, np.newaxis]
    
    return points_2d, depths


def create_ground_truth_maps(metadata_path, camera_ann, image_size):
    """
    Create ground truth maps by projecting trunk cylinders to image space.
    
    Args:
        metadata_path: Path to tree metadata JSON
        camera_ann: Camera annotation dict from frame JSON
        image_size: (H, W) target image size
    
    Returns:
        seg_mask: (H, W) binary segmentation mask
        depth_map: (H, W) depth values
        radius_map: (H, W) radius values
        length_map: (H, W) length values
    """
    H, W = image_size
    
    # Initialize maps
    seg_mask = np.zeros((H, W), dtype=np.float32)
    depth_map = np.zeros((H, W), dtype=np.float32)
    radius_map = np.zeros((H, W), dtype=np.float32)
    length_map = np.zeros((H, W), dtype=np.float32)
    
    # Load tree metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract trunk cylinders
    cylinder_data = metadata.get('cylinder_data', {})
    trunk_cylinders = []
    
    for color_key, cylinder_info in cylinder_data.items():
        if isinstance(cylinder_info, dict):
            part_name = cylinder_info.get('part_name', '')
            if 'trunk' in part_name.lower():
                trunk_cylinders.append({
                    'centroid': np.array(cylinder_info['centroid']),
                    'radius': cylinder_info['radius'],
                    'length': cylinder_info['length'],
                    'orientation': np.array(cylinder_info['orientation'])
                })
    
    if len(trunk_cylinders) == 0:
        return seg_mask, depth_map, radius_map, length_map
    
    # Get camera parameters
    camera_K, img_w, img_h = compute_camera_intrinsics_from_annotation(camera_ann)
    camera_location = np.array(camera_ann['camera']['location'])
    camera_rotation = np.array(camera_ann['camera']['rotation_euler'])
    
    # Project each cylinder to image space
    for cyl in trunk_cylinders:
        centroid = cyl['centroid']
        radius = cyl['radius']
        length = cyl['length']
        orientation = cyl['orientation']
        orientation = orientation / (np.linalg.norm(orientation) + 1e-9)
        
        # Create cylinder endpoints
        e_minus = centroid - (length / 2) * orientation
        e_plus = centroid + (length / 2) * orientation
        
        # Project to image
        points_3d = np.array([e_minus, e_plus, centroid])
        points_2d, depths = project_3d_to_2d(
            points_3d, camera_K, camera_location, camera_rotation
        )
        
        # Check if points are in image bounds
        valid = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_w) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_h) & \
                (depths > 0)
        
        if valid.any():
            # Scale to target image size
            scale_x = W / img_w
            scale_y = H / img_h
            points_2d_scaled = points_2d.copy()
            points_2d_scaled[:, 0] *= scale_x
            points_2d_scaled[:, 1] *= scale_y
            
            # Draw cylinder as line segment in image
            for i in range(len(points_2d_scaled) - 1):
                if valid[i] and valid[i+1]:
                    pt1 = tuple(points_2d_scaled[i].astype(int))
                    pt2 = tuple(points_2d_scaled[i+1].astype(int))
                    
                    # Draw line with thickness proportional to radius
                    thickness = max(1, int(radius * scale_x * 10))
                    cv2.line(seg_mask, pt1, pt2, 1.0, thickness)
                    
                    # Interpolate depth, radius, length along line
                    num_points = max(10, int(np.linalg.norm(points_2d_scaled[i+1] - points_2d_scaled[i])))
                    for j in range(num_points):
                        t = j / num_points
                        pt = (points_2d_scaled[i] * (1-t) + points_2d_scaled[i+1] * t).astype(int)
                        if 0 <= pt[0] < W and 0 <= pt[1] < H:
                            seg_mask[pt[1], pt[0]] = 1.0
                            depth_map[pt[1], pt[0]] = depths[i] * (1-t) + depths[i+1] * t
                            radius_map[pt[1], pt[0]] = radius
                            length_map[pt[1], pt[0]] = length
    
    return seg_mask, depth_map, radius_map, length_map


class ImageTreeDataset(Dataset):
    """
    Dataset for image-based tree analysis.
    Uses RGB images + camera annotations from generate_dataset.py.
    """
    def __init__(self, 
                 dataset_dir: str,
                 metadata_dir: str,
                 split: str = 'train',
                 image_size: tuple = (480, 640)):
        """
        Args:
            dataset_dir: Directory with rgb/, depth/, ann/ from generate_dataset.py
            metadata_dir: Directory with tree metadata JSONs (for ground truth)
            split: 'train', 'val', 'test'
            image_size: (H, W) for resizing images
        """
        self.dataset_dir = Path(dataset_dir)
        self.metadata_dir = Path(metadata_dir)
        self.image_size = image_size
        
        # Load all annotation files
        ann_dir = self.dataset_dir / 'ann'
        if not ann_dir.exists():
            raise ValueError(f"Annotation directory not found: {ann_dir}")
        
        ann_files = sorted(ann_dir.glob('frame_*.json'))
        
        if len(ann_files) == 0:
            raise ValueError(f"No annotation files found in {ann_dir}")
        
        # Match with tree metadata to get ground truth
        self.samples = []
        for ann_file in ann_files:
            try:
                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)
                
                # Extract tree ID - try to match with metadata files
                # For now, we'll try to find matching metadata based on frame or path
                rgb_path = Path(ann_data['rgb_path'])
                frame_num = ann_data['frame']
                
                # Try to find corresponding metadata file
                # This is a simplified matching - you may need to adjust based on your naming
                metadata_files = list(self.metadata_dir.glob('*_metadata.json'))
                
                if len(metadata_files) > 0:
                    # Use first available metadata file for now
                    # In practice, you'd match based on tree ID in the annotation
                    metadata_file = metadata_files[0]  # Simplified - should match properly
                    
                    self.samples.append({
                        'rgb_path': rgb_path,
                        'depth_path': self._get_depth_path(ann_data, frame_num),
                        'annotation': ann_data,
                        'metadata_file': metadata_file,
                        'frame': frame_num
                    })
            except Exception as e:
                print(f"Warning: Skipping {ann_file}: {e}")
                continue
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples found!")
        
        # Split dataset
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        n_train = int(0.8 * len(self.samples))
        n_val = int(0.1 * len(self.samples))
        
        if split == 'train':
            self.samples = [self.samples[i] for i in indices[:n_train]]
        elif split == 'val':
            self.samples = [self.samples[i] for i in indices[n_train:n_train+n_val]]
        else:
            self.samples = [self.samples[i] for i in indices[n_train+n_val:]]
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _get_depth_path(self, ann_data, frame_num):
        """Get depth file path from annotation"""
        pattern = ann_data['depth_file_pattern']
        depth_file = pattern.replace('####', f'{frame_num:04d}')
        return Path(depth_file)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load RGB image
        rgb = Image.open(sample['rgb_path']).convert('RGB')
        rgb = rgb.resize((self.image_size[1], self.image_size[0]))
        rgb = np.array(rgb).astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Load camera intrinsics
        camera = sample['annotation']['camera']
        intrinsics = torch.tensor(camera['intrinsics']['K'], dtype=torch.float32)
        camera_pose = {
            'location': torch.tensor(camera['location'], dtype=torch.float32),
            'rotation': torch.tensor(camera['rotation_euler'], dtype=torch.float32),
        }
        
        # Create camera parameter vector for model
        # Format: [location(3), rotation(3), K_diag(3)] = 9 elements
        K_diag = torch.tensor([
            intrinsics[0, 0],  # fx
            intrinsics[1, 1],  # fy
            intrinsics[0, 2],  # cx (or could use cy)
        ], dtype=torch.float32)
        
        camera_params = torch.cat([
            camera_pose['location'],
            camera_pose['rotation'],
            K_diag
        ])
        
        # Create ground truth maps
        seg_mask, depth_gt, radius_map, length_map = create_ground_truth_maps(
            sample['metadata_file'],
            sample['annotation'],
            self.image_size
        )
        
        return {
            'rgb': rgb,
            'intrinsics': intrinsics,
            'camera_params': camera_params,
            'camera_pose': camera_pose,
            'segmentation': torch.from_numpy(seg_mask).long(),
            'depth_gt': torch.from_numpy(depth_gt).unsqueeze(0),  # (1, H, W)
            'radius': torch.from_numpy(radius_map).unsqueeze(0),  # (1, H, W)
            'length': torch.from_numpy(length_map).unsqueeze(0),  # (1, H, W)
            'frame': sample['frame']
        }

