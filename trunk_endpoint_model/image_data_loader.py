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
        camera_rotation: (3,) Euler angles (rotation_euler from annotation) - world to camera rotation
    
    Returns:
        points_2d: (N, 2) array of 2D image coordinates
        depths: (N,) array of depths
    """
    try:
        from scipy.spatial.transform import Rotation
        # Blender's rotation_euler gives world-to-camera rotation
        R_world_to_cam = Rotation.from_euler('XYZ', camera_rotation, degrees=False).as_matrix()
    except ImportError:
        # Fallback: simple rotation matrix from Euler angles (XYZ order for Blender)
        rx, ry, rz = camera_rotation
        cos_rx, sin_rx = np.cos(rx), np.sin(rx)
        cos_ry, sin_ry = np.cos(ry), np.sin(ry)
        cos_rz, sin_rz = np.cos(rz), np.sin(rz)
        
        # Rotation matrix for XYZ Euler angles (Blender convention)
        R_world_to_cam = np.array([
            [cos_ry*cos_rz, -cos_ry*sin_rz, sin_ry],
            [cos_rx*sin_rz + sin_rx*sin_ry*cos_rz, cos_rx*cos_rz - sin_rx*sin_ry*sin_rz, -sin_rx*cos_ry],
            [sin_rx*sin_rz - cos_rx*sin_ry*cos_rz, sin_rx*cos_rz + cos_rx*sin_ry*sin_rz, cos_rx*cos_ry]
        ])
    
    # Transform points from world to camera coordinates
    # Blender's camera rotation is the rotation from world to camera
    # But we need camera-to-world rotation for the transformation
    # So we invert it: R_cam_to_world = R_world_to_cam^T
    R_cam_to_world = R_world_to_cam.T
    
    # Transform: point_cam = R_cam_to_world^T * (point_world - camera_location)
    # Which is: point_cam = R_world_to_cam * (point_world - camera_location)
    points_relative = points_3d - camera_location
    points_cam = (R_world_to_cam @ points_relative.T).T
    
    # In Blender: camera looks down -Z, so depth is -Z_cam
    # But standard CV: camera looks down +Z, so we might need to flip
    # Let's try using -Z as depth (Blender convention)
    depths = -points_cam[:, 2]  # Negative Z is forward in Blender
    
    # Project to image plane (only points in front of camera)
    valid = depths > 0
    points_2d = np.zeros((len(points_3d), 2))
    
    if valid.any():
        # For projection, use standard camera coordinates
        # Blender camera: X right, Y up, Z backward (so -Z is forward)
        # Standard CV: X right, Y down, Z forward
        # So we need: u = fx * X/Z, v = fy * Y/Z, but Z = -Z_blender
        # Actually, let's use the Blender convention directly
        X_cam = points_cam[valid, 0]
        Y_cam = points_cam[valid, 1]
        Z_cam = -points_cam[valid, 2]  # Forward is -Z in Blender
        
        # Project: [u, v] = K * [X, Y, -Z] / (-Z) = K * [X, Y, -Z] / Z_forward
        points_2d[valid, 0] = (camera_K[0, 0] * X_cam + camera_K[0, 2] * Z_cam) / Z_cam
        points_2d[valid, 1] = (camera_K[1, 1] * Y_cam + camera_K[1, 2] * Z_cam) / Z_cam
    
    return points_2d, depths


def create_ground_truth_maps(metadata_path, camera_ann, image_size, debug=False):
    """
    Create ground truth maps by projecting trunk cylinders to image space.
    
    Args:
        metadata_path: Path to tree metadata JSON
        camera_ann: Camera annotation dict from frame JSON (should have 'points' for world coords)
        image_size: (H, W) target image size
        debug: If True, print debug information
    
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
    
    # Use trunk points from annotation if available (they're already in world coords)
    # Otherwise, estimate tree world position from first trunk point
    use_annotation_points = 'points' in camera_ann and len(camera_ann['points']) > 0
    
    if use_annotation_points:
        # Use trunk points directly - they're already in world coordinates
        trunk_points_world = [np.array(p['pos_world']) for p in camera_ann['points']]
        if len(trunk_points_world) > 0:
            # Estimate tree world offset from first point
            first_point_world = trunk_points_world[0]
            first_cylinder_local = trunk_cylinders[0]['centroid']
            tree_world_offset = first_point_world - first_cylinder_local
            if debug:
                print(f"Using {len(trunk_points_world)} trunk points from annotation")
                print(f"Estimated tree world offset: {tree_world_offset}")
    else:
        tree_world_offset = np.array([0.0, 0.0, 0.0])
    
    # Transform cylinder centroids to world coordinates
    for cyl in trunk_cylinders:
        cyl['centroid'] = cyl['centroid'] + tree_world_offset
    
    # Get camera parameters
    camera_K, img_w, img_h = compute_camera_intrinsics_from_annotation(camera_ann)
    camera_location = np.array(camera_ann['camera']['location'])
    camera_rotation = np.array(camera_ann['camera']['rotation_euler'])
    
    # Debug: Track projection statistics
    total_cylinders = len(trunk_cylinders)
    projected_count = 0
    in_bounds_count = 0
    
    # Project each cylinder to image space
    for cyl_idx, cyl in enumerate(trunk_cylinders):
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
        
        # Debug first few cylinders
        if debug and cyl_idx < 3:
            print(f"  Cylinder {cyl_idx}: centroid={centroid}, depth={depths}, "
                  f"2d={points_2d}, valid={valid}, in_bounds={valid.any()}")
        
        if valid.any():
            projected_count += 1
            in_bounds_count += 1
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
        elif depths[valid].any():  # Projected but outside bounds
            projected_count += 1
    
    # Debug output
    if debug and total_cylinders > 0:
        print(f"Projection stats: {total_cylinders} trunk cylinders, "
              f"{projected_count} projected, {in_bounds_count} in image bounds")
        if in_bounds_count == 0:
            print(f"  WARNING: No cylinders projected into image bounds!")
            print(f"  Camera location: {camera_location}")
            print(f"  Camera rotation: {camera_rotation}")
            print(f"  Image size: {img_w}x{img_h}")
            # Sample a few cylinder centroids
            if len(trunk_cylinders) > 0:
                sample_centroids = [cyl['centroid'] for cyl in trunk_cylinders[:3]]
                print(f"  Sample cylinder centroids: {sample_centroids}")
    
    # Fallback: If no cylinders projected, try using trunk points from annotation
    if in_bounds_count == 0 and use_annotation_points and len(trunk_points_world) > 0:
        if debug:
            print(f"  Fallback: Using {len(trunk_points_world)} trunk points from annotation")
        
        # Project trunk points
        points_3d = np.array(trunk_points_world)
        points_2d, depths = project_3d_to_2d(
            points_3d, camera_K, camera_location, camera_rotation
        )
        
        # Check which points are in bounds
        valid = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_w) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_h) & \
                (depths > 0)
        
        if valid.any():
            if debug:
                print(f"  {valid.sum()} trunk points are in image bounds")
            
            # Scale to target image size
            scale_x = W / img_w
            scale_y = H / img_h
            points_2d_scaled = points_2d[valid] * np.array([scale_x, scale_y])
            depths_valid = depths[valid]
            
            # Draw points as small circles
            for pt_2d, depth_val in zip(points_2d_scaled, depths_valid):
                pt = tuple(pt_2d.astype(int))
                if 0 <= pt[0] < W and 0 <= pt[1] < H:
                    # Draw small circle (radius ~3 pixels)
                    cv2.circle(seg_mask, pt, 3, 1.0, -1)
                    depth_map[pt[1], pt[0]] = depth_val
                    # Use average radius/length from cylinders
                    if len(trunk_cylinders) > 0:
                        avg_radius = np.mean([c['radius'] for c in trunk_cylinders])
                        avg_length = np.mean([c['length'] for c in trunk_cylinders])
                        radius_map[pt[1], pt[0]] = avg_radius
                        length_map[pt[1], pt[0]] = avg_length
    
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
    
    def _load_exr_depth(self, depth_path):
        """Load depth from EXR file."""
        try:
            import OpenEXR
            import Imath
            
            exr_file = OpenEXR.InputFile(str(depth_path))
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # Check available channels
            available_channels = list(header['channels'].keys())
            
            # Try common depth channel names (R, Y, or first available)
            depth_channel = None
            for channel_name in ['R', 'Y', 'Z', 'Depth']:
                if channel_name in available_channels:
                    depth_channel = channel_name
                    break
            
            # If none found, use first channel
            if depth_channel is None and len(available_channels) > 0:
                depth_channel = available_channels[0]
            
            if depth_channel is None:
                exr_file.close()
                return None
            
            # Read depth channel
            depth_str = exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))
            depth = np.frombuffer(depth_str, dtype=np.float32)
            depth = depth.reshape((height, width))
            
            exr_file.close()
            return depth
        except Exception as e:
            print(f"Warning: Could not load EXR depth from {depth_path}: {e}")
            return None
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load RGB image
        rgb = Image.open(sample['rgb_path']).convert('RGB')
        rgb = rgb.resize((self.image_size[1], self.image_size[0]))
        rgb = np.array(rgb).astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Load depth map (already rendered - contains tree geometry in 2D!)
        depth_exr = self._load_exr_depth(sample['depth_path'])
        if depth_exr is not None:
            # Resize depth to match image size
            from scipy.ndimage import zoom
            depth_exr = zoom(depth_exr, (self.image_size[0] / depth_exr.shape[0], 
                                        self.image_size[1] / depth_exr.shape[1]), 
                            order=1)
        else:
            depth_exr = np.zeros(self.image_size, dtype=np.float32)
        
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
        
        # Create ground truth maps from depth map (much simpler - no projection needed!)
        # Segmentation: reasonable depth values = tree pixel
        # Filter out very high values (likely background/sentinel values)
        # Typical depth should be in reasonable range (e.g., 0.1 to 100 meters)
        max_reasonable_depth = 100.0  # Adjust based on your scene scale
        seg_mask = ((depth_exr > 0) & (depth_exr < max_reasonable_depth)).astype(np.float32)
        depth_gt = depth_exr.astype(np.float32)
        # Set background depth to 0
        depth_gt[seg_mask == 0] = 0.0
        
        # For radius and length, use average values from metadata cylinders
        radius_map = np.zeros_like(depth_gt)
        length_map = np.zeros_like(depth_gt)
        
        try:
            # Load metadata to get average radius/length for tree pixels
            with open(sample['metadata_file'], 'r') as f:
                metadata = json.load(f)
            cylinder_data = metadata.get('cylinder_data', {})
            trunk_cylinders = []
            for color_key, cylinder_info in cylinder_data.items():
                if isinstance(cylinder_info, dict):
                    part_name = cylinder_info.get('part_name', '')
                    if 'trunk' in part_name.lower():
                        trunk_cylinders.append(cylinder_info)
            
            if len(trunk_cylinders) > 0:
                avg_radius = np.mean([c['radius'] for c in trunk_cylinders])
                avg_length = np.mean([c['length'] for c in trunk_cylinders])
                # Assign to all tree pixels
                radius_map = seg_mask * avg_radius
                length_map = seg_mask * avg_length
            else:
                # Default values if no trunk cylinders found
                radius_map = seg_mask * 0.01  # 1cm default
                length_map = seg_mask * 0.1   # 10cm default
        except Exception as e:
            # Fallback: use default values
            radius_map = seg_mask * 0.01
            length_map = seg_mask * 0.1
        
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

