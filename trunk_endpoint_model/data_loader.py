# trunk_endpoint_model/image_data_loader.py
"""
Data loader for image-based multi-task learning.
Uses RGB images + camera info from generate_dataset.py output.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

class ImageTreeDataset(Dataset):
    """
    Dataset for image-based tree analysis.
    Uses RGB images + camera annotations from generate_dataset.py.
    """
    def __init__(self, 
                 dataset_dir: str,
                 metadata_dir: str,  # Path to trees/metadata/ for ground truth
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
        ann_files = sorted(ann_dir.glob('frame_*.json'))
        
        # Match with tree metadata to get ground truth
        self.samples = []
        for ann_file in ann_files:
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            
            # Extract tree ID from RGB path or frame number
            # You'll need to match this to your tree metadata
            rgb_path = Path(ann_data['rgb_path'])
            frame_num = ann_data['frame']
            
            # Match to tree metadata (you'll need to implement this matching)
            # For now, assume we can find corresponding metadata
            tree_id = self._extract_tree_id(rgb_path, frame_num)
            metadata_file = self.metadata_dir / f"{tree_id}_metadata.json"
            
            if metadata_file.exists():
                self.samples.append({
                    'rgb_path': rgb_path,
                    'depth_path': self._get_depth_path(ann_data, frame_num),
                    'annotation': ann_data,
                    'metadata_file': metadata_file,
                    'frame': frame_num
                })
        
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
    
    def _extract_tree_id(self, rgb_path, frame_num):
        """Extract tree ID from path - implement based on your naming convention"""
        # Example: if rgb_path contains tree info, extract it
        # Or use frame number to match
        return "lpy_envy_00000"  # Placeholder
    
    def _get_depth_path(self, ann_data, frame_num):
        """Get depth file path from annotation"""
        pattern = ann_data['depth_file_pattern']
        return pattern.replace('####', f'{frame_num:04d}')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load RGB image
        rgb = Image.open(sample['rgb_path']).convert('RGB')
        rgb = rgb.resize((self.image_size[1], self.image_size[0]))
        rgb = np.array(rgb).astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # Load depth (if available)
        depth = None
        if Path(sample['depth_path']).exists():
            # Load EXR depth file
            import OpenEXR
            depth = self._load_exr_depth(sample['depth_path'])
        
        # Load camera intrinsics
        camera = sample['annotation']['camera']
        intrinsics = torch.tensor(camera['intrinsics']['K'], dtype=torch.float32)
        camera_pose = {
            'location': torch.tensor(camera['location'], dtype=torch.float32),
            'rotation': torch.tensor(camera['rotation_euler'], dtype=torch.float32),
        }
        
        # Load ground truth from metadata
        with open(sample['metadata_file'], 'r') as f:
            metadata = json.load(f)
        
        # Create ground truth maps
        seg_mask, depth_gt, radius_map, length_map = self._create_ground_truth_maps(
            metadata, camera, self.image_size
        )
        
        return {
            'rgb': rgb,
            'depth': depth,
            'intrinsics': intrinsics,
            'camera_pose': camera_pose,
            'segmentation': seg_mask,
            'depth_gt': depth_gt,
            'radius': radius_map,
            'length': length_map,
            'frame': sample['frame']
        }
    
    def _create_ground_truth_maps(self, metadata, camera, image_size):
        """
        Project cylinder data to image space to create ground truth maps.
        This is a complex function that projects 3D cylinders to 2D image.
        """
        # This would project trunk cylinders to image coordinates
        # and create segmentation mask, depth map, radius map, length map
        # Implementation depends on your specific needs
        H, W = image_size
        seg_mask = np.zeros((H, W), dtype=np.float32)
        depth_gt = np.zeros((H, W), dtype=np.float32)
        radius_map = np.zeros((H, W), dtype=np.float32)
        length_map = np.zeros((H, W), dtype=np.float32)
        
        # Project cylinders to image space
        # (This is a simplified placeholder - full implementation needed)
        
        return (
            torch.from_numpy(seg_mask),
            torch.from_numpy(depth_gt),
            torch.from_numpy(radius_map),
            torch.from_numpy(length_map)
        )