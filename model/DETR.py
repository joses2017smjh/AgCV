"""
DETR-style Cylinder Detection Model for Tree Branch Detection
Predicts cylinders (start_point, end_point, radius) from RGB-D images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
import cv2


# ============================================================================
# 1. MASK UTILITIES - RGB→ID mapping and binary mask extraction
# ============================================================================

class MaskProcessor:
    """Process segmentation masks and convert to binary masks per object"""
    
    def __init__(self, id_channel='blue'):
        """
        Args:
            id_channel: Which channel contains instance ID ('red', 'green', or 'blue')
        """
        self.channel_map = {'red': 0, 'green': 1, 'blue': 2}
        self.id_channel_idx = self.channel_map[id_channel]
    
    def mask_id_to_binary_masks(self, image_mask: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Convert RGB mask to dictionary of binary masks per instance ID
        
        Args:
            image_mask: (H, W, 3) RGB mask where one channel encodes instance IDs
            
        Returns:
            Dict mapping instance_id -> binary_mask (H, W) boolean array
        """
        # Extract ID channel
        id_channel = image_mask[:, :, self.id_channel_idx]
        
        # Get unique IDs (excluding background = 0)
        unique_ids = np.unique(id_channel)
        unique_ids = unique_ids[unique_ids > 0]
        
        binary_masks = {}
        for inst_id in unique_ids:
            binary_masks[int(inst_id)] = (id_channel == inst_id)
        
        return binary_masks
    
    def visualize_masks(self, binary_masks: Dict[int, np.ndarray]) -> np.ndarray:
        """Create colored visualization of all masks"""
        if not binary_masks:
            return None
        
        h, w = next(iter(binary_masks.values())).shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        for idx, (inst_id, mask) in enumerate(binary_masks.items()):
            # Assign unique color to each instance
            color = np.array([
                (inst_id * 50) % 255,
                (inst_id * 100) % 255,
                (inst_id * 150) % 255
            ])
            vis[mask] = color
        
        return vis
    def save_binary_masks(self, binary_masks: Dict[int, np.ndarray], 
                          output_dir: str, frame_id: str):
        """
        Save binary masks to disk
        
        Args:
            binary_masks: Dict of instance_id -> binary mask
            output_dir: Directory to save masks (e.g., 'C:/Users/joses/Desktop/tree_dataset/mask_id')
            frame_id: Frame identifier for naming
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual masks
        for inst_id, mask in binary_masks.items():
            mask_uint8 = (mask * 255).astype(np.uint8)
            output_path = os.path.join(output_dir, f"frame_{frame_id}_id_{inst_id:03d}.png")
            cv2.imwrite(output_path, mask_uint8)
        
        # Also save a combined visualization
        vis = self.visualize_masks(binary_masks)
        if vis is not None:
            vis_path = os.path.join(output_dir, f"frame_{frame_id}_all_masks.png")
            cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        print(f"Saved {len(binary_masks)} masks to {output_dir}")

# ============================================================================
# 2. CYLINDER FITTING - Extract 3D cylinders from masks
# ============================================================================

def backproject_depth_to_points(depth: np.ndarray, K: np.ndarray, 
                                 mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Backproject depth pixels to 3D camera coordinates
    
    Args:
        depth: (H, W) depth map in meters
        K: (3, 3) camera intrinsic matrix
        mask: (H, W) optional boolean mask to only backproject certain pixels
        
    Returns:
        (N, 3) array of 3D points in camera frame
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    if mask is not None:
        u = u[mask]
        v = v[mask]
        z = depth[mask]
    else:
        u = u.flatten()
        v = v.flatten()
        z = depth.flatten()
    
    # Filter valid depths
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]
    
    if len(z) == 0:
        return np.empty((0, 3))
    
    # Backproject to camera coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = np.stack([x, y, z], axis=1)
    return points.astype(np.float32)


def fit_cylinder_pca(points: np.ndarray, min_points: int = 10) -> Optional[Dict]:
    """
    Fit cylinder to 3D points using PCA
    
    Args:
        points: (N, 3) array of 3D points
        min_points: minimum points required for fitting
        
    Returns:
        Dict with keys: 'start', 'end', 'radius', 'axis', 'centroid'
        Returns None if fitting fails
    """
    if len(points) < min_points:
        return None
    
    # Remove outliers using statistical filtering
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    inliers = points[distances < mean_dist + 2 * std_dist]
    
    if len(inliers) < min_points:
        return None
    
    # PCA to find principal axis
    pca = PCA(n_components=3)
    pca.fit(inliers)
    
    # Principal axis is the direction with most variance
    axis = pca.components_[0]
    centroid = np.mean(inliers, axis=0)
    
    # Project points onto axis to find endpoints
    projected = (inliers - centroid) @ axis
    min_proj = np.min(projected)
    max_proj = np.max(projected)
    
    # Cylinder endpoints
    start = centroid + min_proj * axis
    end = centroid + max_proj * axis
    
    # Compute radius: perpendicular distances from axis
    to_points = inliers - centroid
    proj_on_axis = (to_points @ axis).reshape(-1, 1) * axis
    perpendicular = to_points - proj_on_axis
    distances = np.linalg.norm(perpendicular, axis=1)
    radius = np.median(distances)  # Use median for robustness
    
    return {
        'start': start.astype(np.float32),
        'end': end.astype(np.float32),
        'radius': float(radius),
        'axis': axis.astype(np.float32),
        'centroid': centroid.astype(np.float32),
        'length': float(np.linalg.norm(end - start)),
        'num_points': len(inliers)
    }


def mask_to_cylinder(mask: np.ndarray, depth: np.ndarray, K: np.ndarray,
                     R: Optional[np.ndarray] = None, 
                     t: Optional[np.ndarray] = None) -> Optional[Dict]:
    """
    Convert binary mask + depth to cylinder parameters
    
    Args:
        mask: (H, W) binary mask
        depth: (H, W) depth map
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix (camera to world), optional
        t: (3,) translation vector (camera location), optional
        
    Returns:
        Dict with cylinder parameters in world or camera frame
    """
    # Backproject masked region to 3D
    points_cam = backproject_depth_to_points(depth, K, mask)
    
    if len(points_cam) == 0:
        return None
    
    # Fit cylinder in camera frame
    cylinder = fit_cylinder_pca(points_cam)
    
    if cylinder is None:
        return None
    
    # Transform to world frame if R, t provided
    if R is not None and t is not None:
        cylinder['start'] = (R @ cylinder['start']) + t
        cylinder['end'] = (R @ cylinder['end']) + t
        cylinder['axis'] = R @ cylinder['axis']
        cylinder['centroid'] = (R @ cylinder['centroid']) + t
        cylinder['frame'] = 'world'
    else:
        cylinder['frame'] = 'camera'
    
    return cylinder


# ============================================================================
# 3. DETR-STYLE MODEL ARCHITECTURE
# ============================================================================

class CylinderDETR(nn.Module):
    """
    DETR-style model for cylinder detection
    Predicts a fixed set of cylinders from RGB-D input
    """
    
    def __init__(self, 
                 num_queries: int = 100,
                 hidden_dim: int = 256,
                 nheads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 backbone: str = 'resnet50'):
        """
        Args:
            num_queries: Maximum number of cylinders to detect
            hidden_dim: Dimension of transformer
            nheads: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            num_decoder_layers: Number of transformer decoder layers
            backbone: Backbone architecture ('resnet50', 'resnet101', 'vit')
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Backbone for feature extraction
        self.backbone = self._build_backbone(backbone, hidden_dim)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Prediction heads
        self.cylinder_head = CylinderHead(hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding2D(hidden_dim)
    
    def _build_backbone(self, backbone_name: str, hidden_dim: int):
        """Build backbone network"""
        if backbone_name == 'resnet50':
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            # Remove final FC layer
            backbone = nn.Sequential(*list(resnet.children())[:-2])
            # Add projection to hidden_dim
            self.backbone_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
            
            # ADD THIS: Preprocessing layer to handle 4 channels (RGB + Depth)
            self.input_proj = nn.Conv2d(4, 3, kernel_size=1)
            
        elif backbone_name == 'resnet101':
            import torchvision.models as models
            resnet = models.resnet101(pretrained=True)
            backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.backbone_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
            
            # ADD THIS: Preprocessing layer to handle 4 channels (RGB + Depth)
            self.input_proj = nn.Conv2d(4, 3, kernel_size=1)
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented")
        
        return backbone
    
    def forward(self, rgb: torch.Tensor, depth: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            rgb: (B, 3, H, W) RGB images
            depth: (B, 1, H, W) depth maps (optional, can be concatenated)
            
        Returns:
            Dict with:
                'cylinder_params': (B, num_queries, 7) - [x1, y1, z1, x2, y2, z2, radius]
                'confidence': (B, num_queries) - detection confidence
        """
        # Optionally concatenate depth
        if depth is not None:
            x = torch.cat([rgb, depth], dim=1)  # (B, 4, H, W)
            x = self.input_proj(x)
        else:
            x = rgb
        
        # Extract features
        features = self.backbone(x)  # (B, 2048, H', W')
        features = self.backbone_proj(features)  # (B, hidden_dim, H', W')
        
        # Add positional encoding
        B, C, H, W = features.shape
        pos_encoding = self.pos_encoder(H, W).to(features.device)  # (H*W, C)
        pos_encoding = pos_encoding.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, C)
        
        # Flatten spatial dimensions
        features_flat = features.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        
        # Get object queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, C)
        
        # Transformer
        # Encoder
        memory = self.transformer.encoder(features_flat + pos_encoding)
        
        # Decoder
        decoder_out = self.transformer.decoder(queries, memory)  # (B, num_queries, C)
        
        # Prediction heads
        outputs = self.cylinder_head(decoder_out)
        
        return outputs


class CylinderHead(nn.Module):
    """Prediction head for cylinder parameters"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # MLP for cylinder parameters (x1, y1, z1, x2, y2, z2, radius)
        self.cylinder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 7)
        )
        
        # Classification head (object vs no-object)
        self.confidence_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, num_queries, hidden_dim)
            
        Returns:
            Dict with 'cylinder_params' and 'confidence'
        """
        cylinder_params = self.cylinder_mlp(x)  # (B, num_queries, 7)
        confidence = self.confidence_mlp(x).squeeze(-1)  # (B, num_queries)
        
        return {
            'cylinder_params': cylinder_params,
            'confidence': confidence
        }


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for image features"""
    
    def __init__(self, hidden_dim: int, temperature: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
    
    def forward(self, H: int, W: int) -> torch.Tensor:
        """
        Generate 2D positional encoding
        
        Returns:
            (H*W, hidden_dim) positional encoding
        """
        y_embed = torch.arange(H, dtype=torch.float32).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32).unsqueeze(0).repeat(H, 1)
        
        # Normalize to [0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W
        
        dim_t = torch.arange(self.hidden_dim // 2, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.hidden_dim // 2))
        
        pos_x = x_embed.flatten()[:, None] / dim_t
        pos_y = y_embed.flatten()[:, None] / dim_t
        
        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)
        
        pos = torch.cat([pos_y, pos_x], dim=1)  # (H*W, hidden_dim)
        
        return pos


# ============================================================================
# 4. LOSS FUNCTIONS
# ============================================================================

class CylinderLoss(nn.Module):
    """Loss function for cylinder detection"""
    
    def __init__(self, 
                 weight_endpoint: float = 1.0,
                 weight_radius: float = 1.0,
                 weight_confidence: float = 1.0):
        super().__init__()
        self.weight_endpoint = weight_endpoint
        self.weight_radius = weight_radius
        self.weight_confidence = weight_confidence
    
    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            predictions: Dict from model output
            targets: Dict with 'cylinder_params' and 'valid_mask'
            
        Returns:
            Dict with total loss and component losses
        """
        pred_cylinders = predictions['cylinder_params']  # (B, N, 7)
        pred_confidence = predictions['confidence']  # (B, N)
        
        target_cylinders = targets['cylinder_params']  # (B, M, 7)
        target_valid = targets['valid_mask']  # (B, M) boolean
        
        # Hungarian matching between predictions and targets
        # (Simplified - in practice use scipy.optimize.linear_sum_assignment)
        
        # Endpoint loss (L1 distance between start/end points)
        endpoint_loss = F.l1_loss(
            pred_cylinders[..., :6],  # x1, y1, z1, x2, y2, z2
            target_cylinders[..., :6],
            reduction='mean'
        )
        
        # Radius loss
        radius_loss = F.l1_loss(
            pred_cylinders[..., 6],
            target_cylinders[..., 6],
            reduction='mean'
        )
        
        # Confidence loss (binary cross-entropy)
        target_conf = target_valid.float()
        confidence_loss = F.binary_cross_entropy(pred_confidence, target_conf)
        
        total_loss = (
            self.weight_endpoint * endpoint_loss +
            self.weight_radius * radius_loss +
            self.weight_confidence * confidence_loss
        )
        
        return {
            'loss': total_loss,
            'endpoint_loss': endpoint_loss,
            'radius_loss': radius_loss,
            'confidence_loss': confidence_loss
        }


# ============================================================================
# 5. DATASET AND DATA LOADING
# ============================================================================

class TreeCylinderDataset(torch.utils.data.Dataset):
    """Dataset for tree cylinder detection from RGB-D + masks"""
    
    def __init__(self, data_dir: str, transform=None):
        """
        Args:
            data_dir: Directory containing rgb/, depth/, masks/, annotations/
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mask_processor = MaskProcessor(id_channel='blue')
        
        # Load file lists
        self.samples = self._load_samples()
    
    def _load_samples(self):
        """Load list of available samples"""
        import glob
        rgb_files = sorted(glob.glob(f"{self.data_dir}/rgb/*.png"))
        samples = []
        for rgb_path in rgb_files:
            frame_id = rgb_path.split('_')[-1].split('.')[0]
            samples.append({
                'frame_id': frame_id,
                'rgb': rgb_path,
                'depth': f"{self.data_dir}/depth/depth_{frame_id}.exr",
                'mask': f"{self.data_dir}/masks/mask_{frame_id}.png",
                'annotation': f"{self.data_dir}/ann/frame_{frame_id}.json"
            })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load RGB
        rgb = cv2.imread(sample['rgb'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load depth
        depth = cv2.imread(sample['depth'], cv2.IMREAD_UNCHANGED)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        
        # Load mask
        mask_img = cv2.imread(sample['mask'])
        binary_masks = self.mask_processor.mask_id_to_binary_masks(mask_img)
        
        # Load annotations (camera params + ground truth cylinders)
        import json
        with open(sample['annotation'], 'r') as f:
            ann = json.load(f)
        
        K = np.array(ann['camera']['intrinsics']['K'], dtype=np.float32)
        
        # Extract cylinders from masks
        cylinders = []
        for inst_id, mask in binary_masks.items():
            cyl = mask_to_cylinder(mask, depth, K)
            if cyl is not None:
                cylinders.append(cyl)
        
        # Convert to tensor format
        if self.transform:
            rgb, depth = self.transform(rgb, depth)
        else:
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            depth = torch.from_numpy(depth).unsqueeze(0).float()
        
        return {
            'rgb': rgb,
            'depth': depth,
            'cylinders': cylinders,
            'K': K,
            'frame_id': sample['frame_id']
        }


# ============================================================================
# 6. INFERENCE / USAGE EXAMPLE
# ============================================================================

def inference_example():
    """Example usage of the model"""
    
    # Initialize model
    model = CylinderDETR(
        num_queries=50,
        hidden_dim=256,
        backbone='resnet50'
    )
    model.eval()
    
    # Dummy input
    rgb = torch.randn(1, 3, 480, 640)
    depth = torch.randn(1, 1, 480, 640)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(rgb, depth)
    
    # Parse outputs
    cylinder_params = outputs['cylinder_params'][0]  # (num_queries, 7)
    confidence = outputs['confidence'][0]  # (num_queries,)
    
    # Filter by confidence threshold
    threshold = 0.5
    valid_mask = confidence > threshold
    detected_cylinders = cylinder_params[valid_mask]
    
    print(f"Detected {detected_cylinders.shape[0]} cylinders")
    for i, cyl in enumerate(detected_cylinders):
        x1, y1, z1, x2, y2, z2, radius = cyl
        print(f"Cylinder {i}: start=({x1:.2f}, {y1:.2f}, {z1:.2f}), "
              f"end=({x2:.2f}, {y2:.2f}, {z2:.2f}), radius={radius:.3f}")

def process_and_save_masks_example():
    """Example: Process masks from dataset and save to mask_id folder"""
    import glob
    import json
    import os
    
    BASE_DIR = r"C:\Users\joses\Desktop\tree_dataset"
    MASK_DIR = os.path.join(BASE_DIR, "masks")  # Input masks
    MASK_ID_DIR = os.path.join(BASE_DIR, "mask_id")  # Output processed masks
    ANN_DIR = os.path.join(BASE_DIR, "ann")
    
    processor = MaskProcessor(id_channel='blue')
    
    # Find all mask files
    mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "mask_*.png")))
    
    for mask_path in mask_files:
        # Extract frame ID
        frame_id = os.path.basename(mask_path).split('_')[-1].split('.')[0]
        
        print(f"Processing frame {frame_id}...")
        
        # Load mask
        mask_img = cv2.imread(mask_path)
        if mask_img is None:
            print(f"  Failed to load {mask_path}")
            continue
        
        # Convert to binary masks
        binary_masks = processor.mask_id_to_binary_masks(mask_img)
        
        # Save processed masks
        processor.save_binary_masks(binary_masks, MASK_ID_DIR, frame_id)
        
        print(f"  Found {len(binary_masks)} instances")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'process_masks':
        print("Processing and saving masks...")
        process_and_save_masks_example()
    else:
        # Test mask processing
        print("Testing mask processing...")
        processor = MaskProcessor(id_channel='blue')
        
        # Test cylinder fitting
        print("\nTesting cylinder fitting...")
        # ... rest of existing code ...
    # Generate random points along a cylinder
    t = np.linspace(0, 1, 100)
    axis = np.array([0, 0, 1])
    points = np.outer(t, axis) + np.random.randn(100, 3) * 0.05
    cylinder = fit_cylinder_pca(points)
    if cylinder:
        print(f"Fitted cylinder: length={cylinder['length']:.2f}, radius={cylinder['radius']:.3f}")
    
    # Test model
    print("\nTesting model...")
    inference_example()