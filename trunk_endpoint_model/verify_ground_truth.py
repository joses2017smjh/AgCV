"""
Script to verify that create_ground_truth_maps is creating valid segmentation masks.
This will help diagnose why losses might be zero.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import torch
from image_data_loader import ImageTreeDataset, create_ground_truth_maps

def verify_sample(dataset_dir, metadata_dir, sample_idx=0, split='train'):
    """Verify a single sample from the dataset."""
    print(f"Loading dataset from {dataset_dir}...")
    dataset = ImageTreeDataset(
        dataset_dir=dataset_dir,
        metadata_dir=metadata_dir,
        split=split,
        image_size=(480, 640)
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("ERROR: No samples in dataset!")
        return
    
    if sample_idx >= len(dataset):
        sample_idx = 0
        print(f"Sample index out of range, using first sample")
    
    print(f"\nVerifying sample {sample_idx}...")
    sample = dataset[sample_idx]
    
    # Get the sample info
    rgb = sample['rgb']
    seg_mask = sample['segmentation']
    depth_gt = sample['depth_gt']
    radius_map = sample['radius']
    length_map = sample['length']
    
    print(f"\n=== Sample Statistics ===")
    print(f"RGB shape: {rgb.shape}")
    print(f"Segmentation shape: {seg_mask.shape}")
    print(f"Depth GT shape: {depth_gt.shape}")
    print(f"Radius map shape: {radius_map.shape}")
    print(f"Length map shape: {length_map.shape}")
    
    # Convert to numpy for analysis
    seg_mask_np = seg_mask.numpy() if isinstance(seg_mask, torch.Tensor) else seg_mask
    depth_gt_np = depth_gt.squeeze().numpy() if isinstance(depth_gt, torch.Tensor) else depth_gt.squeeze()
    
    # Statistics
    total_pixels = seg_mask_np.size
    tree_pixels = (seg_mask_np > 0).sum()
    background_pixels = (seg_mask_np == 0).sum()
    tree_percentage = (tree_pixels / total_pixels) * 100
    
    print(f"\n=== Segmentation Mask Statistics ===")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Tree pixels (class 1): {tree_pixels:,} ({tree_percentage:.2f}%)")
    print(f"Background pixels (class 0): {background_pixels:,} ({100-tree_percentage:.2f}%)")
    print(f"Unique values in seg_mask: {np.unique(seg_mask_np)}")
    
    if tree_pixels == 0:
        print("\n⚠️  WARNING: No tree pixels found in segmentation mask!")
        print("This is why losses are zero - there's no ground truth to learn from.")
        print("\nPossible causes:")
        print("1. No trunk cylinders in metadata file")
        print("2. Cylinders don't project into image bounds")
        print("3. Camera parameters are incorrect")
        print("4. Tree ID matching is wrong (using wrong metadata file)")
    else:
        print(f"\n✓ Found {tree_pixels:,} tree pixels - ground truth looks valid!")
    
    # Depth statistics
    if tree_pixels > 0:
        tree_depths = depth_gt_np[seg_mask_np > 0]
        print(f"\n=== Depth Statistics (tree pixels only) ===")
        print(f"Min depth: {tree_depths.min():.4f}")
        print(f"Max depth: {tree_depths.max():.4f}")
        print(f"Mean depth: {tree_depths.mean():.4f}")
        print(f"Std depth: {tree_depths.std():.4f}")
    
    # Radius statistics
    if tree_pixels > 0:
        radius_map_np = radius_map.squeeze().numpy() if isinstance(radius_map, torch.Tensor) else radius_map.squeeze()
        tree_radii = radius_map_np[seg_mask_np > 0]
        print(f"\n=== Radius Statistics (tree pixels only) ===")
        print(f"Min radius: {tree_radii.min():.6f}")
        print(f"Max radius: {tree_radii.max():.6f}")
        print(f"Mean radius: {tree_radii.mean():.6f}")
    
    # Visualize
    print(f"\nCreating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # RGB image
    rgb_np = rgb.permute(1, 2, 0).numpy() if isinstance(rgb, torch.Tensor) else rgb
    rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    # Segmentation mask
    axes[0, 1].imshow(seg_mask_np, cmap='gray')
    axes[0, 1].set_title(f'Segmentation Mask\n({tree_pixels:,} tree pixels, {tree_percentage:.2f}%)')
    axes[0, 1].axis('off')
    
    # Overlay
    overlay = rgb_np.copy()
    overlay[seg_mask_np > 0] = [1.0, 0.0, 0.0]  # Red for tree pixels
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('RGB + Segmentation Overlay')
    axes[0, 2].axis('off')
    
    # Depth map
    depth_vis = depth_gt_np.copy()
    depth_vis[depth_vis == 0] = np.nan
    im = axes[1, 0].imshow(depth_vis, cmap='viridis')
    axes[1, 0].set_title('Depth Map (tree pixels only)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Radius map
    radius_vis = radius_map.squeeze().numpy() if isinstance(radius_map, torch.Tensor) else radius_map.squeeze()
    radius_vis[radius_vis == 0] = np.nan
    im = axes[1, 1].imshow(radius_vis, cmap='hot')
    axes[1, 1].set_title('Radius Map (tree pixels only)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Length map
    length_vis = length_map.squeeze().numpy() if isinstance(length_map, torch.Tensor) else length_map.squeeze()
    length_vis[length_vis == 0] = np.nan
    im = axes[1, 2].imshow(length_vis, cmap='plasma')
    axes[1, 2].set_title('Length Map (tree pixels only)')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    output_path = Path('ground_truth_verification.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()
    
    # Check metadata file
    print(f"\n=== Checking Metadata File ===")
    sample_info = dataset.samples[sample_idx]
    metadata_file = sample_info['metadata_file']
    print(f"Metadata file: {metadata_file}")
    
    if Path(metadata_file).exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        cylinder_data = metadata.get('cylinder_data', {})
        trunk_cylinders = []
        
        for color_key, cylinder_info in cylinder_data.items():
            if isinstance(cylinder_info, dict):
                part_name = cylinder_info.get('part_name', '')
                if 'trunk' in part_name.lower():
                    trunk_cylinders.append(cylinder_info)
        
        print(f"Total cylinders in metadata: {len(cylinder_data)}")
        print(f"Trunk cylinders found: {len(trunk_cylinders)}")
        
        if len(trunk_cylinders) == 0:
            print("⚠️  WARNING: No trunk cylinders in metadata file!")
            print("This explains why the segmentation mask is empty.")
    else:
        print(f"⚠️  WARNING: Metadata file not found: {metadata_file}")
    
    return tree_pixels > 0


def verify_multiple_samples(dataset_dir, metadata_dir, n_samples=10, split='train'):
    """Verify multiple samples to get statistics."""
    print(f"Loading dataset from {dataset_dir}...")
    dataset = ImageTreeDataset(
        dataset_dir=dataset_dir,
        metadata_dir=metadata_dir,
        split=split,
        image_size=(480, 640)
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    n_samples = min(n_samples, len(dataset))
    tree_pixel_counts = []
    
    for i in range(n_samples):
        sample = dataset[i]
        seg_mask = sample['segmentation']
        seg_mask_np = seg_mask.numpy() if isinstance(seg_mask, torch.Tensor) else seg_mask
        tree_pixels = (seg_mask_np > 0).sum()
        tree_pixel_counts.append(tree_pixels)
    
    print(f"\n=== Statistics across {n_samples} samples ===")
    print(f"Mean tree pixels per sample: {np.mean(tree_pixel_counts):.0f}")
    print(f"Min tree pixels: {np.min(tree_pixel_counts)}")
    print(f"Max tree pixels: {np.max(tree_pixel_counts)}")
    print(f"Samples with tree pixels: {(np.array(tree_pixel_counts) > 0).sum()}/{n_samples}")
    print(f"Samples without tree pixels: {(np.array(tree_pixel_counts) == 0).sum()}/{n_samples}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify ground truth maps')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Directory with rgb/, depth/, ann/ from generate_dataset.py')
    parser.add_argument('--metadata_dir', type=str, required=True,
                       help='Directory with tree metadata JSONs')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to verify (default: 0)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--verify_multiple', action='store_true',
                       help='Verify multiple samples instead of visualizing one')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of samples to verify (if --verify_multiple)')
    
    args = parser.parse_args()
    
    if args.verify_multiple:
        verify_multiple_samples(
            args.dataset_dir,
            args.metadata_dir,
            n_samples=args.n_samples,
            split=args.split
        )
    else:
        is_valid = verify_sample(
            args.dataset_dir,
            args.metadata_dir,
            sample_idx=args.sample_idx,
            split=args.split
        )
        
        if is_valid:
            print("\n✓ Ground truth looks valid!")
            sys.exit(0)
        else:
            print("\n✗ Ground truth is invalid - no tree pixels found!")
            sys.exit(1)

