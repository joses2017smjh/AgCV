import os
import numpy as np
import matplotlib.pyplot as plt

# Enable EXR support in OpenCV BEFORE importing cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

BASE_DIR = r"C:\Users\joses\Desktop\tree_dataset"
DEPTH_DIR = os.path.join(BASE_DIR, "depth")
RGB_DIR = os.path.join(BASE_DIR, "rgb")

# Change this to view different frames
FRAME_TO_VIEW = 30


def load_depth_exr(path):
    """Load EXR depth as float32 (single channel) using OpenCV."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth file: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    return depth


def visualize_depth_and_rgb(frame_num):
    """Visualize depth and RGB images side by side."""
    depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_num:04d}.exr")
    rgb_path = os.path.join(RGB_DIR, f"rgb_{frame_num:04d}.png")
    
    # Load depth
    depth = load_depth_exr(depth_path)

    # --- NEW: choose threshold range ---
    MIN_DEPTH = 0.2   # ignore 0 and very tiny depths (tune this)
    MAX_DEPTH = 2  # ignore super far / bad values (tune this)
    
    # Build mask of valid depths
    valid_mask = np.isfinite(depth) & (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    
    # Load RGB
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Print statistics
    print(f"Frame {frame_num}")
    print(f"Depth shape: {depth.shape}")
    print(f"RGB shape: {rgb.shape}")
    print(f"Depth min (finite): {np.nanmin(depth[valid_mask]):.3f}, "
          f"max (finite): {np.nanmax(depth[valid_mask]):.3f}")
    non_zero = valid_mask.sum()
    print(f"Valid depth pixels: {non_zero} / {depth.size} "
          f"({100 * non_zero / depth.size:.1f}%)")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show RGB
    axes[0].imshow(rgb)
    axes[0].set_title(f'RGB - Frame {frame_num}')
    axes[0].axis('off')
    
    # Show depth (clipped values)
    depth_vis = depth.copy().astype(np.float32)
    depth_vis[~valid_mask] = np.nan  # everything invalid becomes transparent
    im1 = axes[1].imshow(depth_vis, cmap='viridis')
    axes[1].set_title(f'Depth {MIN_DEPTH}–{MAX_DEPTH} m')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Show depth histogram for valid range only
    valid_depths = depth[valid_mask]
    if len(valid_depths) > 0:
        axes[2].hist(valid_depths, bins=100, edgecolor='black')
        axes[2].set_title('Depth Distribution (clipped)')
        axes[2].set_xlabel('Depth (meters)')
        axes[2].set_ylabel('Frequency')
        axes[2].axvline(valid_depths.mean(), linestyle='--',
                        label=f'Mean: {valid_depths.mean():.2f}m')
        axes[2].axvline(np.median(valid_depths), linestyle='--',
                        label=f'Median: {np.median(valid_depths):.2f}m')
        axes[2].legend()
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Single frame view
    visualize_depth_and_rgb(FRAME_TO_VIEW)
    
    # Optionally show multiple frames
    print("\nShowing multiple frames (1, 30, 60, 90)...")
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    MIN_DEPTH = 0.2
    MAX_DEPTH = 2

    for idx, frame_num in enumerate([1, 30, 60, 90]):
        depth_path = os.path.join(DEPTH_DIR, f"depth_{frame_num:04d}.exr")
        rgb_path = os.path.join(RGB_DIR, f"rgb_{frame_num:04d}.png")
        
        if not os.path.exists(depth_path):
            continue
        
        depth = load_depth_exr(depth_path)
        valid_mask = np.isfinite(depth) & (depth > MIN_DEPTH) & (depth < MAX_DEPTH)

        
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f'RGB - Frame {frame_num}')
        axes[idx, 0].axis('off')
        
        depth_vis = depth.copy().astype(np.float32)
        depth_vis[~valid_mask] = np.nan
        axes[idx, 1].imshow(depth_vis, cmap='viridis')
        axes[idx, 1].set_title(f'Depth - Frame {frame_num}')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
