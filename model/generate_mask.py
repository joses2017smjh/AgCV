"""
Generate instance segmentation masks from RGB images
Uses Segment Anything Model (SAM) or Mask R-CNN
"""

import cv2
import numpy as np
import torch
import os
import glob
from pathlib import Path

def generate_masks_with_sam2(rgb_dir, output_dir, model_size='large'):
    """
    Generate masks using SAM 2 (Segment Anything Model 2)
    Uses local sam2 folder and checkpoints
    
    Args:
        model_size: 'tiny', 'small', 'base_plus', or 'large' (recommended)
    """
    import torch
    import sys
    
    # Add local sam2 folder to Python path
    script_dir = os.path.dirname(__file__)
# Point to the actual sam2 package inside the repository
    sam2_local_dir = os.path.join(script_dir, 'sam2', 'sam2')    
    if sam2_local_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as e:
        print(f"ERROR: Could not import SAM 2: {e}")
        print(f"Looking for sam2 folder at: {sam2_local_dir}")
        print(f"Folder exists: {os.path.exists(sam2_local_dir)}")
        return
    
    # Use local directories
    checkpoint_dir = os.path.join(script_dir, 'checkpoints')
# Config YAMLs are in configs/sam2/ subdirectory
    configs_dir = os.path.join(sam2_local_dir, 'configs', 'sam2')    # SAM 2 model configurations
    sam2_config_names = {
        'tiny': 'sam2_hiera_t.yaml',
        'small': 'sam2_hiera_s.yaml',
        'base_plus': 'sam2_hiera_b+.yaml',
        'large': 'sam2_hiera_l.yaml'
    }
    
    sam2_checkpoints = {
        'tiny': 'sam2_hiera_tiny.pt',
        'small': 'sam2_hiera_small.pt',
        'base_plus': 'sam2_hiera_base_plus.pt',
        'large': 'sam2_hiera_large.pt'
    }
    
    config_name = sam2_config_names.get(model_size, 'sam2_hiera_l.yaml')
    checkpoint_name = sam2_checkpoints.get(model_size, 'sam2_hiera_large.pt')
    
    # Full paths
    config_file = os.path.join(configs_dir, config_name)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"SAM 2 Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model size: {model_size}")
    print(f"SAM2 folder: {sam2_local_dir}")
    print(f"Config file: {config_file}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Verify files exist
    if not os.path.exists(sam2_local_dir):
        print(f"❌ ERROR: sam2 folder not found at {sam2_local_dir}")
        print("Please ensure you copied the entire sam2 folder to the model directory")
        return
    
    if not os.path.exists(config_file):
        print(f"❌ ERROR: Config file not found at {config_file}")
        print(f"\nContents of {sam2_local_dir}:")
        if os.path.exists(sam2_local_dir):
            print(os.listdir(sam2_local_dir))
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print(f"\nContents of {checkpoint_dir}:")
        if os.path.exists(checkpoint_dir):
            print(os.listdir(checkpoint_dir))
        return
    
    print("✓ All files found!\n")
    
    try:
        # Build SAM 2 model
        print("Loading SAM 2 model...")
        sam2_model = build_sam2(config_file, checkpoint_path, device=device)
        
        # Create mask generator
        print("Creating mask generator...")
        mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        print("✓ Model loaded successfully!\n")
        
    except Exception as e:
        print(f"❌ ERROR loading SAM 2 model: {e}")
        print("\nFull error traceback:")
        import traceback
        traceback.print_exc()
        return
    
    # Process all RGB images
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "rgb_*.png")))
    
    if len(rgb_files) == 0:
        print(f"❌ No RGB images found in {rgb_dir}")
        print("Looking for files matching pattern: rgb_*.png")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Processing {len(rgb_files)} images")
    print(f"{'='*60}\n")
    
    for i, rgb_path in enumerate(rgb_files):
        frame_id = os.path.basename(rgb_path).split('_')[-1].split('.')[0]
        print(f"[{i+1}/{len(rgb_files)}] Frame {frame_id}...", end=' ')
        
        # Load RGB
        image = cv2.imread(rgb_path)
        if image is None:
            print(f"❌ ERROR loading image")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        try:
            masks = mask_generator.generate(image_rgb)
            print(f"✓ {len(masks)} masks", end=' ')
        except Exception as e:
            print(f"❌ ERROR: {e}")
            continue
        
        # Create instance mask
        h, w = image_rgb.shape[:2]
        instance_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Sort by area
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        valid_count = 0
        for inst_id, mask_data in enumerate(masks, start=1):
            if inst_id > 254:
                break
            
            binary_mask = mask_data['segmentation']
            if np.sum(binary_mask) < 50:
                continue
            
            valid_count += 1
            instance_mask[binary_mask, 2] = valid_count
        
        # Save mask
        output_path = os.path.join(output_dir, f"mask_{frame_id}.png")
        cv2.imwrite(output_path, instance_mask)
        
        # Save visualization
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        vis_mask = np.zeros_like(image_rgb)
        np.random.seed(42 + i)
        for inst_id in range(1, valid_count + 1):
            color = np.random.randint(50, 255, 3)
            vis_mask[instance_mask[:, :, 2] == inst_id] = color
        
        blended = cv2.addWeighted(image_rgb, 0.6, vis_mask, 0.4, 0)
        vis_path = os.path.join(vis_dir, f"vis_{frame_id}.png")
        cv2.imwrite(vis_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        print(f"→ {valid_count} instances saved")
    
    print(f"\n{'='*60}")
    print(f"✓ Complete!")
    print(f"Masks: {output_dir}")
    print(f"Visualizations: {os.path.join(output_dir, 'visualizations')}")
    print(f"{'='*60}\n")


def generate_masks_simple_color_segmentation(rgb_dir, output_dir):
    """
    Simple mask generation using color-based segmentation
    Good for trees: segments green/brown regions
    """
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "rgb_*.png")))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(rgb_files)} RGB images to process")
    
    for i, rgb_path in enumerate(rgb_files):
        frame_id = os.path.basename(rgb_path).split('_')[-1].split('.')[0]
        print(f"\nProcessing frame {frame_id} ({i+1}/{len(rgb_files)})...")
        
        # Load RGB
        image = cv2.imread(rgb_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Segment green regions (leaves)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Segment brown regions (branches)
        lower_brown = np.array([10, 30, 30])
        upper_brown = np.array([25, 200, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Combine masks
        tree_mask = cv2.bitwise_or(green_mask, brown_mask)
        
        # Find connected components (individual tree parts)
        num_labels, labels = cv2.connectedComponents(tree_mask)
        
        # Create instance mask
        h, w = image.shape[:2]
        instance_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Encode each component as a separate instance
        for inst_id in range(1, num_labels):
            component_mask = (labels == inst_id)
            # Only keep components larger than min size
            if np.sum(component_mask) > 100:  # min 100 pixels
                instance_mask[component_mask, 2] = inst_id % 255
        
        # Save mask
        output_path = os.path.join(output_dir, f"mask_{frame_id}.png")
        cv2.imwrite(output_path, instance_mask)
        print(f"  Saved mask with {num_labels-1} instances to {output_path}")


def generate_masks_manual_annotation(rgb_dir, output_dir):
    """
    Interactive manual annotation tool
    Click to select regions and assign instance IDs
    """
    print("Manual annotation not yet implemented")
    print("Use tools like LabelMe, CVAT, or Labelbox for manual annotation")


if __name__ == "__main__":
    import sys
    
    BASE_DIR = r"C:\Users\joses\Desktop\tree_dataset"
    RGB_DIR = os.path.join(BASE_DIR, "rgb")
    MASK_DIR = os.path.join(BASE_DIR, "masks")
    
    print("=" * 60)
    print("MASK GENERATION TOOL")
    print("=" * 60)
    print("\nChoose method:")
    print("1. SAM 2 Large (Recommended) - Best quality")
    print("2. SAM 2 Base Plus - Good balance")
    print("3. SAM 2 Small - Faster, less accurate")
    print("4. Color Segmentation - Simple, works for trees")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n--- Using SAM 2 Large ---")
        generate_masks_with_sam2(RGB_DIR, MASK_DIR, model_size='large')
    elif choice == "2":
        print("\n--- Using SAM 2 Base Plus ---")
        generate_masks_with_sam2(RGB_DIR, MASK_DIR, model_size='base_plus')
    elif choice == "3":
        print("\n--- Using SAM 2 Small ---")
        generate_masks_with_sam2(RGB_DIR, MASK_DIR, model_size='small')
    elif choice == "4":
        print("\n--- Using Color-Based Segmentation ---")
        generate_masks_simple_color_segmentation(RGB_DIR, MASK_DIR)
    else:
        print("Invalid choice")