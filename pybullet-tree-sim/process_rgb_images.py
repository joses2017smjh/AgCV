"""
Process all RGB images using Depth Anything V2
Generates depth maps and saves them to depthanyv2 folder
"""

import os
import glob
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Configuration
RGB_DIR = r"C:\Users\joses\Desktop\tree_dataset\rgb"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "depthanyv2")
MODEL_TYPE = "large"  # Options: 'small', 'base', 'large'


def download_and_load_model(model_type="small"):
    """
    Download and load Depth Anything V2 model.
    """
    print(f"Loading Depth Anything V2 ({model_type}) model...")
    
    try:
        # Try to import transformers (Hugging Face)
        from transformers import pipeline
        
        model_name_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        
        model_name = model_name_map.get(model_type, model_name_map['small'])
        
        print(f"Downloading model: {model_name}")
        print("This may take a few minutes on first run...")
        
        # Load the depth estimation pipeline
        pipe = pipeline(
            task="depth-estimation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        
        print(f"Model loaded successfully!")
        print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        return pipe
        
    except ImportError:
        print("ERROR: transformers package not found.")
        print("Please install it: pip install transformers")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def process_single_image(pipe, rgb_path, output_path):
    """
    Process a single RGB image to generate depth map.
    """
    # Load RGB image
    image = Image.open(rgb_path).convert("RGB")
    
    # Run depth estimation
    result = pipe(image)
    
    # Get depth map (PIL Image)
    depth_image = result["depth"]
    
    # Convert to numpy array
    depth_array = np.array(depth_image)
    
    # Normalize to 0-255 for visualization
    depth_normalized = ((depth_array - depth_array.min()) / 
                       (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
    
    # Save depth map as PNG
    cv2.imwrite(output_path, depth_normalized)
    
    # Also save raw depth as numpy array (.npy) for later use
    npy_path = output_path.replace('.png', '.npy')
    np.save(npy_path, depth_array)
    
    return depth_array


def process_all_images():
    """
    Process all RGB images in the dataset.
    """
    # Check if RGB directory exists
    if not os.path.exists(RGB_DIR):
        print(f"ERROR: RGB directory not found: {RGB_DIR}")
        return
    
    # Get all RGB images
    rgb_files = sorted(glob.glob(os.path.join(RGB_DIR, "rgb_*.png")))
    
    if len(rgb_files) == 0:
        print(f"ERROR: No RGB images found in {RGB_DIR}")
        return
    
    print(f"Found {len(rgb_files)} RGB images to process")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load model
    pipe = download_and_load_model(MODEL_TYPE)
    if pipe is None:
        print("Failed to load model. Exiting.")
        return
    
    # Process each image
    print("\nProcessing images...")
    for rgb_path in tqdm(rgb_files, desc="Processing"):
        # Get filename
        filename = os.path.basename(rgb_path)
        frame_num = filename.replace('rgb_', '').replace('.png', '')
        
        # Output paths
        output_path = os.path.join(OUTPUT_DIR, f"depth_{frame_num}.png")
        
        # Skip if already processed
        if os.path.exists(output_path):
            continue
        
        try:
            # Process image
            process_single_image(pipe, rgb_path, output_path)
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Depth maps saved to: {OUTPUT_DIR}")
    print(f"Total images processed: {len(rgb_files)}")
    print(f"{'='*60}")


def visualize_sample(frame_num=1):
    """
    Visualize a sample RGB and depth pair.
    """
    import matplotlib.pyplot as plt
    
    rgb_path = os.path.join(RGB_DIR, f"rgb_{frame_num:04d}.png")
    depth_path = os.path.join(OUTPUT_DIR, f"depth_{frame_num:04d}.png")
    
    if not os.path.exists(depth_path):
        print(f"Depth map not found: {depth_path}")
        return
    
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb)
    axes[0].set_title(f'RGB - Frame {frame_num}')
    axes[0].axis('off')
    
    axes[1].imshow(depth, cmap='inferno')
    axes[1].set_title(f'Depth (Depth Anything V2) - Frame {frame_num}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'comparison_frame_{frame_num:04d}.png'), dpi=150)
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("Depth Anything V2 - RGB Image Processor")
    print("="*60)
    print(f"Input:  {RGB_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model:  {MODEL_TYPE}")
    print("="*60)
    
    # Process all images
    process_all_images()
    
    # Optionally visualize a sample
    try:
        print("\nGenerating sample visualization...")
        visualize_sample(frame_num=1)
    except Exception as e:
        print(f"Could not generate visualization: {e}")

