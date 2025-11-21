"""
Quick test script to process a single image with Depth Anything V2
Use this to test the setup before processing all images
"""

import os
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt

# Test with frame 1
TEST_FRAME = 30
RGB_DIR = r"C:\Users\joses\Desktop\tree_dataset\rgb"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "depthanyv2")


def test_depth_estimation():
    """Test depth estimation on a single image."""
    
    print("="*60)
    print("Depth Anything V2 - Single Image Test")
    print("="*60)
    
    # Check input file
    rgb_path = os.path.join(RGB_DIR, f"rgb_{TEST_FRAME:04d}.png")
    if not os.path.exists(rgb_path):
        print(f"ERROR: Test image not found: {rgb_path}")
        return
    
    print(f"Test image: {rgb_path}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Import transformers
    try:
        from transformers import pipeline
    except ImportError:
        print("\nERROR: transformers not installed")
        print("Install with: pip install transformers")
        return
    
    # Load model
    print("\nLoading Depth Anything V2 Large model...")
    print("(First run will download ~1GB - this may take several minutes)")
    
    try:
        pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large-hf",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Load and process image
    print(f"\nProcessing frame {TEST_FRAME}...")
    image = Image.open(rgb_path).convert("RGB")
    
    result = pipe(image)
    depth_image = result["depth"]
    depth_array = np.array(depth_image)
    
    print(f"✓ Depth estimation complete!")
    print(f"  Depth shape: {depth_array.shape}")
    print(f"  Depth range: [{depth_array.min():.2f}, {depth_array.max():.2f}]")
    
    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save normalized depth image
    depth_normalized = ((depth_array - depth_array.min()) / 
                       (depth_array.max() - depth_array.min()) * 255).astype(np.uint8)
    
    output_path = os.path.join(OUTPUT_DIR, f"test_depth_{TEST_FRAME:04d}.png")
    cv2.imwrite(output_path, depth_normalized)
    print(f"✓ Saved to: {output_path}")
    
    # Visualize
    print("\nGenerating visualization...")
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(rgb)
    axes[0].set_title(f'RGB Image - Frame {TEST_FRAME}')
    axes[0].axis('off')
    
    im = axes[1].imshow(depth_normalized, cmap='inferno')
    axes[1].set_title(f'Depth Map (Depth Anything V2)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save comparison
    comparison_path = os.path.join(OUTPUT_DIR, f"test_comparison_{TEST_FRAME:04d}.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison saved to: {comparison_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("Test successful!")
    print("You can now run 'process_rgb_images.py' to process all images")
    print("="*60)


if __name__ == "__main__":
    test_depth_estimation()

