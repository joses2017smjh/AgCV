"""
Setup script for Depth Anything V2
This will download the model weights and install dependencies.
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install required packages."""
    packages = [
        'torch',
        'torchvision',
        'opencv-python',
        'pillow',
        'numpy',
        'tqdm',
        'huggingface-hub'
    ]
    
    print("Installing dependencies...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\nAll dependencies installed!")


def download_model():
    """Download Depth Anything V2 model from Hugging Face."""
    from huggingface_hub import hf_hub_download
    
    # Download the Depth Anything V2 model (small version for faster processing)
    # You can change to 'depth-anything/Depth-Anything-V2-Base' or 'Large' for better quality
    model_name = "depth-anything/Depth-Anything-V2-Large"
    
    print(f"\nDownloading {model_name} from Hugging Face...")
    
    # Create weights directory
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    try:
        # Download model weights
        model_path = hf_hub_download(
            repo_id=model_name,
            filename="pytorch_model.bin",
            cache_dir=weights_dir
        )
        print(f"Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nAlternative: You can manually download from:")
        print(f"https://huggingface.co/{model_name}")
        return None


if __name__ == "__main__":
    print("="*60)
    print("Depth Anything V2 Setup")
    print("="*60)
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Download model
    download_model()
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("Run 'python process_rgb_images.py' to process your images")
    print("="*60)

