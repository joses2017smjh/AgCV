# Depth Anything V2 - RGB to Depth Processing

This directory contains scripts to process RGB images using Depth Anything V2 monocular depth estimation.

## Directory Structure

```
Proccs/
├── depthanyv2/           # Output depth maps will be saved here
├── process_rgb_images.py # Main processing script
├── setup_depthanything.py # Setup and installation script
├── run_depth_estimation.bat # Quick run script for Windows
└── README.md             # This file
```

## Quick Start

### Option 1: Use the Batch File (Easiest)
Just double-click `run_depth_estimation.bat` and it will:
1. Install all dependencies
2. Download the Depth Anything V2 model
3. Process all RGB images
4. Save depth maps to `depthanyv2/` folder

### Option 2: Manual Setup

1. **Install Dependencies:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install transformers pillow opencv-python tqdm accelerate
   ```

2. **Run Processing:**
   ```bash
   python process_rgb_images.py
   ```

## Configuration

Edit `process_rgb_images.py` to change settings:

- `RGB_DIR`: Input directory with RGB images
- `OUTPUT_DIR`: Where to save depth maps
- `MODEL_TYPE`: Choose model size
  - `'small'`: Faster, lower quality (default)
  - `'base'`: Balanced
  - `'large'`: Slower, highest quality

## Output

For each RGB image `rgb_XXXX.png`, the script generates:
- `depth_XXXX.png`: Visualized depth map (0-255 grayscale)
- `depth_XXXX.npy`: Raw depth values (numpy array)
- `comparison_frame_0001.png`: Sample comparison image

## Model Info

- **Model**: Depth Anything V2 by TikTok/ByteDance
- **Source**: https://huggingface.co/depth-anything
- **Paper**: https://arxiv.org/abs/2406.09414
- **Type**: Monocular depth estimation (single image → depth map)

## GPU Support

The script automatically uses GPU (CUDA) if available, otherwise CPU.
For faster processing, ensure you have a CUDA-compatible GPU.

## Troubleshooting

### Model download fails
- Check internet connection
- Models are cached in `~/.cache/huggingface/`
- Manual download: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

### Out of memory
- Use smaller model type (`'small'` instead of `'large'`)
- Process images one at a time
- Use CPU instead of GPU (edit script: `device=-1`)

### Slow processing
- First run downloads ~200MB model (one-time)
- CPU processing is slower (consider GPU)
- Use `'small'` model for faster results

## Notes

- First run will download the model (~200MB for small, ~1GB for large)
- Processing time: ~1-2 seconds per image (GPU) or 5-10 seconds (CPU)
- Depth maps are relative (not metric depth in meters)

