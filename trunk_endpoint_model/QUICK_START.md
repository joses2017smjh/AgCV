# Quick Start Guide

## 1. Verify Ground Truth Maps

Before training, verify that your ground truth maps are being created correctly:

### Verify a single sample (with visualization):
```bash
python3 trunk_endpoint_model/verify_ground_truth.py \
    --dataset_dir /home/joses/Ag_Cv/tree_dataset \
    --metadata_dir /home/joses/Ag_Cv/trees/metadata \
    --sample_idx 0 \
    --split train
```

This will:
- Load a sample from your dataset
- Show statistics about tree pixels in the segmentation mask
- Create a visualization saved as `ground_truth_verification.png`
- Check if the metadata file has trunk cylinders

### Verify multiple samples (statistics only):
```bash
python3 trunk_endpoint_model/verify_ground_truth.py \
    --dataset_dir /home/joses/Ag_Cv/tree_dataset \
    --metadata_dir /home/joses/Ag_Cv/trees/metadata \
    --verify_multiple \
    --n_samples 20 \
    --split train
```

This will show statistics across multiple samples to see if the problem is widespread.

### What to look for:
- **Tree pixels > 0**: Good! Ground truth is valid
- **Tree pixels = 0**: Problem! This is why losses are zero. Check:
  1. Does the metadata file have trunk cylinders?
  2. Are the camera parameters correct?
  3. Do the cylinders project into the image bounds?

## 2. Run Training

Once you've verified the ground truth looks good, run training:

### Basic training (with wandb):
```bash
python3 trunk_endpoint_model/train_image.py \
    --dataset_dir /home/joses/Ag_Cv/tree_dataset \
    --metadata_dir /home/joses/Ag_Cv/trees/metadata \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 100 \
    --lr 0.0001 \
    --use_camera \
    --wandb_project tree-image-multitask \
    --wandb_name resnet-unet-experiment \
    --output_dir ./checkpoints_image
```

### Training with mixed precision (saves memory):
```bash
python3 trunk_endpoint_model/train_image.py \
    --dataset_dir /home/joses/Ag_Cv/tree_dataset \
    --metadata_dir /home/joses/Ag_Cv/trees/metadata \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 100 \
    --lr 0.0001 \
    --mixed_precision \
    --use_camera \
    --wandb_project tree-image-multitask \
    --wandb_name resnet-unet-experiment-fp16 \
    --output_dir ./checkpoints_image
```

### Training without wandb:
```bash
python3 trunk_endpoint_model/train_image.py \
    --dataset_dir /home/joses/Ag_Cv/tree_dataset \
    --metadata_dir /home/joses/Ag_Cv/trees/metadata \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 100 \
    --no_wandb \
    --output_dir ./checkpoints_image
```

### If you get CUDA OOM errors:
1. Reduce batch size: `--batch_size 1`
2. Increase gradient accumulation: `--gradient_accumulation_steps 8`
3. Use mixed precision: `--mixed_precision`
4. Reduce image size: `--image_size 320 480`

## 3. Monitor Training

### Check the debug output:
On the first batch of the first epoch, you'll see:
```
Debug - First batch losses:
  Total loss: X.XXXXXX
  Seg loss: X.XXXXXX
  Depth loss: X.XXXXXX
  ...
  Seg target unique values: [0, 1]  # Should have both 0 and 1
  Seg target sum (tree pixels): XXXX  # Should be > 0
```

If `Seg target sum (tree pixels)` is 0, your ground truth is empty!

### Check wandb:
- Open your wandb project: https://wandb.ai/your-username/tree-image-multitask
- Look for:
  - `train/epoch/loss` - should decrease over time
  - `train/epoch/cross_entropy_loss` - should decrease
  - `train/epoch/pixel_accuracy` - should increase
  - `train/epoch/learning_rate` - should decrease when scheduler triggers

## 4. Common Issues

### Issue: All losses are 0.0000
**Cause**: No tree pixels in ground truth
**Solution**: 
1. Run verification script to check
2. Fix `create_ground_truth_maps` if needed
3. Check metadata file has trunk cylinders

### Issue: CUDA out of memory
**Solution**: Use smaller batch size, gradient accumulation, or mixed precision (see above)

### Issue: Validation metrics show 0.0000
**Cause**: Same as above - no tree pixels, or validation set is empty
**Solution**: Check validation set has samples with tree pixels

## 5. Key Arguments

- `--dataset_dir`: Path to directory with `rgb/`, `depth/`, `ann/` subdirectories
- `--metadata_dir`: Path to directory with `*_metadata.json` files
- `--batch_size`: Number of samples per batch (default: 2)
- `--gradient_accumulation_steps`: Accumulate gradients over N batches (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.0001)
- `--mixed_precision`: Use FP16 to save memory
- `--use_camera`: Use camera parameters in model (default: True)
- `--wandb_project`: Weights & Biases project name
- `--wandb_name`: Run name in wandb
- `--no_wandb`: Disable wandb logging
- `--output_dir`: Directory to save checkpoints

