# Integration Summary: Camera Setup + Image-Based Multi-Task Model

## What Was Integrated

### 1. Camera Setup from `generate_dataset.py`
- **Camera intrinsics computation**: Extracts K matrix (focal length, principal point)
- **Camera pose**: Location and rotation (Euler angles)
- **Projection utilities**: Projects 3D cylinder data to 2D image coordinates

### 2. Image-Based Data Loader (`image_data_loader.py`)
- Loads RGB images from `tree_dataset/rgb/`
- Loads camera annotations from `tree_dataset/ann/`
- Projects trunk cylinders to image space to create ground truth maps:
  - Segmentation mask (tree pixels)
  - Depth map
  - Radius map
  - Length map
- Integrates camera parameters for model input

### 3. Multi-Task CNN Model (`image_model.py`)
- **ResNet-50 encoder**: Pretrained backbone for feature extraction
- **U-Net decoder**: Upsamples features to full resolution
- **Camera fusion**: Camera parameters (location, rotation, intrinsics) are learned and fused
- **Multi-task heads**: Separate heads for segmentation, depth, radius, length

### 4. Multi-Task Loss (`image_losses.py`)
- **Segmentation loss**: Cross-entropy + Dice loss (handles class imbalance)
- **Depth/Radius/Length loss**: Masked Huber loss (only on tree pixels)
- All losses are weighted and combined

### 5. Training Script (`train_image.py`)
- Complete training loop with:
  - **Batch-level logging**: Metrics logged every N batches
  - **Epoch-level logging**: Average metrics per epoch
  - **Train and Val metrics**: Separate tracking for both splits
  - **Accuracy metrics**: Segmentation accuracy, IoU, MAE for depth/radius/length
  - **Loss metrics**: All loss components tracked separately

## Metrics Logged to Wandb

### Batch Level (logged every 10 train batches, every 5 val batches)
- `train/batch/loss` - Total loss
- `train/batch/segmentation_loss` - Segmentation loss
- `train/batch/depth_loss` - Depth loss
- `train/batch/radius_loss` - Radius loss
- `train/batch/length_loss` - Length loss
- `train/batch/segmentation_accuracy` - Pixel accuracy
- `train/batch/segmentation_iou` - Intersection over Union
- `train/batch/depth_mae` - Depth mean absolute error
- `train/batch/radius_mae` - Radius mean absolute error
- `train/batch/length_mae` - Length mean absolute error
- Same for `val/batch/*`

### Epoch Level (logged every epoch)
- `train/epoch/loss` - Average total loss
- `train/epoch/segmentation_loss` - Average segmentation loss
- `train/epoch/depth_loss` - Average depth loss
- `train/epoch/radius_loss` - Average radius loss
- `train/epoch/length_loss` - Average length loss
- `train/epoch/segmentation_accuracy` - Average pixel accuracy
- `train/epoch/segmentation_iou` - Average IoU
- `train/epoch/depth_mae` - Average depth MAE
- `train/epoch/radius_mae` - Average radius MAE
- `train/epoch/length_mae` - Average length MAE
- Same for `val/epoch/*`

## Data Flow

```
generate_dataset.py
    ↓
Creates: tree_dataset/
    ├── rgb/          (RGB images)
    ├── depth/        (Depth maps)
    └── ann/          (Camera annotations)
    
image_data_loader.py
    ↓
Loads: RGB + Camera + Metadata
    ↓
Projects: 3D cylinders → 2D image maps
    ↓
Outputs: (RGB, Camera, Segmentation, Depth, Radius, Length)

train_image.py
    ↓
Model: ResNet + U-Net + Camera Fusion
    ↓
Predicts: Segmentation, Depth, Radius, Length
    ↓
Loss: Multi-task loss (CE+Dice + Masked Huber)
    ↓
Metrics: Accuracy + Loss (batch + epoch level)
    ↓
Wandb: All metrics logged
```

## Key Features

1. **Camera Integration**: Camera parameters from scanning are used in the model
2. **Multi-Task Learning**: Single model predicts 4 outputs simultaneously
3. **Masked Losses**: Depth/radius/length losses only apply to tree pixels
4. **Comprehensive Metrics**: Both accuracy and loss tracked at batch and epoch level
5. **Wandb Integration**: Full logging for experiment tracking

## Usage

```bash
# Train the image-based model
python3 trunk_endpoint_model/train_image.py \
    --dataset_dir /home/joses/Ag_Cv/tree_dataset \
    --metadata_dir /home/joses/Ag_Cv/trees/metadata \
    --batch_size 8 \
    --epochs 100 \
    --use_camera \
    --wandb_project tree-image-multitask
```

## Differences from Point Cloud Model

| Feature | Point Cloud Model | Image Model |
|---------|------------------|-------------|
| Input | 3D point clouds (.ply) | RGB images |
| Output | 2 trunk endpoints | Segmentation + Depth + Radius + Length |
| Architecture | PointNet | ResNet + U-Net |
| Camera | Not used | Fully integrated |
| Ground Truth | From JSON cylinders | Projected from cylinders to images |
| Use Case | 3D geometry analysis | 2D image analysis |

Both models can be used together or independently depending on your needs!


