# Weights & Biases (wandb) Setup Guide

## Installation

Install wandb along with other dependencies:

```bash
pip install -r requirements.txt
```

Or install wandb separately:

```bash
pip install wandb
```

## Initial Setup

1. **Create a wandb account** (if you don't have one):
   - Visit https://wandb.ai
   - Sign up for a free account

2. **Login to wandb**:
   ```bash
   wandb login
   ```
   This will prompt you for your API key, which you can find at https://wandb.ai/authorize

## Usage

### Training with wandb

The training script automatically logs to wandb. Just run:

```bash
python3 trunk_endpoint_model/train.py \
    --data_dir /home/joses/Ag_Cv/trees \
    --num_points 2048 \
    --batch_size 32 \
    --epochs 100 \
    --output_dir ./checkpoints \
    --wandb_project trunk-endpoint-regression \
    --wandb_name my-experiment-name
```

**Tracked Metrics:**
- Training loss (total, endpoint, direction)
- Validation loss (total, endpoint, direction)
- Validation endpoint error (L2 distance)
- Validation axis error (angle in radians and degrees)
- Learning rate
- Model checkpoints (best model saved to wandb)

**Hyperparameters logged:**
- num_points
- include_normals
- model architecture
- batch_size
- epochs
- learning_rate
- beta (direction loss weight)
- train/val sample counts

### Disable wandb

If you want to train without wandb:

```bash
python3 trunk_endpoint_model/train.py \
    --data_dir /home/joses/Ag_Cv/trees \
    --no_wandb \
    ...
```

### Evaluation with wandb

Log evaluation results to wandb:

```bash
python3 trunk_endpoint_model/evaluate.py \
    --data_dir /home/joses/Ag_Cv/trees \
    --checkpoint ./checkpoints/best_model.pth \
    --split test \
    --wandb_project trunk-endpoint-regression \
    --wandb_name test-evaluation
```

## Viewing Results

1. **Web Interface**: Visit https://wandb.ai to see your runs
2. **Compare runs**: Use the wandb web UI to compare different experiments
3. **Download artifacts**: Model checkpoints are automatically saved to wandb

## Example Metrics Tracked

- **Loss Metrics**:
  - `train/loss`: Total training loss
  - `train/endpoint_loss`: Endpoint regression loss
  - `train/direction_loss`: Direction consistency loss
  - `val/loss`: Total validation loss
  - `val/endpoint_loss`: Validation endpoint loss
  - `val/direction_loss`: Validation direction loss

- **Accuracy Metrics**:
  - `val/endpoint_error`: Mean L2 distance between predicted and ground truth endpoints
  - `val/axis_error_rad`: Mean angle error in radians
  - `val/axis_error_deg`: Mean angle error in degrees

- **Training Info**:
  - `learning_rate`: Current learning rate
  - `epoch`: Current epoch number

## Tips

1. **Organize experiments**: Use descriptive `--wandb_name` to identify different experiments
2. **Compare hyperparameters**: wandb automatically tracks all hyperparameters for easy comparison
3. **Monitor training**: Watch training progress in real-time on the wandb dashboard
4. **Save best models**: Best model checkpoints are automatically uploaded to wandb


