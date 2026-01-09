"""
Training script for image-based multi-task tree analysis.
Uses ResNet + U-Net architecture with camera integration.
"""
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import wandb
import numpy as np

from image_data_loader import ImageTreeDataset
from image_model import ResNetUNetMultiTask
from image_losses import MultiTaskLoss


def compute_accuracy_metrics(predictions, targets):
    """
    Compute accuracy metrics for all tasks.
    
    Returns:
        dict with accuracy metrics
    """
    metrics = {}
    
    # Segmentation accuracy (pixel-level/token-level accuracy)
    seg_pred = predictions['segmentation'].argmax(dim=1)  # (B, H, W)
    seg_target = targets['segmentation']  # (B, H, W)
    seg_acc = (seg_pred == seg_target).float().mean().item()
    metrics['segmentation_accuracy'] = seg_acc
    metrics['pixel_accuracy'] = seg_acc  # Token-level accuracy (same as pixel-level)
    
    # IoU for segmentation
    intersection = ((seg_pred == 1) & (seg_target == 1)).float().sum()
    union = ((seg_pred == 1) | (seg_target == 1)).float().sum()
    iou = (intersection / (union + 1e-8)).item()
    metrics['segmentation_iou'] = iou
    
    # Per-class accuracy
    for class_id in [0, 1]:
        class_mask = (seg_target == class_id)
        if class_mask.sum() > 0:
            class_acc = ((seg_pred == class_id) & class_mask).float().sum() / class_mask.sum().item()
            metrics[f'class_{class_id}_accuracy'] = class_acc.item()
        else:
            metrics[f'class_{class_id}_accuracy'] = 0.0
    
    # Depth error (only on tree pixels)
    mask = (seg_target > 0).float().unsqueeze(1)  # (B, 1, H, W)
    depth_pred = predictions['depth']
    depth_target = targets['depth_gt']
    
    if mask.sum() > 0:
        depth_error = torch.abs(depth_pred - depth_target) * mask
        depth_mae = (depth_error.sum() / mask.sum()).item()
        metrics['depth_mae'] = depth_mae
    else:
        metrics['depth_mae'] = 0.0
    
    # Radius error (only on tree pixels)
    radius_pred = predictions['radius']
    radius_target = targets['radius']
    
    if mask.sum() > 0:
        radius_error = torch.abs(radius_pred - radius_target) * mask
        radius_mae = (radius_error.sum() / mask.sum()).item()
        metrics['radius_mae'] = radius_mae
    else:
        metrics['radius_mae'] = 0.0
    
    # Length error (only on tree pixels)
    length_pred = predictions['length']
    length_target = targets['length']
    
    if mask.sum() > 0:
        length_error = torch.abs(length_pred - length_target) * mask
        length_mae = (length_error.sum() / mask.sum()).item()
        metrics['length_mae'] = length_mae
    else:
        metrics['length_mae'] = 0.0
    
    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                gradient_accumulation_steps=1, mixed_precision=False, log_batch=True):
    """
    Train for one epoch with batch and epoch-level logging.
    """
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_depth_loss = 0.0
    total_radius_loss = 0.0
    total_length_loss = 0.0
    total_ce_loss = 0.0  # Track cross-entropy loss separately
    
    # Accumulate metrics for epoch average
    epoch_metrics = {
        'segmentation_accuracy': [],
        'pixel_accuracy': [],
        'segmentation_iou': [],
        'depth_mae': [],
        'radius_mae': [],
        'length_mae': []
    }
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        rgb = batch['rgb'].to(device, non_blocking=True)
        camera_params = batch['camera_params'].to(device, non_blocking=True)
        
        targets = {
            'segmentation': batch['segmentation'].to(device, non_blocking=True),
            'depth_gt': batch['depth_gt'].to(device, non_blocking=True),
            'radius': batch['radius'].to(device, non_blocking=True),
            'length': batch['length'].to(device, non_blocking=True),
        }
        
        # Forward pass with mixed precision
        if mixed_precision:
            with torch.cuda.amp.autocast():
                predictions = model(rgb, camera_params)
                loss_dict = criterion(predictions, targets)
                loss = loss_dict['total'] / gradient_accumulation_steps
        else:
            predictions = model(rgb, camera_params)
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total'] / gradient_accumulation_steps
        
        # Debug: Check for NaN or zero losses (only on first batch of first epoch)
        if epoch == 0 and batch_idx == 0:
            print(f"\nDebug - First batch losses:")
            print(f"  Total loss: {loss_dict['total'].item():.6f}")
            print(f"  Seg loss: {loss_dict['segmentation'].item():.6f}")
            print(f"  Depth loss: {loss_dict['depth'].item():.6f}")
            print(f"  Radius loss: {loss_dict['radius'].item():.6f}")
            print(f"  Length loss: {loss_dict['length'].item():.6f}")
            print(f"  Seg target unique values: {torch.unique(targets['segmentation'])}")
            print(f"  Seg target sum (tree pixels): {(targets['segmentation'] > 0).sum().item()}")
            if torch.isnan(loss_dict['total']) or loss_dict['total'].item() == 0.0:
                print(f"  WARNING: Loss is NaN or zero!")
        
        # Backward pass
        if mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights only after accumulating gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            # Clear cache periodically
            if (batch_idx + 1) % (gradient_accumulation_steps * 5) == 0:
                torch.cuda.empty_cache()
        
        # Accumulate losses (use full loss, not divided)
        total_loss += loss_dict['total'].item()
        total_seg_loss += loss_dict['segmentation'].item()
        total_depth_loss += loss_dict['depth'].item()
        total_radius_loss += loss_dict['radius'].item()
        total_length_loss += loss_dict['length'].item()
        
        # Compute accuracy metrics
        batch_metrics = compute_accuracy_metrics(predictions, targets)
        for key in epoch_metrics:
            if key in batch_metrics:
                epoch_metrics[key].append(batch_metrics[key])
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Compute cross-entropy loss separately
        seg_pred_logits = predictions['segmentation']
        seg_target_labels = targets['segmentation']
        ce_loss_batch = F.cross_entropy(seg_pred_logits, seg_target_labels).item()
        total_ce_loss += ce_loss_batch
        
        # Log batch-level metrics (only log on gradient update steps)
        if log_batch and (batch_idx + 1) % gradient_accumulation_steps == 0:
            wandb.log({
                'train/batch/loss': loss_dict['total'].item(),
                'train/batch/segmentation_loss': loss_dict['segmentation'].item(),
                'train/batch/cross_entropy_loss': ce_loss_batch,
                'train/batch/depth_loss': loss_dict['depth'].item(),
                'train/batch/radius_loss': loss_dict['radius'].item(),
                'train/batch/length_loss': loss_dict['length'].item(),
                'train/batch/segmentation_accuracy': batch_metrics['segmentation_accuracy'],
                'train/batch/pixel_accuracy': batch_metrics.get('pixel_accuracy', batch_metrics['segmentation_accuracy']),
                'train/batch/segmentation_iou': batch_metrics['segmentation_iou'],
                'train/batch/depth_mae': batch_metrics['depth_mae'],
                'train/batch/radius_mae': batch_metrics['radius_mae'],
                'train/batch/length_mae': batch_metrics['length_mae'],
                'train/batch/learning_rate': current_lr,
                'batch': batch_idx + epoch * len(dataloader),
            })
    
    # Handle remaining gradients if any
    if len(dataloader) % gradient_accumulation_steps != 0:
        if mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
    
    # Compute epoch averages
    n_batches = len(dataloader)
    if n_batches == 0:
        return {
            'loss': 0.0,
            'segmentation_loss': 0.0,
            'cross_entropy_loss': 0.0,
            'depth_loss': 0.0,
            'radius_loss': 0.0,
            'length_loss': 0.0,
            'segmentation_accuracy': 0.0,
            'pixel_accuracy': 0.0,
            'segmentation_iou': 0.0,
            'depth_mae': 0.0,
            'radius_mae': 0.0,
            'length_mae': 0.0,
        }
    
    epoch_avg_metrics = {
        'loss': total_loss / n_batches,
        'segmentation_loss': total_seg_loss / n_batches,
        'cross_entropy_loss': total_ce_loss / n_batches,
        'depth_loss': total_depth_loss / n_batches,
        'radius_loss': total_radius_loss / n_batches,
        'length_loss': total_length_loss / n_batches,
    }
    
    for key in epoch_metrics:
        if len(epoch_metrics[key]) > 0:
            epoch_avg_metrics[key] = np.mean(epoch_metrics[key])
        else:
            epoch_avg_metrics[key] = 0.0
    
    return epoch_avg_metrics


def validate(model, dataloader, criterion, device, epoch, log_batch=True):
    """
    Validate with batch and epoch-level logging.
    """
    model.eval()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_depth_loss = 0.0
    total_radius_loss = 0.0
    total_length_loss = 0.0
    
    # Accumulate metrics for epoch average
    epoch_metrics = {
        'segmentation_accuracy': [],
        'pixel_accuracy': [],
        'segmentation_iou': [],
        'depth_mae': [],
        'radius_mae': [],
        'length_mae': []
    }
    
    # Track cross-entropy loss separately
    total_ce_loss = 0.0
    
    n_batches = len(dataloader)
    if n_batches == 0:
        return {
            'loss': float('inf'),
            'segmentation_loss': float('inf'),
            'cross_entropy_loss': float('inf'),
            'depth_loss': float('inf'),
            'radius_loss': float('inf'),
            'length_loss': float('inf'),
            'segmentation_accuracy': 0.0,
            'pixel_accuracy': 0.0,
            'segmentation_iou': 0.0,
            'depth_mae': 0.0,
            'radius_mae': 0.0,
            'length_mae': 0.0,
        }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation Epoch {epoch}")):
            rgb = batch['rgb'].to(device)
            camera_params = batch['camera_params'].to(device)
            
            targets = {
                'segmentation': batch['segmentation'].to(device),
                'depth_gt': batch['depth_gt'].to(device),
                'radius': batch['radius'].to(device),
                'length': batch['length'].to(device),
            }
            
            # Forward pass
            predictions = model(rgb, camera_params)
            
            # Compute loss
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total']
            
            # Accumulate losses
            total_loss += loss.item()
            total_seg_loss += loss_dict['segmentation'].item()
            total_depth_loss += loss_dict['depth'].item()
            total_radius_loss += loss_dict['radius'].item()
            total_length_loss += loss_dict['length'].item()
            
            # Compute cross-entropy loss separately
            seg_pred_logits = predictions['segmentation']
            seg_target_labels = targets['segmentation']
            ce_loss_batch = F.cross_entropy(seg_pred_logits, seg_target_labels).item()
            total_ce_loss += ce_loss_batch
            
            # Compute accuracy metrics
            batch_metrics = compute_accuracy_metrics(predictions, targets)
            for key in epoch_metrics:
                if key in batch_metrics:
                    epoch_metrics[key].append(batch_metrics[key])
            
            # Get current learning rate (from model's optimizer if available)
            # For validation, we'll use the learning rate from the last training step
            current_lr = 0.0  # Will be set from training loop
            
            # Log batch-level metrics
            if log_batch and batch_idx % 5 == 0:  # Log every 5 batches
                wandb.log({
                    'val/batch/loss': loss.item(),
                    'val/batch/segmentation_loss': loss_dict['segmentation'].item(),
                    'val/batch/cross_entropy_loss': ce_loss_batch,
                    'val/batch/depth_loss': loss_dict['depth'].item(),
                    'val/batch/radius_loss': loss_dict['radius'].item(),
                    'val/batch/length_loss': loss_dict['length'].item(),
                    'val/batch/segmentation_accuracy': batch_metrics['segmentation_accuracy'],
                    'val/batch/pixel_accuracy': batch_metrics.get('pixel_accuracy', batch_metrics['segmentation_accuracy']),
                    'val/batch/segmentation_iou': batch_metrics['segmentation_iou'],
                    'val/batch/depth_mae': batch_metrics['depth_mae'],
                    'val/batch/radius_mae': batch_metrics['radius_mae'],
                    'val/batch/length_mae': batch_metrics['length_mae'],
                    'batch': batch_idx + epoch * len(dataloader),
                })
    
    # Compute epoch averages
    if n_batches == 0:
        return {
            'loss': float('inf'),
            'segmentation_loss': float('inf'),
            'cross_entropy_loss': float('inf'),
            'depth_loss': float('inf'),
            'radius_loss': float('inf'),
            'length_loss': float('inf'),
            'segmentation_accuracy': 0.0,
            'pixel_accuracy': 0.0,
            'segmentation_iou': 0.0,
            'depth_mae': 0.0,
            'radius_mae': 0.0,
            'length_mae': 0.0,
        }
    
    epoch_avg_metrics = {
        'loss': total_loss / n_batches,
        'segmentation_loss': total_seg_loss / n_batches,
        'cross_entropy_loss': total_ce_loss / n_batches,
        'depth_loss': total_depth_loss / n_batches,
        'radius_loss': total_radius_loss / n_batches,
        'length_loss': total_length_loss / n_batches,
    }
    
    for key in epoch_metrics:
        if len(epoch_metrics[key]) > 0:
            epoch_avg_metrics[key] = np.mean(epoch_metrics[key])
        else:
            epoch_avg_metrics[key] = 0.0
    
    return epoch_avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train image-based multi-task tree model')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='Directory with rgb/, depth/, ann/ from generate_dataset.py')
    parser.add_argument('--metadata_dir', type=str, required=True,
                       help='Directory with tree metadata JSONs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[480, 640],
                       help='Image size (H, W)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size (reduce if OOM, default: 2)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Number of gradient accumulation steps (effective batch = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (FP16) to save memory')
    parser.add_argument('--use_camera', action='store_true', default=True,
                       help='Use camera parameters in model')
    parser.add_argument('--seg_weight', type=float, default=1.0)
    parser.add_argument('--depth_weight', type=float, default=1.0)
    parser.add_argument('--radius_weight', type=float, default=1.0)
    parser.add_argument('--length_weight', type=float, default=1.0)
    parser.add_argument('--dice_weight', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='./checkpoints_image')
    parser.add_argument('--wandb_project', type=str, default='tree-image-multitask')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--log_batch', action='store_true', default=True,
                       help='Log batch-level metrics')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                'dataset_dir': args.dataset_dir,
                'metadata_dir': args.metadata_dir,
                'image_size': args.image_size,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'use_camera': args.use_camera,
                'seg_weight': args.seg_weight,
                'depth_weight': args.depth_weight,
                'radius_weight': args.radius_weight,
                'length_weight': args.length_weight,
                'dice_weight': args.dice_weight,
            }
        )
        print(f"Initialized wandb run: {wandb.run.name}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if not args.no_wandb:
        wandb.config.update({
            'device': str(device),
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
            'mixed_precision': args.mixed_precision
        })
    
    # Datasets
    train_dataset = ImageTreeDataset(
        args.dataset_dir,
        args.metadata_dir,
        split='train',
        image_size=tuple(args.image_size)
    )
    val_dataset = ImageTreeDataset(
        args.dataset_dir,
        args.metadata_dir,
        split='val',
        image_size=tuple(args.image_size)
    )
    
    # Reduce num_workers to save memory
    num_workers = min(2, args.batch_size)  # Use fewer workers for smaller batches
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    
    if not args.no_wandb:
        wandb.config.update({
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
        })
    
    # Model
    model = ResNetUNetMultiTask(
        num_classes=2, 
        use_camera=args.use_camera,
        image_size=tuple(args.image_size)
    ).to(device)
    
    # Loss and optimizer
    criterion = MultiTaskLoss(
        seg_weight=args.seg_weight,
        depth_weight=args.depth_weight,
        radius_weight=args.radius_weight,
        length_weight=args.length_weight,
        dice_weight=args.dice_weight
    )
    # Adjust learning rate for gradient accumulation (effective batch size)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    adjusted_lr = args.lr * (effective_batch_size / 8.0)  # Scale from base batch size 8
    optimizer = optim.Adam(model.parameters(), lr=adjusted_lr)
    print(f"Using learning rate: {adjusted_lr:.6f} (base: {args.lr:.6f}, effective batch: {effective_batch_size})")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    has_validation = len(val_loader) > 0
    
    if not has_validation:
        print("Warning: No validation samples available.")
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_batch=args.log_batch
        )
        
        # Validate
        if has_validation:
            val_metrics = validate(
                model, val_loader, criterion, device, epoch, args.log_batch
            )
            scheduler.step(val_metrics['loss'])
        else:
            val_metrics = train_metrics
            scheduler.step(train_metrics['loss'])
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Seg Acc: {train_metrics['segmentation_accuracy']:.4f}, "
              f"IoU: {train_metrics['segmentation_iou']:.4f}, "
              f"Depth MAE: {train_metrics['depth_mae']:.4f}, "
              f"Radius MAE: {train_metrics['radius_mae']:.4f}, "
              f"Length MAE: {train_metrics['length_mae']:.4f}")
        
        if has_validation:
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Seg Acc: {val_metrics['segmentation_accuracy']:.4f}, "
                  f"IoU: {val_metrics['segmentation_iou']:.4f}, "
                  f"Depth MAE: {val_metrics['depth_mae']:.4f}, "
                  f"Radius MAE: {val_metrics['radius_mae']:.4f}, "
                  f"Length MAE: {val_metrics['length_mae']:.4f}")
        
        # Log epoch-level metrics to wandb
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'learning_rate': current_lr,
                'train/epoch/learning_rate': current_lr,
                
                # Train epoch metrics
                'train/epoch/loss': train_metrics['loss'],
                'train/epoch/segmentation_loss': train_metrics['segmentation_loss'],
                'train/epoch/cross_entropy_loss': train_metrics.get('cross_entropy_loss', 0.0),
                'train/epoch/depth_loss': train_metrics['depth_loss'],
                'train/epoch/radius_loss': train_metrics['radius_loss'],
                'train/epoch/length_loss': train_metrics['length_loss'],
                'train/epoch/segmentation_accuracy': train_metrics['segmentation_accuracy'],
                'train/epoch/pixel_accuracy': train_metrics.get('pixel_accuracy', train_metrics['segmentation_accuracy']),
                'train/epoch/segmentation_iou': train_metrics['segmentation_iou'],
                'train/epoch/depth_mae': train_metrics['depth_mae'],
                'train/epoch/radius_mae': train_metrics['radius_mae'],
                'train/epoch/length_mae': train_metrics['length_mae'],
            }
            
            if has_validation:
                log_dict.update({
                    # Val epoch metrics
                    'val/epoch/loss': val_metrics['loss'],
                    'val/epoch/segmentation_loss': val_metrics['segmentation_loss'],
                    'val/epoch/cross_entropy_loss': val_metrics.get('cross_entropy_loss', 0.0),
                    'val/epoch/depth_loss': val_metrics['depth_loss'],
                    'val/epoch/radius_loss': val_metrics['radius_loss'],
                    'val/epoch/length_loss': val_metrics['length_loss'],
                    'val/epoch/segmentation_accuracy': val_metrics['segmentation_accuracy'],
                    'val/epoch/pixel_accuracy': val_metrics.get('pixel_accuracy', val_metrics['segmentation_accuracy']),
                    'val/epoch/segmentation_iou': val_metrics['segmentation_iou'],
                    'val/epoch/depth_mae': val_metrics['depth_mae'],
                    'val/epoch/radius_mae': val_metrics['radius_mae'],
                    'val/epoch/length_mae': val_metrics['length_mae'],
                })
            
            wandb.log(log_dict)
        
        # Save best model
        val_loss = val_metrics['loss'] if has_validation else train_metrics['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics if has_validation else None,
            }, checkpoint_path)
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
            
            if not args.no_wandb:
                wandb.save(checkpoint_path)
    
    print("Training complete!")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

