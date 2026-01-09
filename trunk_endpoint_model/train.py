"""
Training script for trunk endpoint regression.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import wandb
import numpy as np

from data_loader import TrunkEndpointDataset
from models import PointNet, PointNetPlusPlus
from losses import EndpointLoss

os.environ['WANDB_API_KEY'] ='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'

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_end_loss = 0.0
    total_dir_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        points = batch['points'].to(device)
        endpoints = batch['endpoints'].to(device)
        
        optimizer.zero_grad()
        
        pred = model(points)
        loss, loss_end, loss_dir = criterion(pred, endpoints)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_end_loss += loss_end.item()
        total_dir_loss += loss_dir.item()
    
    n_batches = len(dataloader)
    return (total_loss / n_batches, 
            total_end_loss / n_batches, 
            total_dir_loss / n_batches)


def compute_metrics(pred_endpoints, target_endpoints):
    """
    Compute endpoint error and axis error for evaluation metrics.
    """
    # Handle order ambiguity
    loss1 = torch.norm(pred_endpoints[:, 0] - target_endpoints[:, 0], dim=1) + \
            torch.norm(pred_endpoints[:, 1] - target_endpoints[:, 1], dim=1)
    
    loss2 = torch.norm(pred_endpoints[:, 0] - target_endpoints[:, 1], dim=1) + \
            torch.norm(pred_endpoints[:, 1] - target_endpoints[:, 0], dim=1)
    
    use_order2 = loss2 < loss1
    
    # Endpoint error (with best matching)
    p1_pred = torch.where(use_order2.unsqueeze(1), 
                         pred_endpoints[:, 1], 
                         pred_endpoints[:, 0])
    p2_pred = torch.where(use_order2.unsqueeze(1), 
                         pred_endpoints[:, 0], 
                         pred_endpoints[:, 1])
    
    p1_target = target_endpoints[:, 0]
    p2_target = target_endpoints[:, 1]
    
    endpoint_error = ((torch.norm(p1_pred - p1_target, dim=1) + 
                      torch.norm(p2_pred - p2_target, dim=1)) / 2).mean().item()
    
    # Axis error
    pred_dir = p2_pred - p1_pred
    target_dir = p2_target - p1_target
    
    pred_dir = pred_dir / (torch.norm(pred_dir, dim=1, keepdim=True) + 1e-8)
    target_dir = target_dir / (torch.norm(target_dir, dim=1, keepdim=True) + 1e-8)
    
    cos_sim = (pred_dir * target_dir).sum(dim=1).clamp(-1, 1)
    axis_error = torch.acos(torch.abs(cos_sim)).mean().item()
    
    return endpoint_error, axis_error


def validate(model, dataloader, criterion, device, compute_accuracy=False):
    model.eval()
    total_loss = 0.0
    total_end_loss = 0.0
    total_dir_loss = 0.0
    all_endpoint_errors = []
    all_axis_errors = []
    
    n_batches = len(dataloader)
    if n_batches == 0:
        # Return dummy values if validation set is empty
        print("Warning: Validation set is empty, returning dummy loss values")
        return (float('inf'), float('inf'), float('inf'), 0.0, 0.0)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            points = batch['points'].to(device)
            endpoints = batch['endpoints'].to(device)
            
            pred = model(points)
            loss, loss_end, loss_dir = criterion(pred, endpoints)
            
            total_loss += loss.item()
            total_end_loss += loss_end.item()
            total_dir_loss += loss_dir.item()
            
            # Compute accuracy metrics
            if compute_accuracy:
                endpoint_error, axis_error = compute_metrics(pred, endpoints)
                all_endpoint_errors.append(endpoint_error)
                all_axis_errors.append(axis_error)
    
    avg_endpoint_error = np.mean(all_endpoint_errors) if all_endpoint_errors else 0.0
    avg_axis_error = np.mean(all_axis_errors) if all_axis_errors else 0.0
    
    return (total_loss / n_batches, 
            total_end_loss / n_batches, 
            total_dir_loss / n_batches,
            avg_endpoint_error,
            avg_axis_error)


def main():
    parser = argparse.ArgumentParser(description='Train trunk endpoint regression model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing ply/ and metadata/ folders')
    parser.add_argument('--num_points', type=int, default=2048,
                       help='Number of points to sample per cloud')
    parser.add_argument('--include_normals', action='store_true',
                       help='Include normal vectors in point features')
    parser.add_argument('--model', type=str, default='pointnet',
                       choices=['pointnet', 'pointnet++'],
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Weight for direction loss')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--wandb_project', type=str, default='trunk-endpoint-regression',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Weights & Biases run name (default: auto-generated)')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Weights & Biases
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                'num_points': args.num_points,
                'include_normals': args.include_normals,
                'model': args.model,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.lr,
                'beta': args.beta,
                'data_dir': args.data_dir,
            }
        )
        print(f"Initialized wandb run: {wandb.run.name}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not args.no_wandb:
        wandb.config.update({'device': str(device)})
    
    # Datasets
    train_dataset = TrunkEndpointDataset(
        args.data_dir, 
        num_points=args.num_points,
        include_normals=args.include_normals,
        split='train'
    )
    val_dataset = TrunkEndpointDataset(
        args.data_dir,
        num_points=args.num_points,
        include_normals=args.include_normals,
        split='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    if not args.no_wandb:
        wandb.config.update({
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
        })
    
    # Model
    input_dim = 6 if args.include_normals else 3
    if args.model == 'pointnet':
        model = PointNet(args.num_points, input_dim).to(device)
    else:
        model = PointNetPlusPlus(args.num_points, input_dim).to(device)
    
    # Loss and optimizer
    criterion = EndpointLoss(beta=args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    has_validation = len(val_loader) > 0
    
    if not has_validation:
        print("Warning: No validation samples available. Will use train loss for model selection.")
    
    for epoch in range(args.epochs):
        train_loss, train_end, train_dir = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        if has_validation:
            val_loss, val_end, val_dir, val_endpoint_error, val_axis_error = validate(
                model, val_loader, criterion, device, compute_accuracy=True
            )
            scheduler.step(val_loss)
        else:
            # Use train loss for scheduling if no validation
            val_loss, val_end, val_dir = train_loss, train_end, train_dir
            val_endpoint_error, val_axis_error = 0.0, 0.0
            scheduler.step(train_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} (end: {train_end:.4f}, dir: {train_dir:.4f})")
        if has_validation:
            print(f"  Val Loss: {val_loss:.4f} (end: {val_end:.4f}, dir: {val_dir:.4f})")
            print(f"  Val Endpoint Error: {val_endpoint_error:.4f}")
            print(f"  Val Axis Error: {np.degrees(val_axis_error):.2f}°")
        else:
            print(f"  Val Loss: N/A (no validation set)")
        
        # Log to wandb
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/endpoint_loss': train_end,
                'train/direction_loss': train_dir,
                'learning_rate': current_lr,
            }
            
            if has_validation:
                log_dict.update({
                    'val/loss': val_loss,
                    'val/endpoint_loss': val_end,
                    'val/direction_loss': val_dir,
                    'val/endpoint_error': val_endpoint_error,
                    'val/axis_error_rad': val_axis_error,
                    'val/axis_error_deg': np.degrees(val_axis_error),
                })
            
            wandb.log(log_dict)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss if has_validation else train_loss,
                'val_endpoint_error': val_endpoint_error if has_validation else 0.0,
                'val_axis_error': val_axis_error if has_validation else 0.0,
            }, checkpoint_path)
            print(f"  Saved best model (loss: {val_loss:.4f})")
            
            if not args.no_wandb:
                wandb.save(checkpoint_path)
    
    print("Training complete!")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()