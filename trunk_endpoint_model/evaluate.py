"""
Evaluation script for trunk endpoint regression.
"""
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb

from data_loader import TrunkEndpointDataset
from models import PointNet, PointNetPlusPlus


def compute_metrics(pred_endpoints, target_endpoints):
    """
    Compute endpoint error and axis error.
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate trunk endpoint regression model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--include_normals', action='store_true')
    parser.add_argument('--model', type=str, default='pointnet',
                       choices=['pointnet', 'pointnet++'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    parser.add_argument('--wandb_project', type=str, default='trunk-endpoint-regression',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Weights & Biases run name for evaluation')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Initialize Weights & Biases for evaluation
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"eval-{args.split}",
            job_type="evaluation",
            config={
                'checkpoint': args.checkpoint,
                'split': args.split,
                'num_points': args.num_points,
                'include_normals': args.include_normals,
                'model': args.model,
                'batch_size': args.batch_size,
            }
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    dataset = TrunkEndpointDataset(
        args.data_dir,
        num_points=args.num_points,
        include_normals=args.include_normals,
        split=args.split
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=4)
    
    # Model
    input_dim = 6 if args.include_normals else 3
    if args.model == 'pointnet':
        model = PointNet(args.num_points, input_dim).to(device)
    else:
        model = PointNetPlusPlus(args.num_points, input_dim).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate
    all_endpoint_errors = []
    all_axis_errors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            points = batch['points'].to(device)
            endpoints = batch['endpoints'].to(device)
            
            pred = model(points)
            endpoint_error, axis_error = compute_metrics(pred, endpoints)
            
            all_endpoint_errors.append(endpoint_error)
            all_axis_errors.append(axis_error)
    
    mean_endpoint_error = np.mean(all_endpoint_errors)
    mean_axis_error = np.mean(all_axis_errors)
    std_endpoint_error = np.std(all_endpoint_errors)
    std_axis_error = np.std(all_axis_errors)
    
    print(f"\nEvaluation Results ({args.split} set):")
    print(f"Endpoint Error: {mean_endpoint_error:.4f} ± {std_endpoint_error:.4f}")
    print(f"Axis Error (rad): {mean_axis_error:.4f} ± {std_axis_error:.4f}")
    print(f"Axis Error (deg): {np.degrees(mean_axis_error):.2f} ± {np.degrees(std_axis_error):.2f}")
    
    # Log to wandb
    if not args.no_wandb:
        wandb.log({
            f'eval_{args.split}/endpoint_error': mean_endpoint_error,
            f'eval_{args.split}/endpoint_error_std': std_endpoint_error,
            f'eval_{args.split}/axis_error_rad': mean_axis_error,
            f'eval_{args.split}/axis_error_rad_std': std_axis_error,
            f'eval_{args.split}/axis_error_deg': np.degrees(mean_axis_error),
            f'eval_{args.split}/axis_error_deg_std': np.degrees(std_axis_error),
            f'eval_{args.split}/num_samples': len(dataset),
        })
        
        wandb.summary.update({
            f'{args.split}_endpoint_error': mean_endpoint_error,
            f'{args.split}_axis_error_deg': np.degrees(mean_axis_error),
        })
        
        wandb.finish()


if __name__ == '__main__':
    main()