# trunk_endpoint_model/image_losses.py
"""
Multi-task loss: Segmentation (Dice + CE) + Depth/Radius/Length (Masked Huber)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, 
                 seg_weight=1.0,
                 depth_weight=1.0,
                 radius_weight=1.0,
                 length_weight=1.0,
                 dice_weight=0.5):
        super(MultiTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.depth_weight = depth_weight
        self.radius_weight = radius_weight
        self.length_weight = length_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.HuberLoss(delta=1.0)
    
    def dice_loss(self, pred, target, num_classes=2):
        """Dice loss for segmentation"""
        dice_scores = []
        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            dice_scores.append(dice)
        return 1 - torch.mean(torch.stack(dice_scores))
    
    def masked_huber_loss(self, pred, target, mask):
        """
        Huber loss applied only to masked (tree) pixels.
        Args:
            pred: (B, 1, H, W) predictions
            target: (B, 1, H, W) ground truth
            mask: (B, 1, H, W) binary mask (1 = tree pixel, 0 = background)
        """
        # Only compute loss on masked pixels
        pred_flat = pred[mask > 0]
        target_flat = target[mask > 0]
        
        if len(pred_flat) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Compute Huber loss element-wise
        diff = pred_flat - target_flat
        delta = 1.0
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        
        return loss.mean()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with 'segmentation', 'depth', 'radius', 'length'
            targets: dict with same keys + 'segmentation_mask' for masking
        """
        # Segmentation loss (Cross-entropy + Dice)
        seg_pred = predictions['segmentation']
        seg_target = targets['segmentation']  # (B, H, W) class indices
        
        ce = self.ce_loss(seg_pred, seg_target)
        dice = self.dice_loss(seg_pred.argmax(dim=1), seg_target)
        seg_loss = ce + self.dice_weight * dice
        
        # Create mask from segmentation (tree pixels = 1)
        mask = (seg_target > 0).float().unsqueeze(1)  # (B, 1, H, W)
        
        # Depth loss (masked Huber)
        depth_target = targets.get('depth_gt', targets.get('depth'))
        depth_loss = self.masked_huber_loss(
            predictions['depth'],
            depth_target,
            mask
        )
        
        # Radius loss (masked Huber)
        radius_loss = self.masked_huber_loss(
            predictions['radius'],
            targets['radius'],
            mask
        )
        
        # Length loss (masked Huber)
        length_loss = self.masked_huber_loss(
            predictions['length'],
            targets['length'],
            mask
        )
        
        # Total loss
        total_loss = (
            self.seg_weight * seg_loss +
            self.depth_weight * depth_loss +
            self.radius_weight * radius_loss +
            self.length_weight * length_loss
        )
        
        return {
            'total': total_loss,
            'segmentation': seg_loss,
            'depth': depth_loss,
            'radius': radius_loss,
            'length': length_loss
        }