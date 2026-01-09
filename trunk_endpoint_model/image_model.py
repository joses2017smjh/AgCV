# trunk_endpoint_model/image_models.py
"""
ResNet encoder + U-Net decoder for multi-task learning.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNetUNetMultiTask(nn.Module):
    """
    ResNet encoder + U-Net decoder for:
    - Segmentation (tree pixels)
    - Depth estimation
    - Radius regression
    - Length regression
    """
    def __init__(self, num_classes=2, use_camera=True, image_size=(480, 640)):
        super(ResNetUNetMultiTask, self).__init__()
        self.use_camera = use_camera
        self.image_size = image_size
        
        # ResNet encoder (backbone)
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 256 channels
            resnet.layer2,  # 512 channels
            resnet.layer3,  # 1024 channels
            resnet.layer4,  # 2048 channels
        )
        
        # Camera feature fusion (optional)
        if use_camera:
            self.camera_fc = nn.Sequential(
                nn.Linear(9, 128),  # 3 location + 3 rotation + 3 intrinsics (flattened)
                nn.ReLU(),
                nn.Linear(128, 256),
            )
        
        # U-Net decoder with target size
        self.decoder = UNetDecoder(2048, 256, target_size=image_size)
        
        # Multi-task heads
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)  # num_classes = 2 (background, tree)
        )
        
        # Depth head
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)  # Single channel depth
        )
        
        # Radius head
        self.radius_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)  # Single channel radius
        )
        
        # Length head
        self.length_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)  # Single channel length
        )
    
    def forward(self, x, camera_params=None):
        # Encoder
        features = self.encoder(x)  # (B, 2048, H/32, W/32)
        
        # Optional: fuse camera parameters
        if self.use_camera and camera_params is not None:
            cam_feat = self.camera_fc(camera_params)  # (B, 256)
            # Use camera features to modulate decoder features instead
            # We'll store this for use in decoder
            self._camera_features = cam_feat
        
        # Decoder
        decoded = self.decoder(features)  # (B, 256, H, W)
        
        # Fuse camera features into decoder output if available
        if self.use_camera and camera_params is not None and hasattr(self, '_camera_features'):
            cam_feat = self._camera_features.unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)
            decoded = decoded + cam_feat  # Broadcast add
        
        # Multi-task predictions
        seg = self.seg_head(decoded)  # (B, num_classes, H, W)
        depth = self.depth_head(decoded)  # (B, 1, H, W)
        radius = self.radius_head(decoded)  # (B, 1, H, W)
        length = self.length_head(decoded)  # (B, 1, H, W)
        
        return {
            'segmentation': seg,
            'depth': depth,
            'radius': radius,
            'length': length
        }


class UNetDecoder(nn.Module):
    """U-Net style decoder with skip connections"""
    def __init__(self, in_channels, out_channels, target_size=None):
        super(UNetDecoder, self).__init__()
        self.target_size = target_size
        
        # Upsampling layers
        # ResNet reduces by 32x, so we need 5 upsampling steps (2^5 = 32)
        self.up1 = nn.ConvTranspose2d(in_channels, 512, 2, stride=2)
        self.conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv5 = nn.Conv2d(32, out_channels, 3, padding=1)
    
    def forward(self, x):
        # Get input size to determine if we need final interpolation
        B, C, H, W = x.shape
        
        x = F.relu(self.bn1(self.conv1(self.up1(x))))
        x = F.relu(self.bn2(self.conv2(self.up2(x))))
        x = F.relu(self.bn3(self.conv3(self.up3(x))))
        x = F.relu(self.bn4(self.conv4(self.up4(x))))
        x = self.conv5(self.up5(x))
        
        # If target size is specified and doesn't match, interpolate
        if self.target_size is not None:
            target_h, target_w = self.target_size
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return x