"""
Simple U-Net architecture for box segmentation.
Includes MLP baseline, CNN, and U-Net for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPSegmentation(nn.Module):
    """
    Simple MLP model for segmentation.
    Flattens input, applies multiple linear layers, reshapes to output mask.
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(MLPSegmentation, self).__init__()
        
        # PathMNIST images are 3x28x28 = 2352 input features
        # Output should be 1x28x28 = 784 features
        self.in_channels = in_channels
        self.out_channels = out_channels
        input_size = in_channels * 28 * 28        
        output_size = out_channels * 28 * 28

        # TODO: Add your own MLP architecture here
        self.flatten = nn.Flatten()
        self.h1= nn.Linear(input_size, 512)
        self.h2= nn.Linear(512, 256)
        self.h3= nn.Linear(256, 512)
        self.h4=nn.Linear(512,output_size)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        x = self.flatten(x)         
        x = self.relu(self.h1(x))
        x = self.relu(self.h2(x))
        x = self.relu(self.h3(x))
        x =  self.h4(x)
        x = x.view(-1, self.out_channels, 28, 28) 
        return torch.sigmoid(x)
        #raise NotImplementedError("MLPSegmentation is not implemented")

class TinyUNet(nn.Module):
    """
    Tiny U-Net for segmentation of 28x28 images.
    Optimized for small images and simple segmentation tasks.
    """
    
    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super(TinyUNet, self).__init__()
        
        # Encoder (contracting path)
        # TODO: Add your own encoder architecture here
        
        a = base_channels
        self.pool = nn.MaxPool2d(2)

        
        self.enc1 = self._double_conv(in_channels, a)
        self.enc2 = self._double_conv(a, a * 2)
        self.bottleneck = self._double_conv(a * 2, a * 4)

        self.up2 = nn.ConvTranspose2d(a * 4, a * 2, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(a * 4, a * 2)
        self.up1 = nn.ConvTranspose2d(a * 2, a, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(a * 2, a)

        self.out_conv = nn.Conv2d(a, out_channels, kernel_size=1)
    @staticmethod
    def _double_conv(cin, cout):
            return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))
        #raise NotImplementedError("TinyUNet is not implemented")

class SimpleCNN(nn.Module):
    """
    Simple CNN model: TODO: Add your own architecture here
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleCNN, self).__init__()
        

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.decoder = nn.Sequential (
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), 
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=1)

        )
    
    def forward(self, x):

        x = self.features(x)           
        x = self.decoder(x)           
        return torch.sigmoid(x)  
        #raise NotImplementedError("CNNModel is not implemented")

def get_segmentation_model(model_name, in_channels=3, out_channels=1):
    """Get segmentation model by name."""
    if model_name == 'mlp':
        return MLPSegmentation(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'unet':
        return TinyUNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'cnn':
        return SimpleCNN(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError("Unknown segmentation model: {}".format(model_name))

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Intersection over Union (IoU) metric for segmentation
def calculate_iou(pred_mask, true_mask, threshold=0.5):
    """
    Calculate Intersection over Union for binary segmentation masks.
    
    Args:
        pred_mask: Predicted segmentation mask [B, 1, H, W] or [B, H, W]
        true_mask: Ground truth segmentation mask [B, 1, H, W] or [B, H, W]
        threshold: Threshold for binarizing predictions
    
    Returns:
        IoU score (float)
    """
    # Convert to binary
    if torch.is_tensor(pred_mask):
        pred_binary = (pred_mask > threshold).float()
    else:
        pred_binary = (pred_mask > threshold).astype(float)
    
    if torch.is_tensor(true_mask):
        true_binary = (true_mask > 0.5).float()
    else:
        true_binary = (true_mask > 0.5).astype(float)
    
    # Flatten for easier computation
    if len(pred_binary.shape) > 2:
        pred_binary = pred_binary.view(pred_binary.size(0), -1)
        true_binary = true_binary.view(true_binary.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred_binary * true_binary).sum(dim=-1)
    union = pred_binary.sum(dim=-1) + true_binary.sum(dim=-1) - intersection
    
    # Handle case where both masks are empty
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    
    return iou.mean().item() if torch.is_tensor(iou) else iou.mean()

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    Better than BCE for imbalanced segmentation.
    """
    
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        #TODO: Compute the DICE loss
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)


        loss = 1 - dice.mean()
        return loss 

class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss for better segmentation performance.
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight      # <-- add this
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss() # TODO: Initialize the BCE loss
        self.dice_loss = DiceLoss() # TODO: Initialize the DICE loss
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice
