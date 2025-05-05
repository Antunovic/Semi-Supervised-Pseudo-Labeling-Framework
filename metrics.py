import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_miou(pred, target, threshold=0.5):
    """
    Compute mean Intersection over Union (mIoU) for a batch.
    
    Args:
        pred (torch.Tensor): Model predictions of shape [batch_size, 1, H, W].
        target (torch.Tensor): Ground truth masks of shape [batch_size, 1, H, W].
        threshold (float): Threshold to binarize predictions.
    
    Returns:
        float: Mean IoU for the batch.
    """
    pred = (torch.sigmoid(pred) > threshold).float()  # Binarize predictions
    intersection = (pred * target).sum(dim=(2, 3))  # Intersection for each image
    union = (pred + target).clamp(0, 1).sum(dim=(2, 3))  # Union for each image
    iou = (intersection / (union + 1e-6))  # Avoid division by zero
    return iou.mean().item()  # Average IoU for the batch



# Define the loss function
loss_fn = nn.BCEWithLogitsLoss(reduction='none')  # Use 'none' to get per-pixel loss

def weighted_bce_with_logits_loss(logits, targets, weights):
    """
    Compute weighted binary cross-entropy loss with logits.

    Args:
        logits (torch.Tensor): The raw model predictions (before applying sigmoid).
                              Shape: (B, 1, H, W)
        targets (torch.Tensor): The ground truth binary masks. Shape: (B, 1, H, W)
        weights (torch.Tensor): The weight matrix for each pixel. Shape: (B, 1, H, W)

    Returns:
        torch.Tensor: Weighted loss (scalar).
    """
    # Compute the per-pixel loss
    per_pixel_loss = loss_fn(logits, targets)

    # Apply pixel-wise weights
    weighted_loss = per_pixel_loss * weights

    # Return the mean weighted loss
    return weighted_loss.mean()

import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
   
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)

    # Flatten the tensors for easier computation
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1)

    # Calculate intersection and union
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)

    # Return Dice Loss (mean over batch)
    return 1 - dice.mean()


def weighted_dice_loss(logits, targets, weights, smooth=1e-6):
    
    # Convert logits to probabilities
    preds = torch.sigmoid(logits)

    # Flatten tensors for batch-wise computation
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    weights = weights.view(weights.shape[0], -1)

    # Compute weighted intersection and union
    intersection = (preds * targets * weights).sum(dim=1)
    union = ((preds + targets) * weights).sum(dim=1)

    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)

    # Compute Weighted Dice Loss
    return 1 - dice.mean()

def combined_bce_dice_loss(logits, targets, smooth=1e-6, bce_weight=0.5, dice_weight=0.5):
    
    # Compute BCE Loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

    # Compute Dice Loss
    preds = torch.sigmoid(logits)  # Convert logits to probabilities
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
    dice_loss = dice_loss.mean()

    # Combine BCE and Dice Loss
    return bce_weight * bce_loss + dice_weight * dice_loss


def combined_weighted_bce_dice_loss(logits, masks, smooth=1e-6, bce_weight=0.5, dice_weight=0.5):
    """
    Compute a combined weighted Binary Cross Entropy (BCE) and weighted Dice Loss.

    Args:
        logits (torch.Tensor): Model output logits (before sigmoid), shape (B, 1, H, W)
        targets (torch.Tensor): Ground truth binary masks, shape (B, 1, H, W)
        weights (torch.Tensor): Pixel-wise confidence weights, shape (B, 1, H, W)
        smooth (float): Smoothing factor to avoid division by zero.
        bce_weight (float): Weight for BCE loss.
        dice_weight (float): Weight for Dice loss.

    Returns:
        torch.Tensor: Combined weighted loss value (scalar).
    """
    # Compute Weighted BCE Loss
    targets = masks[:, 0:1, :, :]  
    weights = masks[:, 1:2, :, :] 

    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    weighted_bce_loss = (bce_loss * weights).mean()

    # Compute Weighted Dice Loss
    preds = torch.sigmoid(logits)
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    weights = weights.view(weights.shape[0], -1)

    intersection = (preds * targets * weights).sum(dim=1)
    union = ((preds + targets) * weights).sum(dim=1)

    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)
    weighted_dice_loss = dice_loss.mean()

    # Combine Weighted BCE and Dice Loss
    return bce_weight * weighted_bce_loss + dice_weight * weighted_dice_loss