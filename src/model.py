from segmentation_models_pytorch import UnetPlusPlus
import config
import random
import numpy as np
import torch
import copy

def set_seeds(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disable optimization for consistent results




import torch
import copy
from segmentation_models_pytorch import UnetPlusPlus

def get_model(encoder_name=config.ENCODER_NAME,
              in_channels=config.IN_CHANNELS,
              classes=config.CLASSES,
              seed=50):
    """
    Returns two segmentation models with frozen encoder weights and saves
    their initial weights for reinitialization.

    Args:
        encoder_name (str): Name of the encoder (e.g., resnet34).
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        classes (int): Number of output classes.

    Returns:
        model_1, model_2, saved_weights_1, saved_weights_2: Two UNet++ models and
        their saved initial weights.
    """

    set_seeds(seed)

    # Initialize models with different pre-trained encoder weights
    model_1 = UnetPlusPlus(encoder_name=encoder_name,
                encoder_weights=config.ENCODER_WEIGHTS,
                in_channels=in_channels,
                classes=classes)
    
    model_2 = UnetPlusPlus(encoder_name=encoder_name,
                encoder_weights=config.ENCODER_WEIGHTS,
                in_channels=in_channels,
                classes=classes)

    # Save initial weights (ensures frozen encoder weights remain frozen when reloaded)
    initial_weights_model_1 = copy.deepcopy(model_1.state_dict())
    initial_weights_model_2 = copy.deepcopy(model_2.state_dict())

    return model_1, model_2, initial_weights_model_1, initial_weights_model_2

 
