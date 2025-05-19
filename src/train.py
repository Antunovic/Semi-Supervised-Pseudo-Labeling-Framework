
import os
import warnings
from pathlib import Path

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


import config
import metrics
import initial_training
from model import get_model
from data_setup import create_dataloader
from metrics import compute_miou
from utils import (
    generate_pseudolabels,
    combine_and_save_pseudolabels,
    load_model,
    evaluate,
    compute_fire_confidence,
    create_directory_structure,
    create_directory_structure_full_supervision,
    copy_images,
    split_data,
    split_data_full_supervision,
    move_and_convert_pseudo_labels
)
from pseudo_training import pseudo_training

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def train_step(model:torch.nn.Module,
          dataloader:torch.utils.data.DataLoader,
          loss_fn:torch.nn.Module,
          optimizer:torch.optim.Optimizer,
          device:torch.device,
          use_weights: bool = False):
    
    model.train()

    train_loss = 0.0
    train_miou = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=True):
        
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        #masks = masks.to(torch.float32)

        #Optimizer zero gradient
        optimizer.zero_grad()

        #Forward pass
        output_logits = model(images)

        if use_weights:
            # Split masks into binary mask (ground truth) and weights
            binary_masks = masks[:, 0:1, :, :]  # Shape: (B, 1, H, W)
            #print(np.unique(binary_masks, return_counts=True))
            weights = masks[:, 1:2, :, :]       # Shape: (B, 1, H, W) 

            loss = loss_fn(output_logits, binary_masks, weights)  # Weighted BCE loss

        else:
            #Calculate the loss
            loss = loss_fn(output_logits,masks)

        loss.backward()

        #Optimizer step
        optimizer.step()

        train_loss += loss.item()
        train_miou += compute_miou(output_logits,masks)

    return train_loss / len(dataloader), train_miou / len(dataloader)


def validate(model:torch.nn.Module,
          dataloader:torch.utils.data.DataLoader,
          loss_fn:torch.nn.Module,
          device:torch.device,
          use_weights:bool=False):
    model.eval()
    val_loss=0.0
    val_miou=0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=True):
            images, masks = images.to(device),masks.to(device)
            
            output_logits = model(images)
            
            loss = loss_fn(output_logits,masks)

            val_loss += loss.item()
            val_miou += compute_miou(output_logits,masks)

    return val_loss / len(dataloader), val_miou / len(dataloader)

def train(epochs,model, train_dataloader, val_dataloader, loss_fn, optimizer, device, use_weights=False):

    best_miou = 0.0  # Initialize best mIoU
    best_model_state = None  # Placeholder for best model

    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")
        # Train for one epoch
        if use_weights:
            train_loss, train_miou = train_step(model, train_dataloader, loss_fn, optimizer, device, use_weights=use_weights)
            print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")

            # Validate using clasical BCE loss because no need for weights when calculating validation loss
            val_loss, val_miou = validate(model, val_dataloader, loss_fn = torch.nn.BCEWithLogitsLoss(), device=device, use_weights=use_weights)
            print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_miou:.4f}")
        
        else:
            train_loss, train_miou = train_step(model, train_dataloader, loss_fn, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")

            # Validate
            val_loss, val_miou = validate(model, val_dataloader, loss_fn, device)
            print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_miou:.4f}")


        # === Save the best model ===
        if val_miou > best_miou:
            best_miou = val_miou
            best_model_state = model.state_dict()
            print(f"🔹 New best model found! Saving model with mIoU: {best_miou:.4f}")

    print(f"\n✅ Training complete. Best model had mIoU: {best_miou:.4f}")

    return best_model_state



def main():
    
    print("Creating direcotry structure...")
    create_directory_structure(config.BASE_PATH)

    print("spliting data...")
    split_data(images_dir=config.IMAGES_DIR, masks_dir=config.MASKS_DIR,base_path=config.BASE_PATH, seed=config.SEED)

    print("Initializing models...")
    model_1, model_2, initial_weights_1, initial_weights_2 = get_model()
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    model_1.to(device)
    model_2.to(device)
    warnings.filterwarnings("ignore", category=UserWarning)
    initial_training.initial_training(model_1=model_1,
                     model_2=model_2,
                     device=device,
                     checkpoint_dir=config.CHECKPOINT_DIR)

    
    #Generate pseudo labels from both models and save them
    print("Generating pseudo-labels")
    generate_pseudolabels(model_1_path=Path(config.CHECKPOINT_DIR) / "model_1.pth",
                          model_2_path=Path(config.CHECKPOINT_DIR) / "model_2.pth",
                          image_dir=config.UNLABELED_IMAGES_DIR,
                          output_dir=config.PSEUDOMASKS_DIR,
                          encoder_name=config.ENCODER_NAME,
                          in_channels=config.IN_CHANNELS,
                          classes=config.CLASSES,
                          device=device)

    #Combine and save generated pseudolabels with weights for calculating loss
    combine_and_save_pseudolabels(pseudo_labels_path_src=config.PSEUDOMASKS_DIR,
                                pseudo_labels_path_dst=config.COMBINED_PSEUDOMASKS_DIR,
                                device=device)

    df = compute_fire_confidence(config.COMBINED_PSEUDOMASKS_DIR)
    move_and_convert_pseudo_labels(df)
        

    for i in range(5):
        print(f"---------------ITERATION: {i}--------------------")
        #Reinitialize models with inital weights
        model_1, model_2, _, _ = get_model()
        model_1.to(device)
        model_2.to(device)

        print("PseudoTraining...")
        
        pseudo_training(model_1=model_1,
                        model_2=model_2,
                        device=device,
                        checkpoint_dir=config.CHECKPOINT_DIR,
                        i=i)
        
        print("Generating pseudo-labels with pseudo models")
        generate_pseudolabels(model_1_path=Path(config.CHECKPOINT_DIR) / f"model_1_pseudo_iter_{i}.pth",
                            model_2_path=Path(config.CHECKPOINT_DIR) / f"model_2_pseudo_iter_{i}.pth",
                            image_dir=config.UNLABELED_IMAGES_DIR,
                            output_dir=config.PSEUDOMASKS_DIR,
                            encoder_name=config.ENCODER_NAME,
                            in_channels=config.IN_CHANNELS,
                            classes=config.CLASSES,
                            device=device)
        
        combine_and_save_pseudolabels(pseudo_labels_path_src=config.PSEUDOMASKS_DIR,
                                pseudo_labels_path_dst=config.COMBINED_PSEUDOMASKS_DIR,
                                device=device)
        
        df = compute_fire_confidence(config.COMBINED_PSEUDOMASKS_DIR)
        move_and_convert_pseudo_labels(df)
    

    test_dataloader = create_dataloader(images_dir=config.VAL_IMAGES_DIR,masks_dir=config.VAL_MASKS_DIR,batch_size=16,is_train=False)

    evaluate(config.MODEL_PATHS,dataloader=test_dataloader,device=device)
    

if __name__ == "__main__":
    main()