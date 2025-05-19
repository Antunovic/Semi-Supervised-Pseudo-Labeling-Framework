import torch
import config
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import get_model
from data_setup import create_dataloader
from metrics import compute_miou,dice_loss,combined_bce_dice_loss
from tqdm.auto import tqdm
from utils import generate_pseudolabels,combine_and_save_pseudolabels
from train import train
import os

def initial_training(model_1, model_2, device, checkpoint_dir):
    print("Creating DataLoaders for initial training...")
    
    train_dataloader_1 = create_dataloader(
        images_dir=config.TRAIN_IMAGES_DIR_1,
        masks_dir=config.TRAIN_MASKS_DIR_1,
        batch_size=config.BATCH_SIZE
    )

    train_dataloader_2 = create_dataloader(
        images_dir=config.TRAIN_IMAGES_DIR_2,
        masks_dir=config.TRAIN_MASKS_DIR_2,
        batch_size=config.BATCH_SIZE
    )

    val_dataloader = create_dataloader(
        images_dir=config.VAL_IMAGES_DIR,
        masks_dir=config.VAL_MASKS_DIR,
        batch_size=config.BATCH_SIZE
    )

    loss_fn = combined_bce_dice_loss
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=config.LEARNING_RATE)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=config.LEARNING_RATE)

    epochs = config.EPOCHS

    # === Train model 1 ===
    print(" Starting training for Model 1...")
    best_model_1_weights = train(
        epochs=epochs,
        model=model_1,
        train_dataloader=train_dataloader_1,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer_1,
        device=device
    )

    if best_model_1_weights:
        model_1.load_state_dict(best_model_1_weights)
        model_1_path = os.path.join(checkpoint_dir, "model_1.pth")
        torch.save(model_1.state_dict(), model_1_path)
        print(f"✅ Model 1 updated with best weights and saved at: {model_1_path}")

    # === Train model 2 ===
    print(" Starting training for Model 2...")
    best_model_2_weights = train(
        epochs=epochs,
        model=model_2,
        train_dataloader=train_dataloader_2,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer_2,
        device=device
    )

    if best_model_2_weights:
        model_2.load_state_dict(best_model_2_weights)
        model_2_path = os.path.join(checkpoint_dir, "model_2.pth")
        torch.save(model_2.state_dict(), model_2_path)
        print(f"✅ Model 2 updated with best weights and saved at: {model_2_path}")