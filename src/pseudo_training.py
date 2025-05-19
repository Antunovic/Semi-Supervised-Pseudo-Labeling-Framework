import torch
import config
from data_setup import create_dataloader
from metrics import combined_bce_dice_loss

import train
import os


def pseudo_training(model_1, model_2, device, checkpoint_dir, i):

    print("Creating DataLoaders for pseudo training...")

    train_dataloader = create_dataloader(images_dir=config.PSEUDOTRAIN_IMAGES_DIR,
                                         masks_dir=config.PSEUDOTRAIN_MASKS_DIR,
                                         batch_size=config.BATCH_SIZE)
    
    val_dataloader = create_dataloader(images_dir=config.VAL_IMAGES_DIR,
                                       masks_dir=config.VAL_MASKS_DIR,
                                       batch_size=config.BATCH_SIZE)

    
    
    #loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = combined_bce_dice_loss

    optimizer_1 = torch.optim.Adam(lr=config.LEARNING_RATE,params=model_1.parameters())
    optimizer_2 = torch.optim.Adam(lr=config.LEARNING_RATE,params=model_2.parameters())

    epochs = config.EPOCHS

    print("Starting pseudo-training for model 1")
    print("Starting training for model 1")
    best_model_1_weights = train.train(epochs=epochs,
                                        model=model_1,
                                        train_dataloader=train_dataloader,
                                        val_dataloader=val_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer_1,
                                        device=device)
    
    print("Starting training for model 2")
    best_model_2_weights = train.train(epochs=epochs,
                                        model=model_2,
                                        train_dataloader=train_dataloader,
                                        val_dataloader=val_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer_2,
                                        device=device)

# === Load best model weights back into the models ===
    if best_model_1_weights:
        model_1.load_state_dict(best_model_1_weights)
        print("✅ Model 1 updated with best weights.")

    if best_model_2_weights:
        model_2.load_state_dict(best_model_2_weights)
        print("✅ Model 2 updated with best weights.")

    # === Save best models ===
    model_1_path = os.path.join(checkpoint_dir, f"model_1_pseudo_iter_{i}.pth")
    model_2_path = os.path.join(checkpoint_dir, f"model_2_pseudo_iter_{i}.pth")

    torch.save(model_1.state_dict(), model_1_path)
    torch.save(model_2.state_dict(), model_2_path)

    print(f"✅ Best models saved at {model_1_path} and {model_2_path}ss")