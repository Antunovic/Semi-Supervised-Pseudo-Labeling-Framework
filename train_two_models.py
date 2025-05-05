import torch
from tqdm import tqdm
from train import train, validate, create_dataloader  # Replace with your utility functions
import config
from model import get_model

# Check if multiple GPUs are available
if torch.cuda.device_count() < 2:
    raise RuntimeError("At least two GPUs are required for this script.")

# Assign each model to a separate GPU
device_1 = torch.device("cuda:0")
device_2 = torch.device("cuda:1")

# Create DataLoaders
train_dataloader = create_dataloader(images_dir=config.TRAIN_IMAGES_DIR,
                                      masks_dir=config.TRAIN_MASKS_DIR,
                                      batch_size=config.BATCH_SIZE)

val_dataloader = create_dataloader(images_dir=config.VAL_IMAGES_DIR,
                                    masks_dir=config.VAL_MASKS_DIR,
                                    batch_size=config.BATCH_SIZE)

# Initialize models
model_1, model_2, initial_weights_1, initial_weights_2 = get_model()

# Move each model to its respective GPU
model_1.to(device_1)
model_2.to(device_2)

# Loss function and optimizers
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer_1 = torch.optim.Adam(lr=config.LEARNING_RATE, params=model_1.parameters())
optimizer_2 = torch.optim.Adam(lr=config.LEARNING_RATE, params=model_2.parameters())

# Number of epochs
epochs = config.EPOCHS

# Training loop
for epoch in tqdm(range(epochs), desc="Training Progress"):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # Train Model 1 on GPU 0
    train_loss_1, train_miou_1 = train(model_1, train_dataloader, loss_fn, optimizer_1, device_1)
    print(f"Model 1 - Train Loss: {train_loss_1:.4f}, Train mIoU: {train_miou_1:.4f}")

    # Validate Model 1 on GPU 0
    val_loss_1, val_miou_1 = validate(model_1, val_dataloader, loss_fn, device_1)
    print(f"Model 1 - Validation Loss: {val_loss_1:.4f}, Validation mIoU: {val_miou_1:.4f}")

    # Train Model 2 on GPU 1
    train_loss_2, train_miou_2 = train(model_2, train_dataloader, loss_fn, optimizer_2, device_2)
    print(f"Model 2 - Train Loss: {train_loss_2:.4f}, Train mIoU: {train_miou_2:.4f}")

    # Validate Model 2 on GPU 1
    val_loss_2, val_miou_2 = validate(model_2, val_dataloader, loss_fn, device_2)
    print(f"Model 2 - Validation Loss: {val_loss_2:.4f}, Validation mIoU: {val_miou_2:.4f}")

# Save both models
torch.save(model_1.state_dict(), "model_1.pth")
torch.save(model_2.state_dict(), "model_2.pth")

print("Training complete. Models saved.")