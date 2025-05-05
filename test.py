# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# from PIL import Image
# from segmentation_models_pytorch import Unet
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import matplotlib.pyplot as plt
# from scipy.ndimage import convolve
# from utils import load_model, predict
# from torchinfo import summary




# model_1_path = r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_1_pseudo_iter_5.pth"
# model_2_path = r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_2_pseudo_iter_5.pth"
# # Load weights into the models
# model_1 = load_model(model_1_path)
# model_2 = load_model(model_2_path)

# model_1 = model_1.to("cuda")
# model_2 = model_2.to("cuda")

# image_path = r"C:\Users\Antonio\Desktop\pseudo_labeling\data\test\images\024_rgb.png"
# image = Image.open(image_path).convert("RGB")
# image = np.array(image)

# mask = predict(model_1,image_path=image_path,device="cuda")[0].cpu().numpy()

# plt.imshow(mask)
# plt.show()

# print("hello world")



from data_setup import CoriscanDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
from tqdm import tqdm
from utils import load_model

# === Define Evaluation Metrics ===
def compute_metrics(pred, target):
    """
    Compute IoU, Dice Score, and Pixel Accuracy.
    """
    pred = pred.flatten()
    target = target.flatten()

    iou = jaccard_score(target, pred, average="binary")
    dice = f1_score(target, pred, average="binary")
    accuracy = (pred == target).sum() / len(target)

    return iou, dice, accuracy


# === Evaluate Models on the Test Dataset ===
def evaluate_models(model_paths, dataloader, device):
    results = {model_name: {"iou": 0, "dice": 0, "accuracy": 0} for model_name in model_paths}

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating Models"):
            images, masks = images.to(device), masks.to(device)

            # Separate binary masks and weights
            binary_masks = masks[:, 0:1, :, :]  # Shape: (B, 1, H, W)

            for model_name, model_path in model_paths.items():
                # Load model
                model = load_model(model_path).to(device)

                # Forward pass
                logits = model(images)
                preds = torch.sigmoid(logits) > 0.5  # Convert to binary predictions

                # Compute metrics
                iou, dice, acc = compute_metrics(preds.cpu().numpy(), binary_masks.cpu().numpy())

                # Accumulate results
                results[model_name]["iou"] += iou
                results[model_name]["dice"] += dice
                results[model_name]["accuracy"] += acc

        # Average results over dataset
        for model_name in results:
            results[model_name]["iou"] /= len(dataloader)
            results[model_name]["dice"] /= len(dataloader)
            results[model_name]["accuracy"] /= len(dataloader)

    return results

# === Plot Evaluation Metrics (6 Graphs) ===
def plot_metrics(results):
    """
    Plots IoU, Dice Score, and Pixel Accuracy separately for Model 1 and Model 2.
    """
    model_1_names = [name for name in results.keys() if "Model 1" in name]
    model_2_names = [name for name in results.keys() if "Model 2" in name]

    ious_model_1 = [results[m]["iou"] for m in model_1_names]
    dices_model_1 = [results[m]["dice"] for m in model_1_names]
    accuracies_model_1 = [results[m]["accuracy"] for m in model_1_names]

    ious_model_2 = [results[m]["iou"] for m in model_2_names]
    dices_model_2 = [results[m]["dice"] for m in model_2_names]
    accuracies_model_2 = [results[m]["accuracy"] for m in model_2_names]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Model 1 Graphs
    axes[0, 0].plot(model_1_names, ious_model_1, marker="o", linestyle="-", color="b", label="IoU")
    axes[0, 0].set_title("Model 1 - Mean IoU")
    axes[0, 0].set_ylabel("IoU Score")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_xticks(range(len(model_1_names)))
    axes[0, 0].set_xticklabels(model_1_names, rotation=45, ha="right")
    axes[0, 0].legend()

    axes[0, 1].plot(model_1_names, dices_model_1, marker="o", linestyle="-", color="g", label="Dice Score")
    axes[0, 1].set_title("Model 1 - Mean Dice Score")
    axes[0, 1].set_ylabel("Dice Score")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_xticks(range(len(model_1_names)))
    axes[0, 1].set_xticklabels(model_1_names, rotation=45, ha="right")
    axes[0, 1].legend()

    axes[0, 2].plot(model_1_names, accuracies_model_1, marker="o", linestyle="-", color="r", label="Pixel Accuracy")
    axes[0, 2].set_title("Model 1 - Mean Pixel Accuracy")
    axes[0, 2].set_ylabel("Pixel Accuracy")
    axes[0, 2].set_xlabel("Iteration")
    axes[0, 2].set_xticks(range(len(model_1_names)))
    axes[0, 2].set_xticklabels(model_1_names, rotation=45, ha="right")
    axes[0, 2].legend()

    # Model 2 Graphs
    axes[1, 0].plot(model_2_names, ious_model_2, marker="o", linestyle="-", color="b", label="IoU")
    axes[1, 0].set_title("Model 2 - Mean IoU")
    axes[1, 0].set_ylabel("IoU Score")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_xticks(range(len(model_2_names)))
    axes[1, 0].set_xticklabels(model_2_names, rotation=45, ha="right")
    axes[1, 0].legend()

    axes[1, 1].plot(model_2_names, dices_model_2, marker="o", linestyle="-", color="g", label="Dice Score")
    axes[1, 1].set_title("Model 2 - Mean Dice Score")
    axes[1, 1].set_ylabel("Dice Score")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_xticks(range(len(model_2_names)))
    axes[1, 1].set_xticklabels(model_2_names, rotation=45, ha="right")
    axes[1, 1].legend()

    axes[1, 2].plot(model_2_names, accuracies_model_2, marker="o", linestyle="-", color="r", label="Pixel Accuracy")
    axes[1, 2].set_title("Model 2 - Mean Pixel Accuracy")
    axes[1, 2].set_ylabel("Pixel Accuracy")
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_xticks(range(len(model_2_names)))
    axes[1, 2].set_xticklabels(model_2_names, rotation=45, ha="right")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()




# === Print Evaluation Metrics ===
def print_metrics(results):
    """
    Prints the evaluation metrics for each model.
    """
    print("\n=== Model Evaluation Results ===")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"  - Mean IoU: {metrics['iou']:.4f}")
        print(f"  - Mean Dice Score: {metrics['dice']:.4f}")
        print(f"  - Mean Pixel Accuracy: {metrics['accuracy']:.4f}")
    print("\n=================================\n")


# === Define paths ===
test_images_dir = r"C:\Users\Antonio\Desktop\pseudo_labeling\data\test\images"
test_masks_dir = r"C:\Users\Antonio\Desktop\pseudo_labeling\data\test\masks"

# === Define Model Checkpoints ===
model_paths = {
    "Model 1 - Initial": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_1_pseudo_iter_4.pth",
    "Model 1 - Iter 0": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_1_pseudo_iter_0.pth",
    "Model 1 - Iter 1": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_1_pseudo_iter_1.pth",
    "Model 1 - Iter 2": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_1_pseudo_iter_2.pth",
    "Model 1 - Iter 3": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_1_pseudo_iter_3.pth",
    "Model 1 - Iter 4": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_1.pth",
    "Model 1 - Iter 5": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_1_pseudo_iter_5.pth",

    "Model 2 - Initial": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_2.pth",
    "Model 2 - Iter 0": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_2_pseudo_iter_0.pth",
    "Model 2 - Iter 1": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_2_pseudo_iter_1.pth",
    "Model 2 - Iter 2": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_2_pseudo_iter_2.pth",
    "Model 2 - Iter 3": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_2_pseudo_iter_3.pth",
    "Model 2 - Iter 4": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_2_pseudo_iter_4.pth",
    "Model 2 - Iter 5": r"C:\Users\Antonio\Desktop\pseudo_labeling\checkpoints\model_2_pseudo_iter_5.pth"
}

# === Define test transforms ===
test_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(mean=(0.4445, 0.3879, 0.3665), std=(0.2817, 0.2424, 0.3274)),
        ToTensorV2(),
    ],
    is_check_shapes=False  # Disable the shape check
)

# === Create Dataset & DataLoader ===
dataset = CoriscanDataset(images_dir=test_images_dir, masks_dir=test_masks_dir, transform=test_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Run Evaluation ===
results = evaluate_models(model_paths, dataloader, device)

print_metrics(results)

plot_metrics(results)
