import os
import shutil
import time
from datetime import datetime
import json
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import convolve
from sklearn.metrics import jaccard_score, f1_score
from tqdm import tqdm
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import UnetPlusPlus

import config

preprocess = A.Compose(
    [
        A.LongestMaxSize(max_size=512),  # Resizes longest side to 512, keeps aspect ratio
        A.PadIfNeeded(
            min_height=512, min_width=512, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=0,  # Pads with black (0)
            p=1.0
        ),
        A.Normalize(mean=(0.4445, 0.3879, 0.3665), std=(0.2817, 0.2424, 0.3274)),
        ToTensorV2(),
    ]
)

# Load the trained model
def load_model(checkpoint_path):
    model = UnetPlusPlus(
        encoder_name=config.ENCODER_NAME,  # Same encoder used during training
        encoder_weights=None,    # Not using pretrained weights
        in_channels=3,
        classes=1                # Binary segmentation
    )
    model.load_state_dict(torch.load(checkpoint_path,weights_only=True))
    model.eval()  # Set to evaluation mode

    return model

def predict(model, image_path, device, threshold=0.5):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)  # Keep for visualization

    # Apply preprocessing
    image = np.array(image)
    transformed = preprocess(image=image)
    input_tensor = transformed["image"].unsqueeze(0)  # Add batch dimension

    # Move to the device
    input_tensor = input_tensor.to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)  # Output logits
        probs = torch.sigmoid(output).squeeze(1)  # Convert logits to probabilities

        #Create 2 channel tensor with predicted class and pixel probabilities for class 1
        predicted = (probs > threshold).float()

        #Combine class and probability information to create pseudolabel
        pseudo_mask= torch.cat((predicted,probs),dim=0)



    return pseudo_mask




def generate_pseudolabels(model_1_path, model_2_path, image_dir, output_dir, encoder_name, in_channels, classes, device="cuda"):
    """
    Generates pseudo-labels for images in the given directory using two models and saves them as CSV files.

    Args:
        model_1_path (str): Path to the weights file for model 1.
        model_2_path (str): Path to the weights file for model 2.
        image_dir (str): Directory containing the images.
        output_dir (str): Directory to save the pseudo-labels.
        encoder_name (str): Name of the encoder for the UNet models (e.g., 'resnet34').
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        classes (int): Number of output classes.
        device (str): Device for inference (e.g., "cuda" or "cpu").

    Returns:
        None
    """
   

    # Load the models
    model_1 = UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)
    model_2 = UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=classes)

    # Load weights into the models
    model_1 = load_model(model_1_path)
    model_2 = load_model(model_2_path)

    # Move models to the specified device
    model_1.to(device)
    model_2.to(device)

    # Set models to evaluation mode
    model_1.eval()
    model_2.eval()

    # Loop through all images in the directory
    for image_name in tqdm(os.listdir(image_dir),desc="Saving pseudolabels"):

        image_path = os.path.join(image_dir, image_name)

        pseudo_mask_1 = predict(model=model_1,image_path=image_path,device=device)
        pseudo_mask_2 = predict(model=model_2,image_path=image_path,device=device)


        # Save the pseudo-labels as CSV
        output_path_1 = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_pseudo1.pt")
        output_path_2 = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_pseudo2.pt")
        torch.save(pseudo_mask_1,output_path_1)
        torch.save(pseudo_mask_2,output_path_2)

        #print(f"Saved pseudo-labels for {image_name}")


def compute_neighborhood_weight(combined_mask, kernel):
    """
    Computes the neighborhood weight based on pixel class.

    Args:
        combined_mask (numpy.ndarray): The combined pseudo-mask of shape [2, H, W].
                                       - Channel 0: Pseudo-mask (classes 0 or 1).
                                       - Channel 1: Placeholder for weights.
        kernel (numpy.ndarray): Kernel for 8-neighborhood counting.

    Returns:
        numpy.ndarray: Updated weights for each pixel based on neighborhood agreement.
    """
    # Extract the pseudo-mask (class predictions)
    class_mask = combined_mask[0]

    # Compute neighborhood agreement for class 0
    neighborhood_0 = convolve((class_mask == 0).astype(np.float32), kernel, mode='constant', cval=0)

    # Compute neighborhood agreement for class 1
    neighborhood_1 = convolve((class_mask == 1).astype(np.float32), kernel, mode='constant', cval=0)

    # Normalize by 8 (number of neighbors)
    neighborhood_0 /= 8.0
    neighborhood_1 /= 8.0

    # Assign neighborhood weights based on the class of each pixel
    neighborhood_weight = np.where(class_mask == 0, neighborhood_0, neighborhood_1)

    return neighborhood_weight


def combine_and_save_pseudolabels(pseudo_labels_path_src, pseudo_labels_path_dst, device="cuda"):
    """
    Combines pseudo-labels from two models and assigns weights based on confidence and agreement.

    Args:
        pseudo_labels_path_src (str): Path to directory containing pseudo-labels.
        pseudo_labels_path_dst (str): Path to save the combined pseudo-labels.
        device (str): Device to store the final tensor (default: "cuda").
    """

    # Get list of pseudo-labels
    pseudo_label_files = [f for f in sorted(os.listdir(pseudo_labels_path_src)) if f.endswith('.pt')]

    # Create dictionary to pair pseudo-labels from both models
    paired_pseudolabels = {}
    for file_name in pseudo_label_files:
        base_name = file_name.split('.')[0][:-1]
        if base_name not in paired_pseudolabels:
            paired_pseudolabels[base_name] = []
        paired_pseudolabels[base_name].append(file_name)

    # Ensure all pseudo-labels come in pairs
    for base_name, files in paired_pseudolabels.items():
        if len(files) != 2:
            raise ValueError(f"Missing pair for {base_name}. Found: {files}")

    # Process each pair of pseudo-labels
    for base_name, files in paired_pseudolabels.items():
        files.sort()

        # Load pseudo-labels from both models
        pl1 = torch.load(os.path.join(pseudo_labels_path_src, files[0]),weights_only=True, map_location="cpu").numpy()
        pl2 = torch.load(os.path.join(pseudo_labels_path_src, files[1]),weights_only=True,map_location="cpu").numpy()

        # Extract predicted class (binary mask) and probabilities (only for class 1)
        class1, probs1 = pl1[0], pl1[1]
        class2, probs2 = pl2[0], pl2[1]

        # Compute probabilities for class 0
        probs1_0 = 1 - probs1
        probs2_0 = 1 - probs2

        # Initialize combined mask and weights
        H, W = class1.shape
        combined_mask = np.zeros((2, H, W), dtype=np.float32)
        weight_mask = np.zeros((H, W), dtype=np.float32)

        # === CASE 1: Agreement Between Models ===
        agreement_mask = class1 == class2

        # Compute confidence as the average probability (for both class 1 and 0)
        confidence_agree_1 = (probs1[agreement_mask] + probs2[agreement_mask]) / 2 
        confidence_agree_0 = (probs1_0[agreement_mask] + probs2_0[agreement_mask]) / 2

        # # Compute weight: (1 - |p1 - p2|) * max(confidence_1, confidence_0)
        weight_mask[agreement_mask] = (1 - np.abs(probs1[agreement_mask] - probs2[agreement_mask])) * np.where(confidence_agree_1 >= confidence_agree_0,
                                                                                                               confidence_agree_1,
                                                                                                               confidence_agree_0)
        
        # weight_mask[agreement_mask] = np.where(confidence_agree_1 >= confidence_agree_0,
        #                                        confidence_agree_1,
        #                                        confidence_agree_0)
        

        # Assign segmentation mask where models agree
        combined_mask[0][agreement_mask] = class1[agreement_mask]

        # === CASE 2: Disagreement Between Models ===
        disagree_mask = ~agreement_mask

        # Use the more confident model for class prediction
        more_confident_model = np.where(probs1[disagree_mask] >= probs2[disagree_mask], 1, 2)
        combined_mask[0][disagree_mask] = np.where(more_confident_model == 1, class1[disagree_mask], class2[disagree_mask])

        # Compute confidence for disagreement (considering both class 1 and 0)
        confidence_disagree_1 = (probs1[disagree_mask] + probs2[disagree_mask]) / 2
        confidence_disagree_0 = (probs1_0[disagree_mask] + probs2_0[disagree_mask]) / 2

        # # Compute weight for disagreement
        weight_mask[disagree_mask] = (1 - np.abs(probs1[disagree_mask] - probs2[disagree_mask])) * np.where(confidence_disagree_1 >= confidence_disagree_0,
                                                                                                            confidence_disagree_1,
                                                                                                            confidence_disagree_0)
        
        # Compute weight for disagreement
        # weight_mask[disagree_mask] = np.where(confidence_disagree_1 >= confidence_disagree_0,
        #                                       confidence_disagree_1,
        #                                       confidence_disagree_0)
        

        # === Neighborhood Weighting ===
        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = 0  # Exclude the center pixel

        #Compute neighborhood weight
        neighborhood_weight = compute_neighborhood_weight(combined_mask, kernel)

        # === Final Weight Calculation ===
        combined_mask[1] = 0.5 * weight_mask + 0.5 * neighborhood_weight

        # Save the final pseudo-label as a .pt file
        combined_pseudolabel = torch.tensor(combined_mask, dtype=torch.float32).to(device)
        torch.save(combined_pseudolabel, os.path.join(pseudo_labels_path_dst, base_name) + ".pt")

    # === Backup Combined Labels ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = os.path.join(os.path.join(config.COMBINED_PSEUDOMASKS_BACKUP, f"backup_combined_iter_{timestamp}"))

    shutil.copytree(pseudo_labels_path_dst, backup_dir)
    print(f"Backup created: {backup_dir}")

    #Delete everything from pseudolabels directory
    print("remove everything from pseudolabels")
    for file in os.listdir(config.PSEUDOMASKS_DIR):
        file_path = os.path.join(config.PSEUDOMASKS_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)



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


def compute_fire_confidence(pseudo_labels_dir):
    """
    Compute average confidence scores for fire pixels in pseudo-labels and rank them.

    Args:
        pseudo_labels_dir (str): Directory containing pseudo-labels (.pt files)

    Returns:
        Pandas DataFrame with filenames and their average fire confidence, sorted by confidence.
    """
    print("COMPUTING FIRE CONFIDENCE SCORES")
    confidence_scores = []

    for file in os.listdir(pseudo_labels_dir):
        if file.endswith(".pt"):
            pseudo_label_path = os.path.join(pseudo_labels_dir, file)
            pseudo_label = torch.load(pseudo_label_path).cpu().numpy()

            binary_mask = pseudo_label[0]  # First channel: fire segmentation mask (0 or 1)
            weights = pseudo_label[1]  # Second channel: confidence scores (0 to 1)

            # Get all confidence values where fire is detected (binary mask == 1)
            fire_pixels = weights[binary_mask == 1]

            if len(fire_pixels) > 0:  # Avoid empty masks
                avg_confidence = np.mean(fire_pixels)
                confidence_scores.append((file, avg_confidence))

    # Convert to DataFrame and sort by confidence in descending order
    df = pd.DataFrame(confidence_scores, columns=["Pseudo-Label", "Avg Fire Confidence"])
    df = df.sort_values(by="Avg Fire Confidence", ascending=False).reset_index(drop=True)

    return df

def copy_images(src_dir, dest_dir):
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)  # Create the destination folder if it doesn't exist

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')  # Define valid image extensions

    for filename in os.listdir(src_dir):
        if filename.lower().endswith(image_extensions):  # Check if it's an image file
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            shutil.copy2(src_path, dest_path)  # Copy with metadata




def create_directory_structure(base_path):
    """
    Creates the required directory structure for pseudo-labeling, including a checkpoints folder.
    """
    # Define main directories
    main_dirs = ["initial_training", "pseudo_training", "test", "validation", "unlabeled", "checkpoints"]
    
    # Define subdirectories for initial training
    initial_subdirs = ["train_1", "train_2"]
    image_label_subdirs = ["images", "labels"]
    
    # Define subdirectories for unlabeled data
    unlabeled_subdirs = ["images", "masks", "pseudolabels", "combined_pseudolabels"]
    
    # Create main directories
    for main_dir in main_dirs:
        main_path = os.path.join(base_path, main_dir)
        os.makedirs(main_path, exist_ok=True)
        
        # Handle specific subdirectories
        if main_dir == "initial_training":
            for sub in initial_subdirs:
                sub_path = os.path.join(main_path, sub)
                os.makedirs(sub_path, exist_ok=True)
                for sub_sub in image_label_subdirs:
                    os.makedirs(os.path.join(sub_path, sub_sub), exist_ok=True)
        
        elif main_dir in ["pseudo_training", "test", "validation"]:
            for sub_sub in image_label_subdirs:
                os.makedirs(os.path.join(main_path, sub_sub), exist_ok=True)
        
        elif main_dir == "unlabeled":
            for sub_sub in unlabeled_subdirs:
                os.makedirs(os.path.join(main_path, sub_sub), exist_ok=True)

    print(f"✅ Directory structure created successfully under: {base_path}")

def create_directory_structure_full_supervision(base_path):
    """
    Creates a simplified directory structure for supervised training, including training, validation,
    test sets, and a checkpoints folder.
    """
    # Define main directories
    main_dirs = ["training", "validation", "test", "checkpoints"]

    # Define subdirectories for image/label organization
    image_label_subdirs = ["images", "labels"]

    # Create main directories and their subdirectories
    for main_dir in main_dirs:
        main_path = os.path.join(base_path, main_dir)
        os.makedirs(main_path, exist_ok=True)
        for sub in image_label_subdirs:
            os.makedirs(os.path.join(main_path, sub), exist_ok=True)

    print(f"✅ Simplified directory structure created successfully under: {base_path}")



def split_data(images_dir, masks_dir, base_path, seed=None, test_split_file=None):
    """
    Splits dataset into fixed test (20%), validation (10%), and training (70%) sets.
    Inside the training set, only a portion defined by config.INITIAL_TRAINING_PERCENTAGE is labeled.
    The rest is used as unlabeled data.

    Parameters:
    - images_dir: Path to all images.
    - masks_dir: Path to all masks.
    - base_path: Output directory.
    - seed: Optional seed for reproducibility.
    - test_split_file: Optional path to fixed test split file.
    """
    if seed is not None:
        random.seed(seed)
        print(f"🔁 Using seed: {seed} for reproducibility.")

    images = sorted(os.listdir(images_dir))
    masks = sorted(os.listdir(masks_dir))
    assert len(images) == len(masks), "Mismatch between images and masks count."

    data = list(zip(images, masks))
    total = len(data)

    # Define test_split_file path if not provided
    if test_split_file is None:
        parent_dir = os.path.dirname(images_dir)
        test_split_file = os.path.join(parent_dir, "fixed_test_split.json")

    # Load or create fixed test split (20%)
    test_count = int(0.20 * total)
    if os.path.exists(test_split_file):
        with open(test_split_file, "r") as f:
            test_image_names = set(json.load(f))
        print(f"📁 Loaded fixed test split from {test_split_file}")
    else:
        random.shuffle(data)
        test_image_names = set([img for img, _ in data[-test_count:]])
        with open(test_split_file, "w") as f:
            json.dump(list(test_image_names), f)
        print(f"💾 Saved fixed test split to {test_split_file}")

    # Separate test and remaining
    test_data = [pair for pair in data if pair[0] in test_image_names]
    remaining_data = [pair for pair in data if pair[0] not in test_image_names]
    random.shuffle(remaining_data)

    # Validation set (10%)
    validation_count = int(0.10 * total)
    validation_data = remaining_data[:validation_count]

    # Training candidates (70%)
    train_data = remaining_data[validation_count:]
    

    # From training set, take INITIAL_TRAINING_PERCENTAGE from total (not from 70%)
    labeled_count = int(config.INITIAL_TRAINING_PERCENTAGE * total)
    assert labeled_count < len(train_data), "Too few training examples for requested label percentage"

    labeled_data = train_data[:labeled_count]
    unlabeled_data = train_data[labeled_count:]

    # Split labeled data into two halves
    train_1_data = labeled_data[:len(labeled_data) // 2]
    train_2_data = labeled_data[len(labeled_data) // 2:]

    # Set config.N globally
    config.N = int(0.2 * len(unlabeled_data))

    def copy_files(data_subset, target_img_dir, target_mask_dir):
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_mask_dir, exist_ok=True)
        for img, mask in data_subset:
            shutil.copy(os.path.join(images_dir, img), os.path.join(target_img_dir, img))
            shutil.copy(os.path.join(masks_dir, mask), os.path.join(target_mask_dir, mask))

    # Copy files
    copy_files(labeled_data, os.path.join(base_path, "pseudo_training", "images"), os.path.join(base_path, "pseudo_training", "labels"))
    copy_files(unlabeled_data, os.path.join(base_path, "unlabeled", "images"), os.path.join(base_path, "unlabeled", "masks"))
    copy_files(validation_data, os.path.join(base_path, "validation", "images"), os.path.join(base_path, "validation", "labels"))
    copy_files(test_data, os.path.join(base_path, "test", "images"), os.path.join(base_path, "test", "labels"))
    copy_files(train_1_data, os.path.join(base_path, "initial_training", "train_1", "images"), os.path.join(base_path, "initial_training", "train_1", "labels"))
    copy_files(train_2_data, os.path.join(base_path, "initial_training", "train_2", "images"), os.path.join(base_path, "initial_training", "train_2", "labels"))

    print("✅ Data successfully split and copied into respective directories.")

def split_data_full_supervision(images_dir, masks_dir, base_path, seed=None, test_split_file=None):
    """
    Splits dataset into fixed test (20%), validation (10%), and training (70%) sets.
    Assumes fully supervised training (no unlabeled subset).
    """

    if seed is not None:
        random.seed(seed)
        print(f"🔁 Using seed: {seed} for reproducibility.")

    images = sorted(os.listdir(images_dir))
    masks = sorted(os.listdir(masks_dir))
    assert len(images) == len(masks), "Mismatch between images and masks count."

    data = list(zip(images, masks))
    total = len(data)

    # Define path for fixed test split
    if test_split_file is None:
        parent_dir = os.path.dirname(images_dir)
        test_split_file = os.path.join(parent_dir, "fixed_test_split.json")

    # Load or create test split
    test_count = int(0.20 * total)
    if os.path.exists(test_split_file):
        with open(test_split_file, "r") as f:
            test_image_names = set(json.load(f))
        print(f"📁 Loaded fixed test split from {test_split_file}")
    else:
        random.shuffle(data)
        test_image_names = set([img for img, _ in data[-test_count:]])
        with open(test_split_file, "w") as f:
            json.dump(list(test_image_names), f)
        print(f"💾 Saved fixed test split to {test_split_file}")

    # Separate test and remaining
    test_data = [pair for pair in data if pair[0] in test_image_names]
    remaining_data = [pair for pair in data if pair[0] not in test_image_names]
    random.shuffle(remaining_data)

    # Validation (10% of total)
    validation_count = int(0.10 * total)
    validation_data = remaining_data[:validation_count]

    # Training = remaining
    train_data = remaining_data[validation_count:]

    def copy_files(data_subset, target_img_dir, target_mask_dir):
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_mask_dir, exist_ok=True)
        for img, mask in data_subset:
            shutil.copy(os.path.join(images_dir, img), os.path.join(target_img_dir, img))
            shutil.copy(os.path.join(masks_dir, mask), os.path.join(target_mask_dir, mask))

    # Copy files to simplified structure
    copy_files(train_data, os.path.join(base_path, "training", "images"), os.path.join(base_path, "training", "labels"))
    copy_files(validation_data, os.path.join(base_path, "validation", "images"), os.path.join(base_path, "validation", "labels"))
    copy_files(test_data, os.path.join(base_path, "test", "images"), os.path.join(base_path, "test", "labels"))

    print("✅ Data successfully split and copied into training, validation, and test folders.")


def move_and_convert_pseudo_labels(df):
    """
    Move the top pseudo-labels and corresponding images to new directories.
    Convert pseudo-labels to binary masks (.npy) and delete the original .pt files.
    """
    import os
    import shutil
    import torch
    import numpy as np

    # Define source and destination directories
    pseudo_labels_src = config.COMBINED_PSEUDOMASKS_DIR
    images_src = config.UNLABELED_IMAGES_DIR

    pseudo_labels_dest = config.PSEUDOTRAIN_MASKS_DIR
    images_dest = config.PSEUDOTRAIN_IMAGES_DIR

    top_n = config.N
    
    # Select top N pseudo-labels based on confidence score
    df = df.sort_values(by="Avg Fire Confidence", ascending=False).head(top_n)

    # Create destination directories
    os.makedirs(pseudo_labels_dest, exist_ok=True)
    os.makedirs(images_dest, exist_ok=True)

    for _, row in df.iterrows():
        pseudo_label_file = os.path.basename(row["Pseudo-Label"])  # Ensure it's just the filename

        pseudo_label_src_path = os.path.join(pseudo_labels_src, pseudo_label_file)
        pseudo_label_dest_path = os.path.join(
            pseudo_labels_dest,
            pseudo_label_file.replace("rgb", "gt").replace("_pseudo.pt", ".npy")
        )

        image_file = pseudo_label_file.replace("_pseudo.pt", ".png")
        image_src_path = os.path.join(images_src, image_file)
        image_dest_path = os.path.join(images_dest, image_file)

        # Process label
        if os.path.exists(pseudo_label_src_path):
            pseudo_label = torch.load(pseudo_label_src_path).cpu().numpy()
            binary_mask = pseudo_label[0]
            np.save(pseudo_label_dest_path, binary_mask)
            os.remove(pseudo_label_src_path)
        else:
            print(f"Pseudo-label not found: {pseudo_label_src_path}")

        # Move image
        if os.path.exists(image_src_path):
            shutil.move(image_src_path, image_dest_path)
        else:
            print(f"Image not found: {image_src_path}")

# === Metrics ===
def compute_metrics(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    iou = jaccard_score(target, pred, average="binary")
    dice = f1_score(target, pred, average="binary")
    accuracy = (pred == target).sum() / len(target)

    return iou, dice, accuracy

# === Evaluate and write to .txt ===
def evaluate(model_paths, dataloader, device, save_path=os.path.join(config.BASE_PATH,"results.txt")):
    results = {model_name: {"iou": 0, "dice": 0, "accuracy": 0} for model_name in model_paths}

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating Models"):
            images, masks = images.to(device), masks.to(device)
            binary_masks = masks[:, 0:1, :, :]  # Shape: (B, 1, H, W)

            for model_name, model_path in model_paths.items():
                model = load_model(model_path).to(device)
                logits = model(images)
                preds = torch.sigmoid(logits) > 0.5
                iou, dice, acc = compute_metrics(preds.cpu().numpy(), binary_masks.cpu().numpy())
                results[model_name]["iou"] += iou
                results[model_name]["dice"] += dice
                results[model_name]["accuracy"] += acc

    for model_name in results:
        results[model_name]["iou"] /= len(dataloader)
        results[model_name]["dice"] /= len(dataloader)
        results[model_name]["accuracy"] /= len(dataloader)

    with open(save_path, "w") as f:
        f.write("=== Model Evaluation Results ===\n\n")
        for model_name, metrics in results.items():
            f.write(f"Model: {model_name}\n")
            f.write(f"  - Mean IoU: {metrics['iou']:.4f}\n")
            f.write(f"  - Mean Dice Score: {metrics['dice']:.4f}\n")
            f.write(f"  - Mean Pixel Accuracy: {metrics['accuracy']:.4f}\n\n")
        f.write("=================================\n")

    print(f"Results saved to: {os.path.abspath(save_path)}")