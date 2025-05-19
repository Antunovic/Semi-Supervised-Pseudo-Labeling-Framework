import os

#Dir with all images and labels
IMAGES_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican512\images"
MASKS_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican512\masks"

#path where data, checkpoints and experimental results will be saved
BASE_PATH = r"proba"
#131 317 457
#seed for data split
SEED = 457

N = 0

INITIAL_TRAINING_PERCENTAGE = 0.1

# Data paths for initial training
TRAIN_IMAGES_DIR_1 = os.path.join(BASE_PATH, "initial_training", "train_1", "images")
TRAIN_MASKS_DIR_1 = os.path.join(BASE_PATH, "initial_training", "train_1", "labels")

TRAIN_IMAGES_DIR_2 = os.path.join(BASE_PATH, "initial_training", "train_2", "images")
TRAIN_MASKS_DIR_2 = os.path.join(BASE_PATH, "initial_training", "train_2", "labels")

# Validation
VAL_IMAGES_DIR = os.path.join(BASE_PATH, "validation", "images")
VAL_MASKS_DIR = os.path.join(BASE_PATH, "validation", "labels")

# Testing
TEST_IMAGES_DIR = os.path.join(BASE_PATH, "test", "images")
TEST_MASKS_DIR = os.path.join(BASE_PATH, "test", "labels")

# Combined labeled data for pseudo training
PSEUDOTRAIN_IMAGES_DIR = os.path.join(BASE_PATH, "pseudo_training", "images")
PSEUDOTRAIN_MASKS_DIR = os.path.join(BASE_PATH, "pseudo_training", "labels")

# Unlabeled data
UNLABELED_IMAGES_DIR = os.path.join(BASE_PATH, "unlabeled", "images")
PSEUDOMASKS_DIR = os.path.join(BASE_PATH, "unlabeled", "pseudolabels")
COMBINED_PSEUDOMASKS_DIR = os.path.join(BASE_PATH, "unlabeled", "combined_pseudolabels")
COMBINED_PSEUDOMASKS_BACKUP = os.path.join(BASE_PATH,"unlabeled")

# Where to store checkpoints
CHECKPOINT_DIR = os.path.join(BASE_PATH,"checkpoints")

MODEL_PATHS = {
    "Model 1 - Initial": os.path.join(CHECKPOINT_DIR,"model_1.pth"),
    "Model 1 - Iter 0": os.path.join(CHECKPOINT_DIR,"model_1_pseudo_iter_0.pth"),
    "Model 1 - Iter 1": os.path.join(CHECKPOINT_DIR,"model_1_pseudo_iter_1.pth"),
    "Model 1 - Iter 2": os.path.join(CHECKPOINT_DIR,"model_1_pseudo_iter_2.pth"),
    "Model 1 - Iter 3": os.path.join(CHECKPOINT_DIR,"model_1_pseudo_iter_3.pth"),
    "Model 1 - Iter 4": os.path.join(CHECKPOINT_DIR,"model_1_pseudo_iter_4.pth"),

    "Model 2 - Initial": os.path.join(CHECKPOINT_DIR,"model_2.pth"),
    "Model 2 - Iter 0": os.path.join(CHECKPOINT_DIR,"model_2_pseudo_iter_0.pth"),
    "Model 2 - Iter 1": os.path.join(CHECKPOINT_DIR,"model_2_pseudo_iter_1.pth"),
    "Model 2 - Iter 2": os.path.join(CHECKPOINT_DIR,"model_2_pseudo_iter_2.pth"),
    "Model 2 - Iter 3": os.path.join(CHECKPOINT_DIR,"model_2_pseudo_iter_3.pth"),
    "Model 2 - Iter 4": os.path.join(CHECKPOINT_DIR,"model_2_pseudo_iter_4.pth"),
    
}

#Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
EPOCHS = 1

# Model settings
ENCODER_NAME = "resnet50"
ENCODER_WEIGHTS = None
IN_CHANNELS = 3
CLASSES = 1

# TRAIN_IMAGES_DIR_1_temp = r"C:\Users\Antonio\Desktop\pseudo_labeling\nebitno\data_small\train_2\images"
# TRAIN_MASKS_DIR_1_temp = r"C:\Users\Antonio\Desktop\pseudo_labeling\nebitno\data_small\train_2\labels"

# VAL_IMAGES_DIR_1_temp = r"C:\Users\Antonio\Desktop\pseudo_labeling\nebitno\data_small\val_1\images"
# VAL_MASKS_DIR_1_temp = r"C:\Users\Antonio\Desktop\pseudo_labeling\nebitno\data_small\val_1\labels"