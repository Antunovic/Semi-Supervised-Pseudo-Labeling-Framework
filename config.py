import os

IMAGES_DIR = r"corsican512\images"
MASKS_DIR = r"corsican512\masks"

BASE_PATH = r"TEST1"

N = 0

INITIAL_TRAINING_PERCENTAGE = 0.05

# Data paths for initial training
TRAIN_IMAGES_DIR_1 = os.path.join(BASE_PATH, "initial_training", "train_1", "images")
TRAIN_MASKS_DIR_1 = os.path.join(BASE_PATH, "initial_training", "train_1", "labels")

TRAIN_IMAGES_DIR_2 = os.path.join(BASE_PATH, "initial_training", "train_2", "images")
TRAIN_MASKS_DIR_2 = os.path.join(BASE_PATH, "initial_training", "train_2", "labels")

# Validation
VAL_IMAGES_DIR = os.path.join(BASE_PATH, "validation", "images")
VAL_MASKS_DIR = os.path.join(BASE_PATH, "validation", "labels")

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