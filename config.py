IMAGES_DIR = r"corsican512\images"
MASKS_DIR = r"corsican512\masks"


#Data paths for initial training
TRAIN_IMAGES_DIR_1 = r"C:\Users\Antonio\Desktop\corsican_flame\images_512"
TRAIN_MASKS_DIR_1 = r"C:\Users\Antonio\Desktop\corsican_flame\labels_512"

TRAIN_IMAGES_DIR_2 = r"C:\Users\Antonio\Desktop\corsican_flame\images_512"
TRAIN_MASKS_DIR_2 = r"C:\Users\Antonio\Desktop\corsican_flame\labels_512"

#Validation
VAL_IMAGES_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican_5proba\validation\images"
VAL_MASKS_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican_5proba\validation\labels"

# #Testing
# TEST_IMAGES_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\data_test2\validation\images"
# TEST_MASKS_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\data_test2\validation\masks"

#Combined labeled data for pseudo training
PSEUDOTRAIN_IMAGES_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican_5proba\pseudo_training\images"
PSEUDOTRAIN_MASKS_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican_5proba\pseudo_training\labels"


UNLABELED_IMAGES_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican_5proba\unlabeled\images"
PSEUDOMASKS_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican_5proba\unlabeled\pseudolabels"
COMBINED_PSEUDOMASKS_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican_5proba\unlabeled\combined_pseudolabels"

BEST_IMAGES_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican_5proba\unlabeled\best_images"
BEST_PSEUDOLABELS_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican_5proba\unlabeled\best_pseudolabels"

#Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
EPOCHS = 30

# Model settings
ENCODER_NAME = "resnet50"
ENCODER_WEIGHTS = None
IN_CHANNELS = 3
CLASSES = 1

#Path to initial model weights for generating pseudo labels
CHECKPOINT_DIR = r"C:\Users\Antonio\Desktop\pseudo_labeling\corsican_5proba"

TRAIN_IMAGES_DIR_1_temp = r"C:\Users\Antonio\Desktop\pseudo_labeling\nebitno\data_small\train_2\images"
TRAIN_MASKS_DIR_1_temp = r"C:\Users\Antonio\Desktop\pseudo_labeling\nebitno\data_small\train_2\labels"

VAL_IMAGES_DIR_1_temp = r"C:\Users\Antonio\Desktop\pseudo_labeling\nebitno\data_small\val_1\images"
VAL_MASKS_DIR_1_temp = r"C:\Users\Antonio\Desktop\pseudo_labeling\nebitno\data_small\val_1\labels"