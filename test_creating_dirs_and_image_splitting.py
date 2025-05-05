from utils import create_directory_structure, split_data
import config


print("Creating direcotry structure...")
create_directory_structure("AA")

print("spliting data...")
split_data(images_dir=config.IMAGES_DIR, masks_dir=config.MASKS_DIR,base_path="AA2",initial_train_percentage=0.05)