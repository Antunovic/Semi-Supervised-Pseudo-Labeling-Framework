import os
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import config
from model import get_model

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import multiprocessing
import cv2



class CoriscanDataset (Dataset):

  def  __init__(self, images_dir, masks_dir, transform=None):

    self.images_dir = images_dir
    self.masks_dir = masks_dir
    self.transform = transform

    self.images_names = sorted(os.listdir(images_dir))
    self.masks_names = sorted(os.listdir(masks_dir))

    assert len(self.images_names) == len(self.masks_names), "Number of images and masks do not match."

  def __len__(self):
    return len(self.images_names)
  
#   def __getitem__(self,index):

#     image_path = os.path.join(self.images_dir,self.images_names[index])
#     mask_path = os.path.join(self.masks_dir,self.masks_names[index])

#     try:
#         image = Image.open(image_path).convert("RGB")
#         image = np.array(image)

#     except Exception as e:
#        raise RuntimeError((f"Error loading image: {image_path}"))  from e

#     try:
#         mask = np.loadtxt(mask_path,delimiter=',',dtype=np.float32)
#         # mask = Image.open(mask_path).convert("L")
#         # mask = np.array(mask)
#     except Exception as e:
#        raise RuntimeError((f"Error loading mask: {mask_path}")) from e


#     if self.transform is not None:
#             transformed = self.transform(image=image, mask=mask)
#             image = transformed["image"]
#             mask = transformed["mask"]

#     mask = mask.unsqueeze(0)
#     mask = mask.to(dtype=torch.float32)

#     return image, mask

  def __getitem__(self, index):
        image_path = os.path.join(self.images_dir, self.images_names[index])
        mask_path = os.path.join(self.masks_dir, self.masks_names[index])

        #image loading
        try:
            with Image.open(image_path) as img:
                image = np.array(img.convert("RGB"), dtype=np.uint8)  # Load directly as NumPy array
        except Exception as e:
            raise RuntimeError(f"Error loading image: {image_path}") from e

        try:
            mask = np.load(mask_path, allow_pickle=False)  # Use .npy instead of CSV
        except Exception as e:
            raise RuntimeError(f"Error loading mask: {mask_path}") from e

        #Apply transformations if provided
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # ✅ Convert mask to Tensor (already float32)
        mask = mask.clone().detach().float().unsqueeze(0)  # Add channel dimension

        return image, mask
  

class CombinedDataset(Dataset):
    
    def __init__(self, labeled_images_dir, masks_dir, unlabeled_images_dir, pseudomasks_dir, transform=None):
        
        self.labeled_images_dir = labeled_images_dir
        self.masks_dir = masks_dir
        self.unlabeled_images_dir = unlabeled_images_dir
        self.pseudomasks_dir = pseudomasks_dir
        self.transform = transform

        self.labeled_images_names = sorted(os.listdir(labeled_images_dir))
        self.masks_names = sorted(os.listdir(masks_dir))
        self.unlabeled_images_names = sorted(os.listdir(unlabeled_images_dir))
        self.pseudomasks_names = sorted(os.listdir(pseudomasks_dir))

        assert len(self.labeled_images_names) == len(self.masks_names), "Number of labeled images and masks do not match."
        assert len(self.unlabeled_images_names) == len(self.pseudomasks_names), "Number of unlabeled images and pseudomasks do not match."
        
        print("labeled_size: ",len(self.labeled_images_names))
        print("unlabeled_size:",len(self.unlabeled_images_names))

    def __len__(self):
        return len(self.unlabeled_images_names) + len(self.labeled_images_names)
    
    def __getitem__(self,idx):
        
        if idx < len(self.labeled_images_names):

            image_path = os.path.join(self.labeled_images_dir, self.labeled_images_names[idx])
            mask_path = os.path.join(self.masks_dir, self.masks_names[idx])

            try:
                image = Image.open(image_path).convert("RGB")
                image = np.array(image)

            except Exception as e:
                raise RuntimeError((f"Error loading image: {image_path}"))  from e

            # try:
            #     mask = np.loadtxt(mask_path,delimiter=',',dtype=np.float32)
            # except Exception as e:
            #     raise RuntimeError((f"Error loading mask: {mask_path}")) from e

            try:
                mask = np.load(mask_path, allow_pickle=False)  # Use .npy instead of CSV
            except Exception as e:
                raise RuntimeError(f"Error loading mask: {mask_path}") from e
            
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

            mask = mask.unsqueeze(0)
            
            #Create neutral weights
            weights = torch.ones(size=mask.shape)

            mask = torch.cat((mask,weights),dim=0)


            return image, mask
        
        else:
            
            pseudo_label_idx = idx - len(self.labeled_images_names)
            
            unlabeled_image_path = os.path.join(self.unlabeled_images_dir,self.unlabeled_images_names[pseudo_label_idx])
            pseudomask_path = os.path.join(self.pseudomasks_dir,self.pseudomasks_names[pseudo_label_idx])
            
            try:
                unlabeled_image = Image.open(unlabeled_image_path).convert("RGB")
                unlabeled_image = unlabeled_image.resize((512, 512), resample=Image.BILINEAR)
                unlabeled_image = np.array(unlabeled_image)

            except Exception as e:
                raise RuntimeError((f"Error loading unlabeled-image: {unlabeled_image_path}"))  from e
            
            try:
                pseudomask = torch.load(pseudomask_path,weights_only=True).cpu().numpy()
            except Exception as e:
                raise RuntimeError((f"Error loading pseudomask: {pseudomask_path}")) from e
            
            weights = torch.tensor(pseudomask[1])

            if self.transform:
                transformed = self.transform(image=unlabeled_image, mask=pseudomask)
                unlabeled_image = transformed["image"]
                pseudomask = transformed["mask"]
            
            mask = pseudomask[0]
            
            mask = torch.stack((mask,weights),dim=0)

        # Return image, mask, weights, and a flag for pseudo-labeled data
        return unlabeled_image, mask


def create_pseudo_dataloader(labeled_images_dir, masks_dir, unlabeled_images_dir, pseudomasks_dir, batch_size):
    
    dataset = CombinedDataset(labeled_images_dir=labeled_images_dir,
                     masks_dir=masks_dir,
                     unlabeled_images_dir=unlabeled_images_dir,
                     pseudomasks_dir=pseudomasks_dir,
                     transform=train_transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size,num_workers=4, shuffle=True,pin_memory=True,persistent_workers=True,prefetch_factor=4)

    return dataloader




def create_dataloader(images_dir, masks_dir, batch_size):
   
   dataset = CoriscanDataset(images_dir=images_dir,masks_dir=masks_dir,transform=train_transform)

   dataloader = DataLoader(dataset,batch_size=batch_size,num_workers=4,shuffle=True,pin_memory=True,persistent_workers=True,prefetch_factor=4)

   return dataloader

def vizualize_batch(dataloader):
    
    images, masks = next(iter(dataloader))
    # Handle single image case
    if images.ndim == 3:  # Single image: [3, H, W]
        images = images.unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
    if masks.ndim == 3:  # Single mask: [1, H, W]
        masks = masks.unsqueeze(0)  # Add batch dimension: [1, 1, H, W]

    # Ensure tensors are on CPU and convert to numpy
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to [batch_size, H, W, 3]
    masks = masks.squeeze(1).cpu().numpy()  # Convert to [batch_size, H, W]

    batch_size = images.shape[0]

    # Plot images and masks
    plt.figure(figsize=(16, 4 * batch_size))
    for i in range(batch_size):
        # Show image
        plt.subplot(batch_size, 2, i * 2 + 1)
        plt.imshow(images[i])
        plt.title(f"Image {i+1}")
        plt.axis("off")

        # Show mask
        plt.subplot(batch_size, 2, i * 2 + 2)
        plt.imshow(masks[i], cmap="gray")
        plt.title(f"Mask {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


train_transform = A.Compose(
    [
        # ✅ Resize while keeping aspect ratio, then pad to 512x512
        # A.LongestMaxSize(max_size=512),  # Resizes longest side to 512, keeps aspect ratio
        # A.PadIfNeeded(
        #     min_height=512, min_width=512, 
        #     border_mode=cv2.BORDER_CONSTANT, 
        #     fill=0,  # Pads with black (0)
        #     p=1.0
        # ),

        # ✅ Your existing transformations
        # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),  # Replaced by Affine
        A.Affine(
            scale=(0.8, 1.2),  # Equivalent to scale_limit=0.2
            translate_percent=(0.2, 0.2),  # Equivalent to shift_limit=0.2
            rotate=(-30, 30),  # Equivalent to rotate_limit=30
            p=0.5
        ),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),

        # ✅ Normalize and convert to tensor
        A.Normalize(mean=(0.4445, 0.3879, 0.3665), std=(0.2817, 0.2424, 0.3274)),
        ToTensorV2(),
    ],
    is_check_shapes=False  # Disable shape check
)


def main():
    
    cd = CombinedDataset(labeled_images_dir=r"C:\Users\Antonio\Desktop\pseudo_labeling\data_test\val_1\images",
                     masks_dir=r"C:\Users\Antonio\Desktop\pseudo_labeling\data_test\val_1\labels",
                     unlabeled_images_dir=r"C:\Users\Antonio\Desktop\pseudo_labeling\data_test\unlabeled\images",
                     pseudomasks_dir=r"C:\Users\Antonio\Desktop\pseudo_labeling\data_test\unlabeled\combined_pseudo",
                     transform=train_transform)
    
    print("hello world")

    dataloader = DataLoader(cd, batch_size=8, shuffle=True)
    
if __name__ == "__main__":
    main()
