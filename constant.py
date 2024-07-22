import torch

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
train_image_dir = './'
train_mask_dir = './'
val_image_dir = './'
val_mask_dir = './'
num_class = 8

# batch size 
batch_size = 4
epoch = 100

