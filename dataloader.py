import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


train_image_dir = './'
train_mask_dir = './'
val_image_dir = './'
val_mask_dir = './'

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        初始化数据集
        Args:
            image_dir (str): 存储图像的目录路径
            mask_dir (str): 存储标签的目录路径
            transform (callable, optional): 对图像和标签进行变换的操作
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 假设标签是单通道灰度图
        
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

# 定义图像和标签的变换
class Resize:
    def __call__(self, image, mask):
        resize_transform = transforms.Resize((224, 224))
        image = resize_transform(image)
        mask = resize_transform(mask)
        return image, mask

class ToTensor:
    def __call__(self, image, mask):
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask

class Normalize:
    def __call__(self, image, mask):
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        return image, mask

transform = transforms.Compose([Resize(), ToTensor(), Normalize()])


def get_dataloader(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, batch_size):
    # 创建训练集和验证集的数据集对象
    train_dataset = SegmentationDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=transform)
    val_dataset = SegmentationDataset(image_dir=val_image_dir, mask_dir=val_mask_dir, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloader(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, batch_size=1)
    print(len(train_loader), len(val_loader))  # 打印数据集的大小
