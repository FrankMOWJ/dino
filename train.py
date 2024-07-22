import torch
from dataloader import get_dataloader
from metrics import ComputeIoU
from surgicaldino import SurgicalDINO
from constant import *
from torch import optim
import torch.nn.functional as F
from seg_decoder_head import compute_segmentation_loss
import logging

# logger 
logger = logging.getLogger(__name__)

# Instantiate Surgical-DINO
model = SurgicalDINO(backbone_size="base", r=4, lora_layer=None, image_shape=(224,224), num_class=num_class, decode_type = 'linear4').to(device = DEVICE)
# data
train_loader, val_loader = get_dataloader(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, batch_size)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, verbose=True)
# criterion
criterion = compute_segmentation_loss
# mIOU
compute_iou = ComputeIoU(num_class=num_class)

def evaluate_miou(model, dataloader, num_classes, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            compute_iou(preds, targets) # 这时会不断累加混淆矩阵的值
    ious = compute_iou.get_ious()
    miou = compute_iou.get_miou(ignore=None)
    cfsmatrix = compute_iou.get_cfsmatrix()

    logger.info(f'ious per class:\n{ious}')
    logger.info(f'miou: {miou}')
    logger.info(f'confusion matrix:\n{cfsmatrix}')

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        evaluate_miou(model, val_loader, num_class, device)
        scheduler.step()
        
    
    logger.info('Finished Training')

if __name__ == "__main__":
    train(model, train_loader, val_loader, criterion, optimizer, epoch, DEVICE)

    # 测试模型
    evaluate_miou(model, val_loader, num_class, DEVICE)

    # TODO: save checkpoint
