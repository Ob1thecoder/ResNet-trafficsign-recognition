# -*- coding: utf-8 -*-
"""
Training script for ResNet-18 on GTSRB traffic sign dataset
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd
import PIL.Image as Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from model import build_resnet18
from evaluation import evaluate


class GTSRB_Train_Loader(Dataset):
    """
    Training data loader for GTSRB dataset
    """
    def __init__(self, data_dir, csv_file=None, transform=None, is_validation=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_validation = is_validation
        
        # Collect all training images and labels
        self.images = []
        self.labels = []
        
        # Read from all class directories
        for class_id in range(43):  # GTSRB has 43 classes
            class_dir = os.path.join(data_dir, f'{class_id:05d}')
            if os.path.exists(class_dir):
                gt_file = os.path.join(class_dir, f'GT-{class_id:05d}.csv')
                if os.path.exists(gt_file):
                    df = pd.read_csv(gt_file, sep=';')
                    for _, row in df.iterrows():
                        img_path = os.path.join(class_dir, row['Filename'])
                        if os.path.exists(img_path):
                            self.images.append(img_path)
                            self.labels.append(class_id)
        
        print(f"Loaded {len(self.images)} training images")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = torch.zeros(3, 48, 48)
            return dummy_image, label

def get_transforms(is_training=True):
    """Get data transforms for training and validation"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

