import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import PIL.Image as Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import build_resnet18
from evaluation import evaluate
from testloader import GTSRB_Test_Loader
from train import GTSRB_Train_Loader, get_transforms, train_one_epoch, validate, plot_training_history

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(118)
    np.random.seed(118)
    
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    TRAIN_IMG_DIR = os.path.join(script_dir, 'GTSRB/Final_Training/Images')
    TEST_IMG_DIR = os.path.join(script_dir, 'GTSRB/Final_Test/Images')
    TEST_GT_CSV = os.path.join(script_dir, 'evaluation/GTSRB_Test_GT.csv')
    CHECKPOINTS_DIR = os.path.join(script_dir, 'checkpoints')
    
    # Create checkpoints directory
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data transforms
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Load training data
    print("Loading training dataset...")
    full_trainset = GTSRB_Train_Loader(
        data_dir=TRAIN_IMG_DIR,
        transform=train_transform
    )
    
    # Split training data into train and validation
    train_indices, val_indices = train_test_split(
        range(len(full_trainset)), 
        test_size=0.2, 
        random_state=118,
        stratify=full_trainset.labels
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_trainset, train_indices)
    
    # Create validation dataset with different transforms
    val_trainset = GTSRB_Train_Loader(
        data_dir=TRAIN_IMG_DIR,
        transform=val_transform
    )
    val_dataset = torch.utils.data.Subset(val_trainset, train_indices)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Load test dataset
    print("Loading test dataset...")
    testset = GTSRB_Test_Loader(
        TEST_PATH=TEST_IMG_DIR,
        TEST_GT_PATH=TEST_GT_CSV
    )
    test_loader = DataLoader(
        testset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(testset)}")
    
    # Model
    model = build_resnet18(num_classes=43, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_path = os.path.join(CHECKPOINTS_DIR, 'resnet18_best.pt')
    
    print("Starting training...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print learning rate change if it happened
        if new_lr != current_lr:
            print(f'Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}')
        
        # Record history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
            }, best_model_path)
            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'resnet18_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, checkpoint_path)
    
    # Plot training history
    plot_path = os.path.join(script_dir, 'training_history.png')
    plot_training_history(train_losses, train_accs, val_losses, val_accs, plot_path)
    print(f"Training history plot saved to: {plot_path}")
    
    # Load best model and evaluate on test set
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Final test evaluation
    with torch.no_grad():
        test_acc = evaluate(model, test_loader)
    
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.3f}%")
    print(f"Final test accuracy: {test_acc:.3f}")
    print(f"Best model saved at: {best_model_path}")

if __name__ == '__main__':
    main()