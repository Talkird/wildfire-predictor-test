import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import kagglehub
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== CONFIGURATION ====================
CONFIG = {
    'data_dir': 'wildfire_data',
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'image_size': 350, 
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'model_save_path': 'wildfire_model.pth'
}

# ==================== DOWNLOAD DATASET ====================
def download_dataset():
    """Download wildfire dataset from Kaggle Hub"""
    print("Downloading wildfire dataset...")
    path = kagglehub.dataset_download("abdelghaniaaba/wildfire-prediction-dataset")
    print(f"Dataset downloaded to: {path}")
    return path

# ==================== DATASET CLASS ====================
class WildfireDataset(Dataset):
    """Custom dataset for wildfire images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size']))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ==================== CNN MODEL ====================
class WildfireCNN(nn.Module):
    """Convolutional Neural Network for wildfire detection"""
    
    def __init__(self, num_classes=2):
        super(WildfireCNN, self).__init__()
        
        # Feature extraction blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==================== TRAINING UTILITIES ====================
def load_dataset(data_path, split='train'):
    """
    Load dataset from the Kaggle wildfire dataset structure.
    Expected structure:
        data_path/
        ├── train/
        │   ├── wildfire/
        │   └── nowildfire/
        ├── test/
        │   ├── wildfire/
        │   └── nowildfire/
        └── valid/
            ├── wildfire/
            └── nowildfire/
    """
    image_paths = []
    labels = []
    
    classes = {'nowildfire': 0, 'wildfire': 1}
    split_dir = os.path.join(data_path, split)
    
    if not os.path.exists(split_dir):
        raise ValueError(f"Split directory not found: {split_dir}")
    
    for class_name, class_idx in classes.items():
        class_dir = os.path.join(split_dir, class_name)
        if os.path.exists(class_dir):
            file_count = 0
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(class_idx)
                    file_count += 1
            print(f"  {class_name}: {file_count} images")
    
    if not image_paths:
        raise ValueError(f"No images found in {split_dir}")
    
    return image_paths, labels

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, val_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(val_loader), 100. * correct / total

def export_to_onnx(model, device, input_size=350):
    """
    Export model to ONNX format for production deployment.
    ONNX enables inference on any platform without PyTorch dependency.
    """
    print("\n" + "="*50)
    print("Exporting model to ONNX format...")
    print("="*50)
    
    try:
        model.eval()
        onnx_path = 'wildfire_model.onnx'
        
        # Create dummy input matching the model's expected input
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=14,
            input_names=['input_image'],
            output_names=['predictions'],
            dynamic_axes={
                'input_image': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            },
            do_constant_folding=True,
            verbose=False
        )
        
        print(f"✓ Model exported to {onnx_path}")
        print(f"  - Input: RGB image ({input_size}x{input_size}px)")
        print(f"  - Output: 2 class probabilities (wildfire, no_wildfire)")
        print(f"  - Benefits: Cross-platform deployment, ~20-40% inference speedup")
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        print("  Install onnx: pip install onnx onnxruntime")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Complete training loop"""
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if len(val_accs) == 1 or val_acc > max(val_accs[:-1]):
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            print(f"Model saved with val accuracy: {val_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs

# ==================== MAIN ====================
def main():
    """Main execution function"""
    # Create device
    device = CONFIG['device']
    print(f"Using device: {device}")
    
    # Download dataset
    data_path = download_dataset()
    
    # Load dataset splits (using pre-existing train/val/test split from Kaggle)
    print("\nLoading training dataset...")
    train_paths, train_labels = load_dataset(data_path, split='train')
    
    print("\nLoading validation dataset...")
    val_paths, val_labels = load_dataset(data_path, split='valid')
    
    print("\nLoading test dataset...")
    test_paths, test_labels = load_dataset(data_path, split='test')
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    print(f"  Test samples: {len(test_paths)}")
    print(f"  Total: {len(train_paths) + len(val_paths) + len(test_paths)}")
    
    # Data augmentation and normalization
    # Normalization uses ImageNet statistics (common for transfer learning)
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and loaders
    train_dataset = WildfireDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = WildfireDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = WildfireDataset(test_paths, test_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=4)
    
    # Initialize model
    print("\nInitializing model...")
    model = WildfireCNN(num_classes=2).to(device)
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Train model
    print("\nStarting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        CONFIG['num_epochs'], device
    )
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating on Test Set...")
    print("="*50)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("\nTraining results saved to training_results.png")
    
    # Load best model and export to ONNX
    model.load_state_dict(torch.load(CONFIG['model_save_path']))
    print(f"\nBest model loaded from {CONFIG['model_save_path']}")
    
    # Export to ONNX for production deployment
    export_to_onnx(model, device)

if __name__ == "__main__":
    main()
