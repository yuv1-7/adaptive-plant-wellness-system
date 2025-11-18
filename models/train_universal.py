"""
Universal training script for plant disease detection
Supports multiple Kaggle datasets with different structures:
- PlantVillage (38 classes, ~54K images)
- PlantDoc (27 classes, ~2.5K images)
- Chilli Disease (5 classes, ~500 images)
- Rose Disease (3 classes, ~15K images)
- Tulsi/Basil datasets

Automatically detects dataset structure and combines them for training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import os
from collections import defaultdict


class UniversalDiseaseTrainer:
    """
    Universal trainer that handles multiple dataset structures
    """
    
    def __init__(self, learning_rate=0.001, num_epochs=10, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transform (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_dataset_structure(self, dataset_path):
        """
        Automatically detect dataset structure
        Returns: structure_type and split info
        """
        dataset_path = Path(dataset_path)
        
        # Check for common structures
        if (dataset_path / 'train').exists() and (dataset_path / 'val').exists():
            return 'train_val', {'train': 'train', 'val': 'val'}
        
        elif (dataset_path / 'train').exists() and (dataset_path / 'test').exists():
            return 'train_test', {'train': 'train', 'val': 'test'}
        
        elif (dataset_path / 'Train').exists() and (dataset_path / 'Validation').exists():
            return 'train_val_caps', {'train': 'Train', 'val': 'Validation'}
        
        elif (dataset_path / 'color').exists():
            # PlantVillage structure
            return 'plantvillage', {'train': 'color', 'val': None}
        
        else:
            # Single directory with class folders
            return 'single_dir', {'train': '.', 'val': None}
    
    def load_dataset(self, dataset_path, is_train=True):
        """
        Load dataset from path with automatic structure detection
        """
        dataset_path = Path(dataset_path)
        structure, split_info = self.detect_dataset_structure(dataset_path)
        
        print(f"Detected structure: {structure}")
        
        transform = self.train_transform if is_train else self.val_transform
        split_name = split_info['train'] if is_train else split_info['val']
        
        if split_name is None:
            return None
        
        data_dir = dataset_path / split_name if split_name != '.' else dataset_path
        
        try:
            dataset = datasets.ImageFolder(root=data_dir, transform=transform)
            return dataset
        except Exception as e:
            print(f"Error loading dataset from {data_dir}: {e}")
            return None
    
    def combine_datasets(self, dataset_paths):
        """
        Combine multiple datasets into one
        """
        print("\n" + "="*60)
        print("Loading and Combining Datasets")
        print("="*60)
        
        train_datasets = []
        val_datasets = []
        all_classes = set()
        
        for dataset_path in dataset_paths:
            print(f"\nProcessing: {dataset_path}")
            
            train_ds = self.load_dataset(dataset_path, is_train=True)
            val_ds = self.load_dataset(dataset_path, is_train=False)
            
            if train_ds:
                train_datasets.append(train_ds)
                all_classes.update(train_ds.classes)
                print(f"  Train: {len(train_ds)} images, {len(train_ds.classes)} classes")
            
            if val_ds:
                val_datasets.append(val_ds)
                print(f"  Val: {len(val_ds)} images, {len(val_ds.classes)} classes")
        
        # Create unified class mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
        
        # Remap class indices for all datasets
        for ds in train_datasets + val_datasets:
            old_to_new = {old_idx: class_to_idx[cls] 
                         for cls, old_idx in ds.class_to_idx.items()}
            
            # Update targets
            for i in range(len(ds)):
                old_idx = ds.targets[i]
                old_class = ds.classes[old_idx]
                ds.targets[i] = class_to_idx[old_class]
            
            ds.class_to_idx = class_to_idx
            ds.classes = sorted(all_classes)
        
        # Combine datasets
        combined_train = ConcatDataset(train_datasets) if train_datasets else None
        combined_val = ConcatDataset(val_datasets) if val_datasets else None
        
        print("\n" + "="*60)
        print(f"Combined Dataset Summary")
        print("="*60)
        print(f"Total classes: {len(all_classes)}")
        if combined_train:
            print(f"Total training images: {len(combined_train)}")
        if combined_val:
            print(f"Total validation images: {len(combined_val)}")
        print("="*60 + "\n")
        
        return combined_train, combined_val, class_to_idx
    
    def create_model(self, num_classes):
        """
        Create ResNet50 model
        """
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze early layers for faster training
        for param in list(model.parameters())[:-30]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        return model.to(self.device)
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                            'acc': f'{100*correct/total:.2f}%'})
        
        return running_loss / len(train_loader), 100 * correct / total
    
    def validate(self, model, val_loader, criterion):
        """Validate the model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                'acc': f'{100*correct/total:.2f}%'})
        
        return running_loss / len(val_loader), 100 * correct / total
    
    def train(self, dataset_paths, save_dir='models'):
        """
        Full training pipeline
        """
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and combine datasets
        train_dataset, val_dataset, class_to_idx = self.combine_datasets(dataset_paths)
        
        if not train_dataset:
            raise ValueError("No training data found!")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Create model
        num_classes = len(class_to_idx)
        model = self.create_model(num_classes)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        # Training loop
        best_val_acc = 0.0
        training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"Validation samples: {len(val_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print("="*60 + "\n")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer
            )
            
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
            # Validate
            if val_loader:
                val_loss, val_acc = self.validate(model, val_loader, criterion)
                training_history['val_loss'].append(val_loss)
                training_history['val_acc'].append(val_acc)
                
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                
                # Learning rate scheduling
                scheduler.step(val_acc)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), 
                             save_dir / 'disease_model_best.pth')
                    print(f"✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        
        # Save final model
        torch.save(model.state_dict(), save_dir / 'disease_model_final.pth')
        
        # Save class mapping
        with open(save_dir / 'class_mapping.json', 'w') as f:
            json.dump(class_to_idx, f, indent=2)
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Models saved to: {save_dir}")
        print("="*60)
        
        return model, class_to_idx, training_history


def main():
    parser = argparse.ArgumentParser(
        description='Universal plant disease detection trainer'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='Paths to dataset directories (can specify multiple)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='models',
        help='Directory to save models (default: models)'
    )
    
    args = parser.parse_args()
    
    # Verify datasets exist
    for dataset_path in args.datasets:
        if not Path(dataset_path).exists():
            print(f"Error: Dataset path does not exist: {dataset_path}")
            return
    
    # Create trainer
    trainer = UniversalDiseaseTrainer(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Train
    model, class_mapping, history = trainer.train(
        dataset_paths=args.datasets,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()