"""
Football Frame Classifier - CNN Model

Classifies football video frames into:
- Valid (overlook/wide-angle field view)
- Invalid (zoomed-in players, fans, celebrations, closeups)

Uses transfer learning with pretrained models for better accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
import json
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import tempfile
import shutil
import threading
from queue import Queue


class FrameClassifierCNN(nn.Module):
    """
    CNN model for classifying football frames.
    Uses pretrained ResNet18 backbone with custom classifier head.
    
    Classes:
        0: Invalid (closeup, fans, players, celebration)
        1: Valid (overlook/wide-angle field view)
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, 
                 backbone: str = 'resnet18', dropout: float = 0.5):
        super(FrameClassifierCNN, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights using Kaiming initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Flatten features for classifier (handles [batch, C, 1, 1] -> [batch, C])
        # Use flatten instead of view for ONNX dynamic batch support
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        return output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def freeze_backbone(self):
        """Freeze backbone weights for fine-tuning only the classifier."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights for full training."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class FootballFrameDataset(Dataset):
    """
    Dataset for football frame classification.
    
    Directory structure:
        data_dir/
            valid/      # Overlook/wide-angle views
                frame1.jpg
                frame2.jpg
                ...
            invalid/    # Closeups, fans, players
                frame1.jpg
                frame2.jpg
                ...
    """
    
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None,
                 image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.image_extensions = image_extensions
        
        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx = {'invalid': 0, 'valid': 1}
        self.idx_to_class = {0: 'invalid', 1: 'valid'}
        
        self._load_samples()
    
    def _load_samples(self):
        """Load all image samples from directory."""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.image_extensions:
                    self.samples.append((str(img_path), class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.data_dir}. "
                           f"Expected subdirectories: 'valid', 'invalid'")
        
        print(f"Loaded {len(self.samples)} samples from {self.data_dir}")
        self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print class distribution."""
        class_counts = {name: 0 for name in self.class_to_idx.keys()}
        for _, class_idx in self.samples:
            class_name = self.idx_to_class[class_idx]
            class_counts[class_name] += 1
        
        print("Class distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} ({count/len(self.samples)*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class VideoFrameDataset(Dataset):
    """
    Dataset for loading frames directly from a video file.
    Useful for inference on entire videos.
    """
    
    def __init__(self, video_path: str, transform: Optional[transforms.Compose] = None,
                 frame_skip: int = 1, max_frames: Optional[int] = None):
        self.video_path = video_path
        self.transform = transform
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate frame indices to process
        self.frame_indices = list(range(0, self.total_frames, frame_skip))
        if max_frames:
            self.frame_indices = self.frame_indices[:max_frames]
    
    def __len__(self) -> int:
        return len(self.frame_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        frame_idx = self.frame_indices[idx]
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Return blank frame on error
            image = Image.new('RGB', (224, 224), color='black')
        else:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        
        if self.transform:
            image = self.transform(image)
        
        return image, frame_idx


def get_train_transforms(image_size: int = 256) -> transforms.Compose:
    """
    Get training data augmentation transforms.
    
    Default image_size=256 for 720p (720x1280) video frames.
    Use 224 for faster training, 384 for higher accuracy.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size: int = 256) -> transforms.Compose:
    """
    Get validation/inference transforms.
    
    Default image_size=256 for 720p (720x1280) video frames.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class FrameClassifierTrainer:
    """
    Trainer class for the frame classifier model.
    Handles training, validation, checkpointing, and early stopping.
    """
    
    def __init__(self, model: FrameClassifierCNN, device: Optional[str] = None,
                 checkpoint_dir: str = 'checkpoints'):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []
        
        print(f"Trainer initialized on device: {self.device}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 50, lr: float = 1e-4, weight_decay: float = 1e-4,
              patience: int = 10, min_delta: float = 0.001,
              freeze_backbone_epochs: int = 5,
              class_weights: Optional[torch.Tensor] = None) -> Dict:
        """
        Train the model with early stopping and learning rate scheduling.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            lr: Learning rate
            weight_decay: L2 regularization weight
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            freeze_backbone_epochs: Number of epochs to train with frozen backbone
            class_weights: Optional class weights for imbalanced data
            
        Returns:
            Training history dictionary
        """
        # Loss function with optional class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience // 2
        )
        
        # Early stopping
        best_val_loss = float('inf')
        best_val_acc = 0.0
        epochs_without_improvement = 0
        
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Patience: {patience}")
        print(f"  Freeze backbone epochs: {freeze_backbone_epochs}")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        # Phase 1: Train with frozen backbone
        if freeze_backbone_epochs > 0:
            print("Phase 1: Training classifier with frozen backbone...")
            self.model.freeze_backbone()
        
        for epoch in range(epochs):
            # Unfreeze backbone after initial epochs
            if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
                print("\nPhase 2: Unfreezing backbone for full training...")
                self.model.unfreeze_backbone()
                # Reduce learning rate for backbone fine-tuning
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr / 10
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e}")
            
            # Check for improvement
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_val_acc = val_acc
                epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint('best_model.pth', epoch, val_loss, val_acc)
                print(f"  -> New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_loss, val_acc)
        
        # Save final model
        self.save_checkpoint('final_model.pth', epoch, val_loss, val_acc)
        
        # Load best model
        self.load_checkpoint('best_model.pth')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'epochs_trained': epoch + 1
        }
    
    def _train_epoch(self, loader: DataLoader, criterion: nn.Module,
                     optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(loader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _validate_epoch(self, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'backbone': self.model.backbone_name,
            'num_classes': self.model.num_classes,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        
        print(f"Loaded checkpoint: {filename}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"  Val Acc: {checkpoint['val_acc']:.2f}%")


class VideoFrameClassifier:
    """
    High-level class for classifying frames in videos.
    Handles model loading, batch inference, and video processing.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None,
                 backbone: str = 'resnet18', threshold: float = 0.5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        
        # Initialize model
        self.model = FrameClassifierCNN(num_classes=2, pretrained=True, backbone=backbone)
        self.model.to(self.device)
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.model.eval()
        
        # Inference transforms
        self.transform = get_val_transforms()
        
        print(f"VideoFrameClassifier initialized on {self.device}")
    
    def load_model(self, model_path: str):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Model loaded from {model_path}")
    
    def classify_frame(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Classify a single frame.
        
        Args:
            frame: BGR frame from OpenCV
            
        Returns:
            (is_valid, confidence) tuple
        """
        # Convert BGR to RGB and to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Transform and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            valid_prob = probs[0, 1].item()  # Probability of "valid" class
        
        is_valid = valid_prob >= self.threshold
        return is_valid, valid_prob
    
    def classify_frames_batch(self, frames: List[np.ndarray], 
                              batch_size: int = 32) -> List[Tuple[bool, float]]:
        """
        Classify multiple frames in batches.
        
        Args:
            frames: List of BGR frames
            batch_size: Batch size for inference
            
        Returns:
            List of (is_valid, confidence) tuples
        """
        results = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Prepare batch
            batch_tensors = []
            for frame in batch_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                tensor = self.transform(image)
                batch_tensors.append(tensor)
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                valid_probs = probs[:, 1].cpu().numpy()
            
            for prob in valid_probs:
                is_valid = prob >= self.threshold
                results.append((is_valid, float(prob)))
        
        return results
    
    def process_video(self, video_path: str, output_path: str,
                      batch_size: int = 32, show_progress: bool = True) -> Dict:
        """
        Process a video and output only valid frames.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            batch_size: Batch size for inference
            show_progress: Show progress bar
            
        Returns:
            Processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        valid_count = 0
        frame_idx = 0
        batch_frames = []
        batch_indices = []
        
        pbar = tqdm(total=total_frames, desc="Processing video", disable=not show_progress)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            
            # Process batch
            if len(batch_frames) >= batch_size:
                results = self.classify_frames_batch(batch_frames, batch_size)
                
                for j, (is_valid, _) in enumerate(results):
                    if is_valid:
                        out.write(batch_frames[j])
                        valid_count += 1
                
                pbar.update(len(batch_frames))
                batch_frames = []
                batch_indices = []
            
            frame_idx += 1
        
        # Process remaining frames
        if batch_frames:
            results = self.classify_frames_batch(batch_frames, batch_size)
            for j, (is_valid, _) in enumerate(results):
                if is_valid:
                    out.write(batch_frames[j])
                    valid_count += 1
            pbar.update(len(batch_frames))
        
        pbar.close()
        cap.release()
        out.release()
        
        stats = {
            'total_frames': total_frames,
            'valid_frames': valid_count,
            'invalid_frames': total_frames - valid_count,
            'selection_ratio': valid_count / max(1, total_frames),
            'input_path': video_path,
            'output_path': output_path
        }
        
        print(f"\nVideo processing complete:")
        print(f"  Total frames: {total_frames}")
        print(f"  Valid frames: {valid_count} ({stats['selection_ratio']*100:.1f}%)")
        print(f"  Output saved to: {output_path}")
        
        return stats


def extract_frames_from_video(video_path: str, output_dir: str,
                              frame_skip: int = 30, max_frames: Optional[int] = None) -> int:
    """
    Extract frames from a video for manual labeling.
    
    Args:
        video_path: Input video path
        output_dir: Output directory for frames
        frame_skip: Extract every N-th frame
        max_frames: Maximum number of frames to extract
        
    Returns:
        Number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            filename = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            extracted += 1
            
            if max_frames and extracted >= max_frames:
                break
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {extracted} frames to {output_dir}")
    return extracted


def compute_class_weights(dataset: FootballFrameDataset) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    class_counts = [0, 0]
    for _, label in dataset.samples:
        class_counts[label] += 1
    
    total = sum(class_counts)
    weights = [total / (2 * count) if count > 0 else 1.0 for count in class_counts]
    
    return torch.tensor(weights, dtype=torch.float32)


def train_classifier(data_dir: str, checkpoint_dir: str = 'checkpoints',
                     backbone: str = 'resnet18', epochs: int = 50,
                     batch_size: int = 32, lr: float = 1e-4,
                     val_split: float = 0.2, num_workers: int = 4) -> Dict:
    """
    Train the frame classifier model.
    
    Args:
        data_dir: Directory containing 'valid' and 'invalid' subdirectories
        checkpoint_dir: Directory for saving checkpoints
        backbone: Model backbone ('resnet18', 'resnet34', 'resnet50', 'mobilenet_v2')
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Validation split ratio
        num_workers: Number of data loading workers
        
    Returns:
        Training history
    """
    # Create datasets
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    full_dataset = FootballFrameDataset(data_dir, transform=train_transform)
    
    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transforms to validation set
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    # Compute class weights
    class_weights = compute_class_weights(full_dataset)
    print(f"Class weights: {class_weights.tolist()}")
    
    # Create model and trainer
    model = FrameClassifierCNN(num_classes=2, pretrained=True, backbone=backbone)
    trainer = FrameClassifierTrainer(model, checkpoint_dir=checkpoint_dir)
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        class_weights=class_weights
    )
    
    return history


class ParallelVideoProcessor:
    """
    High-performance video processor maximizing CPU utilization:
    - Multi-threaded preprocessing (ThreadPoolExecutor)
    - OpenCV preprocessing (faster than PIL)
    - TorchScript JIT compilation
    - PyTorch multi-threaded inference
    - Large batch inference
    """
    
    def __init__(self, model_path: str, backbone: str = 'resnet18',
                 threshold: float = 0.5, batch_size: int = 256,
                 image_size: int = 224, num_threads: int = 64):
        """
        Initialize processor.
        
        Args:
            model_path: Path to trained model checkpoint
            backbone: Model backbone (must match training)
            threshold: Classification threshold
            batch_size: Batch size for inference (larger=faster)
            image_size: Input image size
            num_threads: Threads for preprocessing (0=auto)
        """
        self.model_path = model_path
        self.backbone = backbone
        self.threshold = threshold
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Auto-detect thread count
        if num_threads <= 0:
            num_threads = 64
        self.num_threads = num_threads
        
        # Set PyTorch to use all CPU cores
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize model once
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = FrameClassifierCNN(num_classes=2, pretrained=False, backbone=backbone)
        self.model.to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # JIT compile for faster inference
        try:
            dummy_input = torch.randn(1, 3, image_size, image_size).to(self.device)
            self.model = torch.jit.trace(self.model, dummy_input)
            print("  JIT compilation: enabled")
        except Exception as e:
            print(f"  JIT compilation: disabled ({e})")
        
        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Thread pool for preprocessing
        self.preprocess_pool = ThreadPoolExecutor(max_workers=num_threads)
        
        print(f"ParallelVideoProcessor initialized:")
        print(f"  Model: {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  CPU threads: {num_threads}")
        print(f"  Threshold: {threshold}")
    
    def _preprocess_frame_cv(self, frame: np.ndarray) -> np.ndarray:
        """Fast preprocessing using OpenCV (2-3x faster than PIL)."""
        # Resize with OpenCV
        resized = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize: [0,255] -> [0,1] -> normalized
        normalized = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        # HWC -> CHW
        transposed = normalized.transpose(2, 0, 1)
        return transposed
    
    def _classify_batch(self, frames: List[np.ndarray]) -> List[Tuple[bool, float]]:
        """Classify a batch of frames using multi-threaded preprocessing."""
        if not frames:
            return []
        
        # Multi-threaded preprocessing (utilizes all CPU cores)
        preprocessed = list(self.preprocess_pool.map(self._preprocess_frame_cv, frames))
        batch_np = np.stack(preprocessed)
        batch = torch.from_numpy(batch_np).to(self.device)
        
        # Inference (also multi-threaded via torch.set_num_threads)
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            valid_probs = probs[:, 1].cpu().numpy()
        
        results = [(prob >= self.threshold, float(prob)) for prob in valid_probs]
        return results
    
    def _read_frames_worker(self, cap, frame_queue: Queue, total_frames: int):
        """Worker thread to read frames into queue."""
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
        frame_queue.put(None)  # Signal end
    
    def process_video(self, video_path: str, output_path: str) -> Dict:
        """
        Process every frame with pipelined I/O and compute.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing: {os.path.basename(video_path)}")
        print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Create output video writer
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Pipeline: read frames in background thread
        frame_queue = Queue(maxsize=self.batch_size * 3)
        reader_thread = threading.Thread(
            target=self._read_frames_worker, 
            args=(cap, frame_queue, total_frames)
        )
        reader_thread.start()
        
        valid_count = 0
        batch_frames = []
        frames_read = 0
        
        pbar = tqdm(total=total_frames, desc="Processing", unit="frames")
        
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            
            batch_frames.append(frame)
            frames_read += 1
            
            # Process batch
            if len(batch_frames) >= self.batch_size:
                results = self._classify_batch(batch_frames)
                
                for f, (is_valid, _) in zip(batch_frames, results):
                    if is_valid:
                        out.write(f)
                        valid_count += 1
                
                pbar.update(len(batch_frames))
                batch_frames = []
        
        # Process remaining frames
        if batch_frames:
            results = self._classify_batch(batch_frames)
            for f, (is_valid, _) in zip(batch_frames, results):
                if is_valid:
                    out.write(f)
                    valid_count += 1
            pbar.update(len(batch_frames))
        
        reader_thread.join()
        pbar.close()
        cap.release()
        out.release()
        
        elapsed = time.time() - start_time
        
        stats = {
            'total_frames': frames_read,
            'valid_frames': valid_count,
            'invalid_frames': frames_read - valid_count,
            'selection_ratio': valid_count / max(1, frames_read),
            'processing_time': elapsed,
            'fps_processed': frames_read / elapsed if elapsed > 0 else 0,
            'input_path': video_path,
            'output_path': output_path
        }
        
        print(f"  Done: {valid_count}/{frames_read} valid ({stats['selection_ratio']*100:.1f}%) | {elapsed:.1f}s ({stats['fps_processed']:.1f} fps)")
        
        return stats
    
    def process_videos_batch(self, video_paths: List[str], output_dir: str) -> List[Dict]:
        """
        Process multiple videos.
        
        Args:
            video_paths: List of input video paths
            output_dir: Output directory
            
        Returns:
            List of processing statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_stats = []
        total_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Batch Processing: {len(video_paths)} videos")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
        for i, video_path in enumerate(video_paths):
            print(f"\n[{i+1}/{len(video_paths)}] {os.path.basename(video_path)}")
            
            output_filename = os.path.basename(video_path).replace('.mp4', '_filtered.mp4')
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                stats = self.process_video(video_path, output_path)
                all_stats.append(stats)
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                all_stats.append({'error': str(e), 'input_path': video_path})
        
        total_elapsed = time.time() - total_start
        
        # Summary
        successful = [s for s in all_stats if 'error' not in s]
        total_frames = sum(s['total_frames'] for s in successful)
        total_valid = sum(s['valid_frames'] for s in successful)
        
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete!")
        print(f"{'='*60}")
        print(f"  Videos processed: {len(successful)}/{len(video_paths)}")
        print(f"  Total frames: {total_frames}")
        print(f"  Total valid: {total_valid}")
        print(f"  Total time: {total_elapsed/60:.1f} minutes")
        print(f"  Average speed: {total_frames/total_elapsed:.1f} fps")
        
        return all_stats
