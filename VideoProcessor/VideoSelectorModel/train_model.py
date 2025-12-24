"""
Training Script for Frame Classifier

训练帧分类模型
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VideoProcessor.video_selector_model import train_classifier


if __name__ == "__main__":
    # Training configuration
    DATA_DIR = r"E:\0_projects\00_football_system\frames_dataset"
    CHECKPOINT_DIR = r"E:\0_projects\00_football_system\checkpoints"
    
    print("="*60)
    print("Football Frame Classifier Training")
    print("="*60)
    print(f"\nDataset: {DATA_DIR}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print()
    
    # Train the model
    history = train_classifier(
        data_dir=DATA_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        backbone='resnet18',      # resnet18/resnet34/resnet50/mobilenet_v2/efficientnet_b0
        epochs=30,                # Training epochs
        batch_size=32,            # Batch size (reduce if GPU OOM)
        lr=1e-4,                  # Learning rate
        val_split=0.2,            # 20% for validation
        num_workers=4,            # DataLoader workers
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nBest model saved to: {CHECKPOINT_DIR}/best_model.pth")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    print("\nNext step: Use the model for video processing")
    print("  python process_video.py")
