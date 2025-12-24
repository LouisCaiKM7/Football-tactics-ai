"""
Frame Extraction Script for Dataset Preparation

Extracts frames from football match videos for manual labeling.
After extraction, manually move frames to:
  - frames_dataset/valid/    (overlook/wide-angle field views)
  - frames_dataset/invalid/  (closeups, fans, celebrations, players)
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import random


def extract_frames_for_dataset(
    video_dir: str = "\video_resources",
    output_dir: str = "\frames_dataset",
    frames_per_video: int = 100,
    frame_skip: int = 30,
    random_sample: bool = True,
    seed: int = 42
):
    """
    Extract frames from all videos in a directory for dataset preparation.
    
    Args:
        video_dir: Directory containing video files
        output_dir: Output directory for extracted frames
        frames_per_video: Number of frames to extract per video
        frame_skip: Minimum frames between extractions (if not random)
        random_sample: Randomly sample frames instead of uniform sampling
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    unlabeled_dir = output_dir / "unlabeled"
    valid_dir = output_dir / "valid"
    invalid_dir = output_dir / "invalid"
    
    for d in [unlabeled_dir, valid_dir, invalid_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv')
    video_files = [f for f in video_dir.iterdir() 
                   if f.suffix.lower() in video_extensions]
    
    print(f"Found {len(video_files)} videos in {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frames per video: {frames_per_video}")
    print(f"Random sampling: {random_sample}")
    print(f"\n{'='*60}\n")
    
    total_extracted = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = video_path.stem
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Cannot open {video_path}")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0:
            print(f"Warning: Cannot get frame count for {video_path}")
            cap.release()
            continue
        
        # Determine which frames to extract
        if random_sample:
            # Random sampling across the video
            frame_indices = sorted(random.sample(
                range(0, total_frames), 
                min(frames_per_video, total_frames)
            ))
        else:
            # Uniform sampling
            step = max(1, total_frames // frames_per_video)
            frame_indices = list(range(0, total_frames, step))[:frames_per_video]
        
        # Extract frames
        extracted_count = 0
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Save frame
            timestamp = frame_idx / fps if fps > 0 else 0
            filename = f"{video_name}_frame{frame_idx:06d}_t{timestamp:.1f}s.jpg"
            output_path = unlabeled_dir / filename
            
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_count += 1
        
        cap.release()
        total_extracted += extracted_count
        
        tqdm.write(f"  {video_name}: extracted {extracted_count} frames")
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Total frames extracted: {total_extracted}")
    print(f"\nNext steps:")
    print(f"1. Go to: {unlabeled_dir}")
    print(f"2. Review each frame")
    print(f"3. Move overlook/wide-angle views to: {valid_dir}")
    print(f"4. Move closeups/fans/players to: {invalid_dir}")
    print(f"\nTip: Use an image viewer with keyboard shortcuts for fast sorting")
    print(f"{'='*60}")
    
    return total_extracted


def check_dataset_status(output_dir: str = "\frames_dataset"):
    """Check the current status of the dataset."""
    output_dir = Path(output_dir)
    
    unlabeled_dir = output_dir / "unlabeled"
    valid_dir = output_dir / "valid"
    invalid_dir = output_dir / "invalid"
    
    def count_images(directory):
        if not directory.exists():
            return 0
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        return len([f for f in directory.iterdir() if f.suffix.lower() in extensions])
    
    unlabeled = count_images(unlabeled_dir)
    valid = count_images(valid_dir)
    invalid = count_images(invalid_dir)
    total = unlabeled + valid + invalid
    
    print(f"\nDataset Status: {output_dir}")
    print(f"{'='*40}")
    print(f"  Unlabeled: {unlabeled}")
    print(f"  Valid:     {valid}")
    print(f"  Invalid:   {invalid}")
    print(f"  Total:     {total}")
    
    if total > 0:
        labeled = valid + invalid
        print(f"\n  Progress: {labeled}/{total} ({labeled/total*100:.1f}% labeled)")
    
    if unlabeled == 0 and total > 0:
        print(f"\n  ✓ Dataset is fully labeled and ready for training!")
    
    return {'unlabeled': unlabeled, 'valid': valid, 'invalid': invalid}


if __name__ == "__main__":
    # Extract frames from all videos
    extract_frames_for_dataset(
        video_dir="\video_resources",
        output_dir="\frames_dataset",
        frames_per_video=100,  # 100 frames per video = ~6300 total frames
        random_sample=True,
        seed=42
    )
    
    # Check status
    check_dataset_status()
