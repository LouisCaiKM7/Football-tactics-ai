"""
Video Processing Script - Parallel Version

使用训练好的模型并行处理视频，只保留有效帧（俯瞰/广角视角）
- 64线程并行处理
- 每个视频分成64段分别处理后合并
"""

import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VideoProcessor.video_selector_model import ParallelVideoProcessor


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = r"E:\0_projects\00_football_system\checkpoints\best_model.pth"
    VIDEO_DIR = r"E:\0_projects\00_football_system\video_resources"
    OUTPUT_DIR = r"E:\0_projects\00_football_system\output_videos"
    
    # Videos to process (02.mp4 to 64.mp4), skip already done
    video_files = []
    for i in range(4, 65):  # 02 to 64
        video_path = os.path.join(VIDEO_DIR, f"{i:02d}.mp4")
        output_path = os.path.join(OUTPUT_DIR, f"{i:02d}_filtered.mp4")
        if os.path.exists(video_path) and not os.path.exists(output_path):
            video_files.append(video_path)
    
    print("="*60)
    print("Football Video Frame Filter - Parallel Processing")
    print("="*60)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Input directory: {VIDEO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Videos to process: {len(video_files)}")
    print()
    
    # Initialize processor (max speed: pipeline + JIT + multi-thread)
    processor = ParallelVideoProcessor(
        model_path=MODEL_PATH,
        backbone='resnet18',      # Must match training backbone
        threshold=0.5,            # Frames with confidence > 0.5 are kept
        batch_size=1000,           # Larger batch = better throughput
        image_size=224,          
    )
    
    # Process all videos
    all_stats = processor.process_videos_batch(
        video_paths=video_files,
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("All Processing Complete!")
    print("="*60)
