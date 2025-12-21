"""
Frame Labeling Tool

快速为帧图片分类的小工具
- A键: 移动到 valid/ (有效帧 - 俯瞰/广角视角)
- D键: 移动到 invalid/ (无效帧 - 特写/球迷/球员)
- S键: 跳过当前图片
- Q键: 退出
- Z键: 撤销上一次操作
"""

import cv2
import os
import shutil
from pathlib import Path
from typing import Optional, List, Tuple


class FrameLabeler:
    def __init__(self, dataset_dir: str = r"E:\0_projects\00_football_system\frames_dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.unlabeled_dir = self.dataset_dir / "unlabeled"
        self.valid_dir = self.dataset_dir / "valid"
        self.invalid_dir = self.dataset_dir / "invalid"
        
        # Ensure directories exist
        for d in [self.unlabeled_dir, self.valid_dir, self.invalid_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # History for undo
        self.history: List[Tuple[Path, Path]] = []  # (original, moved_to)
        
        # Get list of images
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.images = self._get_unlabeled_images()
        self.current_index = 0
        
    def _get_unlabeled_images(self) -> List[Path]:
        """Get list of unlabeled images."""
        images = [f for f in self.unlabeled_dir.iterdir() 
                  if f.suffix.lower() in self.image_extensions]
        return sorted(images)
    
    def _refresh_images(self):
        """Refresh the image list."""
        self.images = self._get_unlabeled_images()
        if self.current_index >= len(self.images):
            self.current_index = max(0, len(self.images) - 1)
    
    def move_to_valid(self, image_path: Path) -> bool:
        """Move image to valid folder."""
        dest = self.valid_dir / image_path.name
        try:
            shutil.move(str(image_path), str(dest))
            self.history.append((image_path, dest))
            return True
        except Exception as e:
            print(f"Error moving file: {e}")
            return False
    
    def move_to_invalid(self, image_path: Path) -> bool:
        """Move image to invalid folder."""
        dest = self.invalid_dir / image_path.name
        try:
            shutil.move(str(image_path), str(dest))
            self.history.append((image_path, dest))
            return True
        except Exception as e:
            print(f"Error moving file: {e}")
            return False
    
    def undo(self) -> bool:
        """Undo last move operation."""
        if not self.history:
            return False
        
        original_path, moved_path = self.history.pop()
        try:
            shutil.move(str(moved_path), str(original_path))
            self._refresh_images()
            return True
        except Exception as e:
            print(f"Error undoing: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get current labeling statistics."""
        valid_count = len([f for f in self.valid_dir.iterdir() 
                          if f.suffix.lower() in self.image_extensions])
        invalid_count = len([f for f in self.invalid_dir.iterdir() 
                            if f.suffix.lower() in self.image_extensions])
        unlabeled_count = len(self.images)
        total = valid_count + invalid_count + unlabeled_count
        
        return {
            'valid': valid_count,
            'invalid': invalid_count,
            'unlabeled': unlabeled_count,
            'total': total,
            'progress': (valid_count + invalid_count) / total * 100 if total > 0 else 0
        }
    
    def run(self):
        """Run the labeling tool."""
        print("\n" + "="*60)
        print("Frame Labeling Tool - 帧分类工具")
        print("="*60)
        print("\n快捷键:")
        print("  A - 标记为 Valid (有效帧/俯瞰视角)")
        print("  D - 标记为 Invalid (无效帧/特写/球迷)")
        print("  S - 跳过当前图片")
        print("  Z - 撤销上一次操作")
        print("  Q - 退出")
        print("  ← → - 上一张/下一张 (不移动文件)")
        print("\n" + "="*60 + "\n")
        
        if not self.images:
            print("没有找到待标记的图片!")
            print(f"请先运行 prepare_dataset.py 提取帧到 {self.unlabeled_dir}")
            return
        
        window_name = "Frame Labeler - A:Valid D:Invalid S:Skip Z:Undo Q:Quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        while True:
            self._refresh_images()
            
            if not self.images:
                print("\n所有图片已标记完成!")
                break
            
            # Get current image
            if self.current_index >= len(self.images):
                self.current_index = 0
            
            current_image_path = self.images[self.current_index]
            
            # Load and display image
            img = cv2.imread(str(current_image_path))
            if img is None:
                print(f"无法读取图片: {current_image_path}")
                self.current_index += 1
                continue
            
            # Add info overlay
            stats = self.get_stats()
            info_text = [
                f"[{self.current_index + 1}/{len(self.images)}] {current_image_path.name}",
                f"Valid: {stats['valid']} | Invalid: {stats['invalid']} | Progress: {stats['progress']:.1f}%",
                "A:Valid  D:Invalid  S:Skip  Z:Undo  Q:Quit"
            ]
            
            # Draw info on image
            display_img = img.copy()
            y_offset = 30
            for text in info_text:
                # Draw background rectangle
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(display_img, (10, y_offset - 25), (text_w + 20, y_offset + 5), (0, 0, 0), -1)
                cv2.putText(display_img, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                y_offset += 35
            
            cv2.imshow(window_name, display_img)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\n退出标记工具")
                break
            
            elif key == ord('a') or key == ord('A'):
                # Move to valid
                if self.move_to_valid(current_image_path):
                    print(f"✓ Valid: {current_image_path.name}")
            
            elif key == ord('d') or key == ord('D'):
                # Move to invalid
                if self.move_to_invalid(current_image_path):
                    print(f"✗ Invalid: {current_image_path.name}")
            
            elif key == ord('s') or key == ord('S'):
                # Skip
                self.current_index += 1
                print(f"→ Skipped: {current_image_path.name}")
            
            elif key == ord('z') or key == ord('Z'):
                # Undo
                if self.undo():
                    print("↩ Undo successful")
                else:
                    print("Nothing to undo")
            
            elif key == 81 or key == 2424832:  # Left arrow
                self.current_index = max(0, self.current_index - 1)
            
            elif key == 83 or key == 2555904:  # Right arrow
                self.current_index = min(len(self.images) - 1, self.current_index + 1)
        
        cv2.destroyAllWindows()
        
        # Print final stats
        stats = self.get_stats()
        print("\n" + "="*60)
        print("标记完成统计:")
        print(f"  Valid (有效帧): {stats['valid']}")
        print(f"  Invalid (无效帧): {stats['invalid']}")
        print(f"  Unlabeled (未标记): {stats['unlabeled']}")
        print(f"  Progress: {stats['progress']:.1f}%")
        print("="*60)


if __name__ == "__main__":
    labeler = FrameLabeler(r"E:\0_projects\00_football_system\frames_dataset")
    labeler.run()
