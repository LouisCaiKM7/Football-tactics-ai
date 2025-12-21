"""
High-Performance Video Processing with ONNX Runtime + Multi-Process

优化策略:
- 多进程并行推理 (绑定多个ONNX session)
- 每个进程独立处理视频段
- 完全利用所有CPU核心
- 预计速度: 200-400 fps

"""

import os
import sys
import cv2
import numpy as np
import time
import threading
import multiprocessing as mp
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from typing import List, Dict, Tuple
import tempfile
import shutil

try:
    import onnxruntime as ort
except ImportError:
    print("请先安装 onnxruntime: pip install onnxruntime")
    sys.exit(1)


class ONNXVideoProcessor:
    """
    Ultra-fast video processor using ONNX Runtime.
    Fully utilizes all CPU cores for maximum performance.
    """
    
    def __init__(self, onnx_path: str, threshold: float = 0.5,
                 batch_size: int = 64, image_size: int = 224,
                 num_threads: int = 0):
        """
        Initialize ONNX processor.
        
        Args:
            onnx_path: Path to ONNX model
            threshold: Classification threshold
            batch_size: Batch size for inference
            image_size: Input image size
            num_threads: CPU threads (0=auto, use all)
        """
        self.threshold = threshold
        self.batch_size = batch_size
        self.image_size = image_size
        
        # Auto-detect threads
        if num_threads <= 0:
            num_threads = os.cpu_count() or 8
        self.num_threads = num_threads
        
        # Configure ONNX Runtime for maximum CPU utilization
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable parallel execution
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        # Load ONNX model
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Thread pool for preprocessing
        self.preprocess_pool = ThreadPoolExecutor(max_workers=num_threads)
        
        print(f"ONNXVideoProcessor initialized:")
        print(f"  Model: {onnx_path}")
        print(f"  CPU threads: {num_threads}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  Threshold: {threshold}")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Fast preprocessing with OpenCV."""
        resized = cv2.resize(frame, (self.image_size, self.image_size), 
                            interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        return normalized.transpose(2, 0, 1)  # HWC -> CHW
    
    def _classify_batch(self, frames: List[np.ndarray]) -> List[Tuple[bool, float]]:
        """Classify batch using ONNX Runtime (multi-threaded)."""
        if not frames:
            return []
        
        # Multi-threaded preprocessing
        preprocessed = list(self.preprocess_pool.map(self._preprocess_frame, frames))
        batch = np.stack(preprocessed).astype(np.float32)
        
        # ONNX Runtime inference (uses all CPU cores)
        outputs = self.session.run([self.output_name], {self.input_name: batch})[0]
        
        # Softmax and threshold
        exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
        valid_probs = probs[:, 1]
        
        return [(prob >= self.threshold, float(prob)) for prob in valid_probs]
    
    def _read_frames_worker(self, cap, frame_queue: Queue, total_frames: int):
        """Background thread for reading frames."""
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
        frame_queue.put(None)
    
    def process_video(self, video_path: str, output_path: str) -> Dict:
        """Process video with pipelined I/O and ONNX inference."""
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing: {os.path.basename(video_path)}")
        print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Pipeline: read frames in background
        frame_queue = Queue(maxsize=self.batch_size * 4)
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
            
            if len(batch_frames) >= self.batch_size:
                results = self._classify_batch(batch_frames)
                for f, (is_valid, _) in zip(batch_frames, results):
                    if is_valid:
                        out.write(f)
                        valid_count += 1
                pbar.update(len(batch_frames))
                batch_frames = []
        
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
        fps_processed = frames_read / elapsed if elapsed > 0 else 0
        
        print(f"  Done: {valid_count}/{frames_read} valid ({valid_count/max(1,frames_read)*100:.1f}%)")
        print(f"  Time: {elapsed:.1f}s ({fps_processed:.1f} fps) | Est: {elapsed/60:.1f} min")
        
        return {
            'total_frames': frames_read,
            'valid_frames': valid_count,
            'processing_time': elapsed,
            'fps_processed': fps_processed
        }
    
    def process_videos_batch(self, video_paths: List[str], output_dir: str) -> List[Dict]:
        """Process multiple videos."""
        os.makedirs(output_dir, exist_ok=True)
        all_stats = []
        total_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"ONNX Batch Processing: {len(video_paths)} videos")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")
        
        for i, video_path in enumerate(video_paths):
            print(f"\n[{i+1}/{len(video_paths)}] {os.path.basename(video_path)}")
            output_path = os.path.join(
                output_dir, 
                os.path.basename(video_path).replace('.mp4', '_filtered.mp4')
            )
            
            try:
                stats = self.process_video(video_path, output_path)
                all_stats.append(stats)
            except Exception as e:
                print(f"Error: {e}")
                all_stats.append({'error': str(e)})
        
        total_elapsed = time.time() - total_start
        successful = [s for s in all_stats if 'error' not in s]
        total_frames = sum(s['total_frames'] for s in successful)
        
        print(f"\n{'='*60}")
        print(f"Complete! {len(successful)}/{len(video_paths)} videos")
        print(f"Total time: {total_elapsed/60:.1f} min")
        print(f"Average: {total_frames/total_elapsed:.1f} fps")
        print(f"{'='*60}")
        
        return all_stats


def _process_segment_worker_with_progress(args):
    """Worker function with shared progress counter."""
    (segment_id, video_path, start_frame, end_frame, onnx_path, 
     threshold, batch_size, image_size, temp_dir, progress_counter, valid_counter) = args
    
    # Each process creates its own ONNX session
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4  # 8 processes x 2 threads = 16 cores
    sess_options.inter_op_num_threads = 2
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess(frame):
        resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return ((rgb.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    temp_output = os.path.join(temp_dir, f"seg_{segment_id:03d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    local_valid = 0
    batch_frames = []
    
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        batch_frames.append(frame)
        
        if len(batch_frames) >= batch_size:
            batch_np = np.stack([preprocess(f) for f in batch_frames]).astype(np.float32)
            outputs = session.run([output_name], {input_name: batch_np})[0]
            exp_out = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
            
            for f, prob in zip(batch_frames, probs[:, 1]):
                if prob >= threshold:
                    out.write(f)
                    local_valid += 1
            
            # Update shared progress counter
            with progress_counter.get_lock():
                progress_counter.value += len(batch_frames)
            batch_frames = []
    
    if batch_frames:
        batch_np = np.stack([preprocess(f) for f in batch_frames]).astype(np.float32)
        outputs = session.run([output_name], {input_name: batch_np})[0]
        exp_out = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
        for f, prob in zip(batch_frames, probs[:, 1]):
            if prob >= threshold:
                out.write(f)
                local_valid += 1
        with progress_counter.get_lock():
            progress_counter.value += len(batch_frames)
    
    with valid_counter.get_lock():
        valid_counter.value += local_valid
    
    cap.release()
    out.release()
    return (segment_id, temp_output, local_valid, end_frame - start_frame)


class MultiProcessONNXProcessor:
    """
    Multi-process ONNX processor for 2x speedup.
    Splits video into segments and processes with multiple ONNX sessions.
    """
    
    def __init__(self, onnx_path: str, threshold: float = 0.5,
                 batch_size: int = 128, image_size: int = 224,
                 num_processes: int = 4):
        self.onnx_path = onnx_path
        self.threshold = threshold
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_processes = num_processes
        
        print(f"MultiProcessONNXProcessor:")
        print(f"  Processes: {num_processes}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {image_size}")
    
    def process_video(self, video_path: str, output_path: str) -> Dict:
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"\nProcessing: {os.path.basename(video_path)}")
        print(f"  Frames: {total_frames}, Processes: {self.num_processes}")
        
        temp_dir = tempfile.mkdtemp(prefix="onnx_mp_")
        
        # Shared counters for real-time progress
        progress_counter = mp.Value('i', 0)
        valid_counter = mp.Value('i', 0)
        
        # Split into segments
        frames_per_seg = total_frames // self.num_processes
        segments = []
        for i in range(self.num_processes):
            start = i * frames_per_seg
            end = (i + 1) * frames_per_seg if i < self.num_processes - 1 else total_frames
            segments.append((i, video_path, start, end, self.onnx_path,
                           self.threshold, self.batch_size, self.image_size, temp_dir,
                           progress_counter, valid_counter))
        
        # Start worker processes
        processes = []
        for seg in segments:
            p = mp.Process(target=_process_segment_worker_with_progress, args=(seg,))
            p.start()
            processes.append(p)
        
        # Real-time progress bar with timeout
        with tqdm(total=total_frames, desc="Processing", unit="frames") as pbar:
            last_progress = 0
            max_wait = 600  # Max 10 minutes wait
            start_wait = time.time()
            
            while any(p.is_alive() for p in processes):
                current = progress_counter.value
                if current > last_progress:
                    pbar.update(current - last_progress)
                    last_progress = current
                
                # Force terminate if waiting too long after 100%
                if current >= total_frames and (time.time() - start_wait) > 10:
                    print("\n  Processing complete, terminating workers...")
                    for p in processes:
                        if p.is_alive():
                            p.terminate()
                    break
                    
                time.sleep(0.1)
            
            # Final update
            pbar.update(progress_counter.value - last_progress)
        
        # Clean up processes
        print("  Cleaning up processes...")
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        
        # Merge outputs with progress
        print("  Merging segments...")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_valid = valid_counter.value
        merged_frames = 0
        for i in range(self.num_processes):
            temp_file = os.path.join(temp_dir, f"seg_{i:03d}.mp4")
            if os.path.exists(temp_file):
                seg_cap = cv2.VideoCapture(temp_file)
                while True:
                    ret, frame = seg_cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    merged_frames += 1
                seg_cap.release()
                print(f"    Merged segment {i+1}/{self.num_processes} ({merged_frames} frames)")
        out.release()
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        elapsed = time.time() - start_time
        print(f"  Done: {total_valid}/{total_frames} ({total_valid/max(1,total_frames)*100:.1f}%)")
        print(f"  Time: {elapsed:.1f}s ({total_frames/elapsed:.1f} fps)")
        
        return {'total_frames': total_frames, 'valid_frames': total_valid, 
                'processing_time': elapsed, 'fps_processed': total_frames/elapsed}
    
    def process_videos_batch(self, video_paths: List[str], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        total_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Multi-Process ONNX: {len(video_paths)} videos, {self.num_processes} processes")
        print(f"{'='*60}")
        
        all_stats = []
        for i, vp in enumerate(video_paths):
            print(f"\n[{i+1}/{len(video_paths)}] {os.path.basename(vp)}")
            out_path = os.path.join(output_dir, os.path.basename(vp).replace('.mp4', '_filtered.mp4'))
            try:
                stats = self.process_video(vp, out_path)
                all_stats.append(stats)
            except Exception as e:
                print(f"Error: {e}")
        
        elapsed = time.time() - total_start
        total_frames = sum(s.get('total_frames', 0) for s in all_stats)
        print(f"\n{'='*60}")
        print(f"Complete! {elapsed/60:.1f} min, avg {total_frames/elapsed:.1f} fps")


if __name__ == "__main__":
    ONNX_PATH = r"E:\0_projects\00_football_system\checkpoints\model.onnx"
    VIDEO_DIR = r"E:\0_projects\00_football_system\video_resources"
    OUTPUT_DIR = r"E:\0_projects\00_football_system\output_videos"
    
    # Find videos to process (skip already done)
    video_files = []
    for i in range(4, 65):
        video_path = os.path.join(VIDEO_DIR, f"{i:02d}.mp4")
        output_path = os.path.join(OUTPUT_DIR, f"{i:02d}_filtered.mp4")
        if os.path.exists(video_path) and not os.path.exists(output_path):
            video_files.append(video_path)
    
    print("="*60)
    print("Football Video Filter - Multi-Process ONNX (Real-time Progress)")
    print("="*60)
    print(f"\nModel: {ONNX_PATH}")
    print(f"Videos to process: {len(video_files)}")
    
    # Multi-process ONNX with real-time progress bar
    processor = MultiProcessONNXProcessor(
        onnx_path=ONNX_PATH,
        threshold=0.5,
        batch_size=128,           # Smaller batch = better parallelism
        image_size=224,           # Keep quality
        num_processes=16,          # 8 parallel processes
    )
    
    # Process all videos
    processor.process_videos_batch(video_files, OUTPUT_DIR)
