import cv2
import os
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoMerger:
    """
    A smart video merging utility that handles various video formats,
    resolutions, and edge cases with intelligent processing.

    Features:
    - Automatic detection of video properties
    - Resolution normalization
    - Frame rate synchronization
    - Codec compatibility handling
    - Error recovery mechanisms
    - Progress tracking
    """

    def __init__(self, output_dir: str):
        """
        Initialize VideoMerger with output directory.

        Args:
            output_dir: Directory where merged videos will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Supported video extensions
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.mpg'}

        # Default output settings (will be adjusted based on input)
        self.default_output_settings = {
            'codec': 'mp4v',  # MP4 codec
            'extension': '.mp4',
            'fps': 30,
            'resolution': (1920, 1080)  # Will be adjusted
        }

        # Store video segments for merging
        self.video_segments = []

        logger.info(f"VideoMerger initialized. Output directory: {output_dir}")

    def _validate_video_file(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate if the provided file is a valid video.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(video_path)

        # Check if file exists
        if not path.exists():
            return False, f"File not found: {video_path}"

        # Check file extension
        if path.suffix.lower() not in self.supported_formats:
            return False, f"Unsupported format: {path.suffix}. Supported: {self.supported_formats}"

        # Try to open video with OpenCV
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return False, f"Cannot open video file: {video_path}"

        # Check if video has frames
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False, f"No frames found in video: {video_path}"

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        if fps <= 0:
            return False, f"Invalid frame rate: {fps}"
        if frame_count <= 0:
            return False, f"Invalid frame count: {frame_count}"

        return True, "Valid video file"

    def _get_video_properties(self, video_path: str) -> Dict:
        """
        Extract video properties including resolution, FPS, codec, etc.

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing video properties
        """
        cap = cv2.VideoCapture(video_path)

        properties = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'aspect_ratio': None
        }

        # Calculate aspect ratio
        if properties['height'] > 0:
            properties['aspect_ratio'] = properties['width'] / properties['height']

        # Decode FOURCC codec code to readable format
        try:
            properties['codec_str'] = ''.join([
                chr((properties['codec'] >> 8 * i) & 0xFF)
                for i in range(4)
            ])
        except:
            properties['codec_str'] = 'Unknown'

        cap.release()

        logger.debug(f"Video properties for {video_path}: {properties}")
        return properties

    def _determine_output_settings(self, videos: List[str]) -> Dict:
        """
        Determine optimal output settings based on input videos.

        Args:
            videos: List of video paths to be merged

        Returns:
            Dictionary of output settings
        """
        if not videos:
            return self.default_output_settings.copy()

        # Collect properties of all videos
        all_props = [self._get_video_properties(vid) for vid in videos]

        # Determine common resolution (use the most common or largest)
        resolutions = [(p['width'], p['height']) for p in all_props]

        # Use mode resolution if exists, else use maximum resolution
        from collections import Counter
        resolution_counts = Counter(resolutions)
        most_common_res = resolution_counts.most_common(1)

        if most_common_res:
            output_res = most_common_res[0][0]
        else:
            # Use maximum resolution
            output_res = max(resolutions, key=lambda x: x[0] * x[1])

        # Determine FPS (use mode or average)
        fps_values = [p['fps'] for p in all_props]
        fps_counts = Counter([round(fps, 1) for fps in fps_values])
        most_common_fps = fps_counts.most_common(1)

        if most_common_fps:
            output_fps = most_common_fps[0][0]
        else:
            output_fps = round(np.mean(fps_values), 1)

        # Ensure FPS is reasonable
        output_fps = max(15, min(60, output_fps))

        settings = self.default_output_settings.copy()
        settings['resolution'] = output_res
        settings['fps'] = output_fps

        logger.info(f"Output settings determined: {settings}")
        return settings

    def _resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize frame to target size while maintaining aspect ratio.

        Args:
            frame: Input frame
            target_size: Target (width, height)

        Returns:
            Resized frame
        """
        if frame is None:
            return None

        h, w = frame.shape[:2]
        target_w, target_h = target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)

        if scale != 1:
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Pad if necessary to reach target size
            if new_w != target_w or new_h != target_h:
                pad_w = (target_w - new_w) // 2
                pad_h = (target_h - new_h) // 2

                frame = cv2.copyMakeBorder(
                    frame,
                    pad_h, target_h - new_h - pad_h,
                    pad_w, target_w - new_w - pad_w,
                    cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

        return frame

    def merge_video(self, video_path: str, segment_info: Optional[Dict] = None) -> bool:
        """
        Add a video segment to the merge queue.

        Args:
            video_path: Path to the video file to add
            segment_info: Optional dictionary with segment information
                         (start_time, end_time for trimming)

        Returns:
            True if video was successfully added, False otherwise
        """
        try:
            # Validate video file
            is_valid, error_msg = self._validate_video_file(video_path)
            if not is_valid:
                logger.error(f"Invalid video: {error_msg}")
                return False

            # Get video properties
            properties = self._get_video_properties(video_path)

            # Create segment info
            segment = {
                'path': video_path,
                'properties': properties,
                'segment_info': segment_info or {}
            }

            self.video_segments.append(segment)
            logger.info(f"Added video segment: {video_path}")
            return True

        except Exception as e:
            logger.error(f"Error adding video segment {video_path}: {str(e)}")
            return False

    def merge_all(self, output_filename: Optional[str] = None) -> Optional[str]:
        """
        Merge all added video segments into a single video.

        Args:
            output_filename: Optional output filename
                           (if None, generates timestamp-based name)

        Returns:
            Path to the merged video file if successful, None otherwise
        """
        if not self.video_segments:
            logger.warning("No video segments to merge")
            return None

        try:
            # Determine output settings based on all segments
            video_paths = [seg['path'] for seg in self.video_segments]
            output_settings = self._determine_output_settings(video_paths)

            # Generate output filename if not provided
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"merged_video_{timestamp}{output_settings['extension']}"

            output_path = self.output_dir / output_filename

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*output_settings['codec'])
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                output_settings['fps'],
                output_settings['resolution']
            )

            if not out.isOpened():
                logger.error(f"Cannot create video writer for {output_path}")
                return None

            total_frames = 0
            processed_segments = 0

            # Process each segment
            for i, segment in enumerate(self.video_segments):
                try:
                    video_path = segment['path']
                    segment_info = segment['segment_info']

                    logger.info(f"Processing segment {i + 1}/{len(self.video_segments)}: {video_path}")

                    cap = cv2.VideoCapture(video_path)

                    # Calculate frame range for trimming
                    start_frame = 0
                    end_frame = segment['properties']['frame_count'] - 1

                    if 'start_time' in segment_info and segment_info['start_time'] > 0:
                        start_frame = int(segment_info['start_time'] * segment['properties']['fps'])

                    if 'end_time' in segment_info and segment_info['end_time'] > 0:
                        end_frame = min(
                            int(segment_info['end_time'] * segment['properties']['fps']),
                            end_frame
                        )

                    # Set start position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                    frame_idx = start_frame
                    segment_frames = 0

                    # Read and write frames
                    while frame_idx <= end_frame:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Resize frame to output resolution
                        frame = self._resize_frame(frame, output_settings['resolution'])

                        # Write frame
                        out.write(frame)

                        frame_idx += 1
                        segment_frames += 1
                        total_frames += 1

                        # Progress logging every 100 frames
                        if segment_frames % 100 == 0:
                            logger.debug(f"Segment {i + 1}: Processed {segment_frames} frames")

                    cap.release()
                    processed_segments += 1

                    logger.info(f"Completed segment {i + 1}: {segment_frames} frames")

                except Exception as e:
                    logger.error(f"Error processing segment {video_path}: {str(e)}")
                    # Continue with next segment instead of failing completely

            # Release video writer
            out.release()

            # Verify output file
            if total_frames > 0:
                # Quick verification
                verify_cap = cv2.VideoCapture(str(output_path))
                if verify_cap.isOpened():
                    output_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    verify_cap.release()

                    if output_frames > 0:
                        logger.info(
                            f"Merging completed successfully: "
                            f"{processed_segments}/{len(self.video_segments)} segments, "
                            f"{output_frames} frames, "
                            f"Output: {output_path}"
                        )
                        return str(output_path)

            logger.error("Merging failed: No valid frames were processed")
            # Clean up empty output file
            if output_path.exists():
                output_path.unlink()
            return None

        except Exception as e:
            logger.error(f"Critical error during merging: {str(e)}")
            return None

    def clear_segments(self):
        """Clear all video segments from the merge queue."""
        self.video_segments.clear()
        logger.info("Cleared all video segments")

    def get_segment_count(self) -> int:
        """Get number of video segments in queue."""
        return len(self.video_segments)

    def get_total_duration(self) -> float:
        """Calculate total duration of all video segments."""
        total_duration = 0
        for segment in self.video_segments:
            duration = segment['properties']['duration']
            segment_info = segment['segment_info']

            # Adjust for trimming
            if 'start_time' in segment_info and 'end_time' in segment_info:
                duration = max(0, segment_info['end_time'] - segment_info['start_time'])
            elif 'start_time' in segment_info:
                duration = max(0, duration - segment_info['start_time'])
            elif 'end_time' in segment_info:
                duration = min(duration, segment_info['end_time'])

            total_duration += duration

        return total_duration


# Usage example
def example_usage():
    """Example of how to use the VideoMerger class."""

    # Initialize merger
    merger = VideoMerger("output_videos")

    # Add video segments (you can also specify trimming with segment_info)
    merger.merge_video("segment1.mp4")
    merger.merge_video("segment2.avi", segment_info={'start_time': 5, 'end_time': 30})
    merger.merge_video("segment3.mov")

    # Display information
    print(f"Segments to merge: {merger.get_segment_count()}")
    print(f"Total duration: {merger.get_total_duration():.2f} seconds")

    # Merge all segments
    output_path = merger.merge_all("final_match_highlights.mp4")

    if output_path:
        print(f"Merged video saved to: {output_path}")
    else:
        print("Failed to merge videos")

    # Clear for next batch
    merger.clear_segments()


