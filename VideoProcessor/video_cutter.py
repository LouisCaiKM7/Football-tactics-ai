import cv2
import os
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Data class to store video information"""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    fourcc: str


def _get_video_info(video_path: str) -> Optional[VideoInfo]:
    """
    Extract video information using OpenCV

    Args:
        video_path (str): Path to the video file

    Returns:
        VideoInfo: Video information object or None if video cannot be opened
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return None

    try:
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate duration
        duration_seconds = total_frames / fps if fps > 0 else 0

        # Get codec information
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        # Common codec mappings
        codec_map = {
            "avc1": "H.264",
            "h264": "H.264",
            "hevc": "H.265",
            "h265": "H.265",
            "mp4v": "MPEG-4",
            "mjpg": "Motion JPEG",
            "xvid": "Xvid"
        }
        codec = codec_map.get(fourcc.lower(), fourcc)

        video_info = VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration_seconds,
            codec=codec,
            fourcc=fourcc
        )

        logger.info(f"Video info: {width}x{height}, {fps:.2f} fps, "
                    f"{total_frames} frames, {duration_seconds:.2f} seconds")
        return video_info

    except Exception as e:
        logger.error(f"Error getting video info: {str(e)}")
        return None
    finally:
        cap.release()


class VideoCutter:
    """A class for cutting and processing video files with multiple cutting options"""

    def __init__(self, output_dir: str = "output_videos"):
        """
        Initialize VideoCutter with output directory

        Args:
            output_dir (str): Directory to save output videos. Defaults to "output_videos"
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"VideoCutter initialized. Output directory: {output_dir}")

    def time_checker(self, video_path: str, start_time: float, end_time: float) -> Tuple[bool, str, float, float]:
        """
        Validate if the provided time range is valid for the video.
        Automatically adjusts invalid time values within reasonable limits.

        Args:
            video_path (str): Path to the video file
            start_time (float): Start time in seconds
            end_time (float): End time in seconds

        Returns:
            Tuple[bool, str, float, float]:
                (is_valid, message, adjusted_start_time, adjusted_end_time)
        """
        video_info = _get_video_info(video_path)
        if not video_info:
            return False, "Cannot read video file", start_time, end_time

        # Store original values for logging
        original_start = start_time
        original_end = end_time

        # Auto-adjust start time if negative
        if start_time < 0:
            start_time = 0.0
            logger.warning(f"Start time ({original_start}s) adjusted to 0s (cannot be negative)")

        # Auto-adjust start time if exceeds video duration
        if start_time > video_info.duration_seconds:
            start_time = max(0.0, video_info.duration_seconds - 1.0)  # Set to 1 second before end
            logger.warning(f"Start time ({original_start}s) exceeds video duration "
                           f"({video_info.duration_seconds:.2f}s). Adjusted to {start_time:.2f}s")

        # Auto-adjust end time if exceeds video duration
        if end_time > video_info.duration_seconds:
            end_time = video_info.duration_seconds
            logger.warning(f"End time ({original_end}s) exceeds video duration "
                           f"({video_info.duration_seconds:.2f}s). Adjusted to {end_time:.2f}s")

        # Auto-adjust end time if less than or equal to start time
        if end_time <= start_time:
            if start_time < video_info.duration_seconds:
                # Set end time to start_time + 1 second, but not exceeding video duration
                end_time = min(start_time + 1.0, video_info.duration_seconds)
            else:
                # If start time is at or near the end, set end time to video duration
                end_time = video_info.duration_seconds

            logger.warning(f"End time ({original_end}s) must be greater than start time "
                           f"({original_start}s). Adjusted to {end_time:.2f}s")

        # Final validation after adjustments
        if start_time < 0 or end_time <= start_time or end_time > video_info.duration_seconds:
            # If still invalid after adjustments, return error
            return False, f"Invalid time range after adjustments: {start_time:.2f}s to {end_time:.2f}s", start_time, end_time

        # Log if adjustments were made
        if original_start != start_time or original_end != end_time:
            logger.info(f"Time range adjusted from {original_start:.2f}s-{original_end:.2f}s "
                        f"to {start_time:.2f}s-{end_time:.2f}s")
        else:
            logger.info(f"Time validation passed: {start_time:.2f}s to {end_time:.2f}s "
                        f"(within video duration: {video_info.duration_seconds:.2f}s)")

        return True, f"Valid time range: {start_time:.2f}s to {end_time:.2f}s", start_time, end_time

    def _time_to_frames(self, time_seconds: float, fps: float) -> int:
        """
        Convert time in seconds to frame number

        Args:
            time_seconds (float): Time in seconds
            fps (float): Frames per second

        Returns:
            int: Frame number
        """
        return int(time_seconds * fps)

    def _validate_output_format(self, file_format: str) -> bool:
        """
        Validate if the output file format is supported by OpenCV

        Args:
            file_format (str): File format/extension (e.g., 'mp4', 'avi')

        Returns:
            bool: True if format is valid, False otherwise
        """
        valid_formats = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']
        return file_format.lower() in valid_formats

    def _generate_output_filename(self, video_path: str, suffix: str, file_format: str) -> str:
        """
        Generate output filename with timestamp

        Args:
            video_path (str): Original video path
            suffix (str): Suffix to add to filename
            file_format (str): Output file format

        Returns:
            str: Generated output file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"{base_name}_{suffix}_{timestamp}.{file_format}"
        return os.path.join(self.output_dir, output_name)

    def cut_video(self, video_path: str, start_time: float, end_time: float,
                  file_format: str = "mp4") -> Optional[str]:
        """
        Cut video based on time range

        Args:
            video_path (str): Path to the input video file
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            file_format (str): Output file format (default: 'mp4')

        Returns:
            str: Path to the output video file, or None if failed
        """
        # Validate and adjust time range
        is_valid, message, adjusted_start, adjusted_end = self.time_checker(
            video_path, start_time, end_time
        )

        if not is_valid:
            logger.error(f"Time validation failed: {message}")
            return None

        # Use adjusted time values
        return self._cut_with_adjusted_time(
            video_path, adjusted_start, adjusted_end, file_format
        )

    def _cut_with_adjusted_time(self, video_path: str, start_time: float, end_time: float,
                                file_format: str = "mp4") -> Optional[str]:
        """
        Cut video using adjusted time values

        Args:
            video_path (str): Path to the input video file
            start_time (float): Adjusted start time in seconds
            end_time (float): Adjusted end time in seconds
            file_format (str): Output file format

        Returns:
            str: Path to the output video file, or None if failed
        """
        # Get video information
        video_info = _get_video_info(video_path)
        if not video_info:
            return None

        # Convert adjusted time to frames
        start_frame = self._time_to_frames(start_time, video_info.fps)
        end_frame = self._time_to_frames(end_time, video_info.fps)

        logger.info(f"Cutting from time {start_time:.2f}s to {end_time:.2f}s "
                    f"(frames {start_frame} to {end_frame})")

        # Cut using frame-based method
        return self._cut_video_by_frames(
            video_path, start_frame, end_frame, file_format, video_info
        )

    def cut_video_with_frames(self, video_path: str, start_frame: int,
                              end_frame: int, file_format: str = "mp4") -> Optional[str]:
        """
        Cut video based on frame numbers

        Args:
            video_path (str): Path to the input video file
            start_frame (int): Start frame number (0-indexed)
            end_frame (int): End frame number (exclusive)
            file_format (str): Output file format (default: 'mp4')

        Returns:
            str: Path to the output video file, or None if failed
        """
        # Validate output format
        if not self._validate_output_format(file_format):
            logger.error(f"Unsupported output format: {file_format}")
            return None

        # Get video information
        video_info = _get_video_info(video_path)
        if not video_info:
            return None

        # Validate frame range
        if start_frame < 0:
            logger.error(f"Start frame ({start_frame}) cannot be negative")
            return None

        if end_frame <= start_frame:
            logger.error(f"End frame ({end_frame}) must be greater than start frame ({start_frame})")
            return None

        if end_frame > video_info.total_frames:
            logger.error(f"End frame ({end_frame}) exceeds total frames ({video_info.total_frames})")
            return None

        logger.info(f"Cutting from frame {start_frame} to {end_frame} "
                    f"(approximately {start_frame / video_info.fps:.2f}s to {end_frame / video_info.fps:.2f}s)")

        return self._cut_video_by_frames(
            video_path, start_frame, end_frame, file_format, video_info
        )

    def _cut_video_by_frames(self, video_path: str, start_frame: int, end_frame: int,
                             file_format: str, video_info: VideoInfo) -> Optional[str]:
        """
        Internal method to cut video by frame range

        Args:
            video_path (str): Path to input video
            start_frame (int): Start frame number
            end_frame (int): End frame number
            file_format (str): Output file format
            video_info (VideoInfo): Video information

        Returns:
            str: Path to output video, or None if failed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return None

        # Generate output filename
        output_path = self._generate_output_filename(
            video_path, f"cut_{start_frame}_{end_frame}", file_format
        )

        # Define codec for output
        if file_format.lower() == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif file_format.lower() == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default to mp4v

        # Create VideoWriter
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            video_info.fps,
            (video_info.width, video_info.height)
        )

        if not out.isOpened():
            logger.error(f"Cannot create output video file: {output_path}")
            cap.release()
            return None

        try:
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_count = 0
            total_frames_to_process = end_frame - start_frame

            logger.info(f"Processing {total_frames_to_process} frames...")

            while frame_count < total_frames_to_process:
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"Stopped early at frame {start_frame + frame_count}")
                    break

                # Write frame to output
                out.write(frame)
                frame_count += 1

                # Log progress every 10%
                if total_frames_to_process > 0:
                    progress = (frame_count / total_frames_to_process) * 100
                    if frame_count % max(1, total_frames_to_process // 10) == 0:
                        logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames_to_process})")

            logger.info(f"Successfully cut {frame_count} frames to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error during video cutting: {str(e)}")
            # Clean up partially written file
            if os.path.exists(output_path):
                os.remove(output_path)
            return None

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()

    def cut_video_multiple(self, video_path: str, time_ranges: List[Tuple[float, float]],
                           file_format: str = "mp4") -> List[str]:
        """
        Cut multiple segments from video based on time ranges

        Args:
            video_path (str): Path to the input video file
            time_ranges (List[Tuple[float, float]]): List of (start_time, end_time) tuples
            file_format (str): Output file format (default: 'mp4')

        Returns:
            List[str]: List of paths to output video files
        """
        output_files = []

        for i, (start_time, end_time) in enumerate(time_ranges):
            logger.info(f"Processing segment {i + 1}/{len(time_ranges)}: {start_time}s to {end_time}s")

            output_path = self.cut_video(video_path, start_time, end_time, file_format)

            if output_path:
                output_files.append(output_path)
                logger.info(f"Segment {i + 1} saved to: {output_path}")
            else:
                logger.error(f"Failed to process segment {i + 1}: {start_time}s to {end_time}s")

        logger.info(f"Completed processing {len(output_files)}/{len(time_ranges)} segments")
        return output_files

    def cut_video_multiple_with_frames(self, video_path: str,
                                       frame_ranges: List[Tuple[int, int]],
                                       file_format: str = "mp4") -> List[str]:
        """
        Cut multiple segments from video based on frame ranges

        Args:
            video_path (str): Path to the input video file
            frame_ranges (List[Tuple[int, int]]): List of (start_frame, end_frame) tuples
            file_format (str): Output file format (default: 'mp4')

        Returns:
            List[str]: List of paths to output video files
        """
        output_files = []

        for i, (start_frame, end_frame) in enumerate(frame_ranges):
            logger.info(f"Processing segment {i + 1}/{len(frame_ranges)}: "
                        f"frames {start_frame} to {end_frame}")

            output_path = self.cut_video_with_frames(
                video_path, start_frame, end_frame, file_format
            )

            if output_path:
                output_files.append(output_path)
                logger.info(f"Segment {i + 1} saved to: {output_path}")
            else:
                logger.error(f"Failed to process segment {i + 1}: frames {start_frame} to {end_frame}")

        logger.info(f"Completed processing {len(output_files)}/{len(frame_ranges)} segments")
        return output_files
