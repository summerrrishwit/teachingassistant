import cv2
import os
from pathlib import Path
from typing import List
from PIL import Image

# 注意：此文件不依赖其他app模块，保持独立

def save_uploaded_video(uploaded_file, save_path: Path):
    """Save uploaded video to disk"""
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

def extract_frames_around(video_path: Path, timestamp: float, frame_dir: Path, window: int = 2, fps: int = 1) -> List[Path]:
    """
    Extract frames from [timestamp - window, timestamp + window] at `fps` and save as JPGs.

    Args:
        video_path: Path to input video
        timestamp: Central time in seconds
        frame_dir: Output folder for frames
        window: Seconds before and after timestamp
        fps: Frames per second to sample

    Returns:
        List of saved frame paths (may be shorter if frames are missing)
    """
    os.makedirs(frame_dir, exist_ok=True)
    # Remove old QA frames but keep summary frames
    for file in os.listdir(frame_dir):
        if file.startswith("frame_"):
            try:
                os.remove(os.path.join(frame_dir, file))
            except OSError:
                pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    frame_paths = []
    start_time = max(0, timestamp - window)
    end_time = timestamp + window

    current = start_time
    frame_count = 0

    while current <= end_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, current * 1000)
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_img = Image.fromarray(frame_rgb)
            frame_path = f"{frame_dir}/frame_{frame_count}.jpg"
            frame_img.save(frame_path, "JPEG")
            frame_paths.append(frame_path)
            frame_count += 1
        current += 1.0 / fps

    cap.release()
    if not frame_paths:
        raise RuntimeError("No frames could be extracted around the given timestamp")
    return frame_paths

def extract_key_frames_for_summary(video_path: Path, frame_dir: Path, num_frames: int = 5) -> List[Path]:
    """
    Extract key frames from the entire video for comprehensive analysis.
    
    Args:
        video_path: Path to input video
        frame_dir: Output folder for frames
        num_frames: Number of key frames to extract
        
    Returns:
        List of saved frame paths
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_paths = []
    
    # Clear existing frames
    for file in os.listdir(frame_dir):
        if file.endswith('.jpg'):
            os.remove(os.path.join(frame_dir, file))
    
    # Extract frames at regular intervals
    for i in range(num_frames):
        frame_number = int((i / (num_frames - 1)) * (total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_img = Image.fromarray(frame_rgb)
            frame_path = f"{frame_dir}/summary_frame_{i}.jpg"
            frame_img.save(frame_path, "JPEG")
            frame_paths.append(frame_path)
    
    cap.release()
    return frame_paths
