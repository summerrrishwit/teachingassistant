import cv2
import os
from pathlib import Path
from typing import List
from PIL import Image
import numpy as np

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
        List of saved frame paths
    """
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
    return frame_paths

