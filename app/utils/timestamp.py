"""
时间戳工具函数
"""
from typing import Optional


def parse_timestamp(input_str: str) -> Optional[int]:
    """Convert HH:MM:SS, MM:SS or SS to seconds."""
    input_str = input_str.strip()
    parts = input_str.split(":")
    try:
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        if len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        if len(parts) == 1:
            return int(parts[0])
        return None
    except ValueError:
        return None

