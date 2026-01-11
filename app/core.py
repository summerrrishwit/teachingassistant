"""
核心工具模块 - 常量、日志、单例模式
"""

import logging
from pathlib import Path
from typing import Optional


class RAGConstants:
    """RAG系统相关常量"""
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    DEFAULT_TOP_K = 3
    DEFAULT_SCORE_THRESHOLD = 1.5
    MIN_SEGMENT_LENGTH = 50


class PathConstants:
    """路径相关常量"""
    RUNTIME_DIR = Path("runtime")
    VIDEO_PATH = RUNTIME_DIR / "uploaded_video.mp4"
    FRAME_DIR = RUNTIME_DIR / "frames"
    FAISS_INDEX_DIR = RUNTIME_DIR / "faiss_index"
    LOG_DIR = RUNTIME_DIR / "logs"


# ============================================================================
# 单例模式
# ============================================================================

class Singleton:
    """单例模式基类"""
    _instances = {}
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls)
        return cls._instances[cls]


# ============================================================================
# 日志系统
# ============================================================================

def setup_logger(
    name: str, 
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    设置并配置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# 创建默认日志记录器
default_logger = setup_logger(
    "video_assistant",
    log_file=PathConstants.LOG_DIR / "app.log"
)
