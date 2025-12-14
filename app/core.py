"""
核心工具模块 - 整合所有基础工具类
包含：异常、常量、装饰器、日志、验证器、单例模式
"""

import os
import logging
from pathlib import Path
from functools import wraps
from typing import Callable, Any, Optional, BinaryIO, List

import streamlit as st

# ============================================================================
# 异常类
# ============================================================================

class VideoProcessingError(Exception):
    """视频处理相关错误"""
    pass


class TranscriptionError(Exception):
    """转录相关错误"""
    pass


class LLMServiceError(Exception):
    """LLM服务相关错误"""
    pass


class RAGError(Exception):
    """RAG系统相关错误"""
    pass


class ConfigurationError(Exception):
    """配置相关错误"""
    pass


class FileValidationError(Exception):
    """文件验证相关错误"""
    pass


# ============================================================================
# 常量定义
# ============================================================================

class VideoConstants:
    """视频处理相关常量"""
    DEFAULT_WINDOW_SECONDS = 2
    DEFAULT_FPS = 1
    MAX_FRAMES_PER_EXTRACTION = 10
    MAX_VIDEO_SIZE_MB = 200
    ALLOWED_EXTENSIONS = {'.mp4', '.webm', '.mov'}
    SUPPORTED_FORMATS = ['mp4', 'webm', 'mov']


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


class ModelConstants:
    """模型相关常量"""
    DEFAULT_OLLAMA_MODEL = "gemma3:4b"
    DEFAULT_WHISPER_MODEL = "base"
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class UIConstants:
    """UI相关常量"""
    MAX_CONVERSATION_HISTORY = 10
    MAX_EXPORT_ITEMS = 100
    PROGRESS_UPDATE_INTERVAL = 0.1


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


# ============================================================================
# 装饰器
# ============================================================================

def streamlit_error_handler(func: Callable) -> Callable:
    """
    统一错误处理装饰器
    
    自动捕获异常并在Streamlit中显示友好的错误信息
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except FileValidationError as e:
            default_logger.error(f"文件验证错误: {e}", exc_info=True)
            st.error(f"❌ 文件验证失败: {e}")
            return None
        except VideoProcessingError as e:
            default_logger.error(f"视频处理错误: {e}", exc_info=True)
            st.error(f"❌ 视频处理失败: {e}")
            return None
        except TranscriptionError as e:
            default_logger.error(f"转录错误: {e}", exc_info=True)
            st.error(f"❌ 视频转录失败: {e}")
            return None
        except LLMServiceError as e:
            default_logger.error(f"LLM服务错误: {e}", exc_info=True)
            st.error(f"❌ AI服务错误: {e}")
            return None
        except RAGError as e:
            default_logger.warning(f"RAG错误: {e}", exc_info=True)
            st.warning(f"⚠️ 智能检索错误: {e}")
            return None
        except FileNotFoundError as e:
            default_logger.error(f"文件未找到: {e}", exc_info=True)
            st.error(f"❌ 文件未找到: {e}")
            return None
        except ValueError as e:
            default_logger.warning(f"值错误: {e}")
            st.warning(f"⚠️ {e}")
            return None
        except Exception as e:
            default_logger.exception(f"未预期的错误: {e}")
            st.error(f"❌ 发生未知错误: {e}\n\n请查看日志获取详细信息。")
            return None
    return wrapper


def log_execution_time(func: Callable) -> Callable:
    """
    记录函数执行时间的装饰器
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        import time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            default_logger.debug(f"{func.__name__} 执行时间: {execution_time:.2f}秒")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            default_logger.error(f"{func.__name__} 执行失败 (耗时 {execution_time:.2f}秒): {e}")
            raise
    return wrapper


# ============================================================================
# 验证器
# ============================================================================

def validate_video_file(uploaded_file: BinaryIO) -> bool:
    """
    验证上传的视频文件
    
    Args:
        uploaded_file: 上传的文件对象
    
    Raises:
        FileValidationError: 如果文件验证失败
    
    Returns:
        True if valid
    """
    # 检查扩展名
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in VideoConstants.ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"不支持的文件类型: {file_ext}。"
            f"支持的格式: {', '.join(VideoConstants.ALLOWED_EXTENSIONS)}"
        )
    
    # 检查文件大小
    uploaded_file.seek(0, 2)  # 移动到文件末尾
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)  # 重置到开头
    
    max_size_bytes = VideoConstants.MAX_VIDEO_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        raise FileValidationError(
            f"文件过大: {file_size / 1024 / 1024:.2f}MB。"
            f"最大允许大小: {VideoConstants.MAX_VIDEO_SIZE_MB}MB"
        )
    
    if file_size == 0:
        raise FileValidationError("文件为空")
    
    return True


def validate_timestamp(timestamp_str: str) -> float:
    """
    验证时间戳格式
    
    Args:
        timestamp_str: 时间戳字符串 (HH:MM:SS, MM:SS, 或 SS)
    
    Returns:
        时间戳（秒）
    
    Raises:
        ValueError: 如果时间戳格式无效
    """
    timestamp_str = timestamp_str.strip()
    parts = timestamp_str.split(":")
    
    try:
        if len(parts) == 3:
            h, m, s = map(int, parts)
            if h < 0 or m < 0 or s < 0 or m >= 60 or s >= 60:
                raise ValueError("时间值超出有效范围")
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            if m < 0 or s < 0 or s >= 60:
                raise ValueError("时间值超出有效范围")
            return m * 60 + s
        elif len(parts) == 1:
            seconds = int(parts[0])
            if seconds < 0:
                raise ValueError("时间值不能为负数")
            return seconds
        else:
            raise ValueError("时间戳格式无效")
    except ValueError as e:
        if "invalid literal" in str(e).lower():
            raise ValueError("时间戳包含非数字字符")
        raise

