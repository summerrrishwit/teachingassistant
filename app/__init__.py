"""
AI Video Assistant Application
"""

from .core import (
    RAGConstants,
    PathConstants,
    Singleton,
    setup_logger,
    default_logger,
)

__all__ = [
    'RAGConstants',
    'PathConstants',
    # 工具
    'Singleton',
    'setup_logger',
    'default_logger',
]
