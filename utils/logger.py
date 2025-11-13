# utils/logger.py
import logging
from config import LOG_LEVEL

def get_logger(name: str) -> logging.Logger:
    """获取配置好的日志实例"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # 避免重复添加处理器
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger