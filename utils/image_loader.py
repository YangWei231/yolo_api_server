# utils/image_loader.py
import base64
import io
import requests
from PIL import Image
from fastapi import HTTPException
from .logger import get_logger

logger = get_logger(__name__)

def load_from_base64(base64_str: str) -> Image.Image:
    """从Base64字符串加载图像"""
    try:
        # 去除前缀（如data:image/png;base64,）
        if base64_str.startswith(('data:image/', 'data:application/')):
            base64_str = base64_str.split(',')[1]
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        logger.error(f"Base64图像加载失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"无效的Base64图像: {str(e)}")

def load_from_url(url: str) -> Image.Image:
    """从URL加载图像"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 抛出HTTP错误（如404）
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        logger.error(f"URL图像加载失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"URL加载失败: {str(e)}")

def load_from_file(file_content: bytes) -> Image.Image:
    """从文件字节流加载图像（用于文件上传）"""
    try:
        return Image.open(io.BytesIO(file_content))
    except Exception as e:
        logger.error(f"文件图像加载失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"无效的图像文件: {str(e)}")