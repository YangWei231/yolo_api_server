# utils/image_utils.py
from PIL import Image
from fastapi import HTTPException
from .logger import get_logger

logger = get_logger(__name__)

def crop_roi(image: Image.Image, roi: list[int]) -> tuple[Image.Image, tuple[int, int]]:
    """裁剪ROI区域，返回裁剪后的图像和原始坐标偏移（x1, y1）"""
    try:
        if len(roi) != 4:
            raise ValueError("ROI必须为[x1, y1, x2, y2]（4个整数）")
        x1, y1, x2, y2 = roi
        img_w, img_h = image.size
        
        # 边界校验（确保坐标在图像范围内）
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(x1 + 1, min(x2, img_w))  # 避免x2 <= x1
        y2 = max(y1 + 1, min(y2, img_h))  # 避免y2 <= y1
        
        cropped = image.crop((x1, y1, x2, y2))
        return cropped, (x1, y1)
    except Exception as e:
        logger.error(f"ROI裁剪失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"ROI裁剪失败: {str(e)}")