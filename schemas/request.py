# schemas/request.py
from pydantic import BaseModel
from typing import Optional, List

class DetectionRequest(BaseModel):
    """检测请求模型（支持Base64/URL输入、ROI、高分辨率模式）"""
    image_data: Optional[str] = None  # Base64编码图像
    image_url: Optional[str] = None   # 图像URL
    roi: Optional[List[int]] = None   # 感兴趣区域 [x1, y1, x2, y2]
    high_res: bool = False            # 是否启用高分辨率切片模式
    tile_size: int = 1024             # 切片大小（仅高分辨率模式有效）
    strip_ratio: float = 0.5          # 重叠比例（仅高分辨率模式有效）