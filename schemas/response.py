# schemas/response.py
from pydantic import BaseModel
from typing import List, Optional,Dict

class DetectionResult(BaseModel):
    """单个目标的旋转框检测结果"""
    class_name: str
    class_id: int
    confidence: float
    rbox: List[float]  # [x_center, y_center, width, height, angle（弧度）]

class DetectionResponse(BaseModel):
    """检测接口统一响应模型"""
    success: bool
    message: str = ""
    results: List[DetectionResult] = []
    roi_used: bool = False  # 是否使用ROI
    high_res_used: bool = False  # 是否使用高分辨率模式
    tiles_count: int = 0    # 切片数量（仅高分辨率模式有效）
    class_counts: Optional[Dict[str, int]] = None  # 新增：每个类别的数量统计（键为class_name，值为数量）