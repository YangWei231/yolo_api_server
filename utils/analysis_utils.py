# utils/analysis_utils.py
from typing import List, Dict
from schemas import DetectionResult
from .logger import get_logger

logger = get_logger(__name__)

def count_detection_classes(results: List[DetectionResult]) -> Dict[str, int]:
    """统计检测结果中每个类别的数量"""
    class_counts = {}
    for result in results:
        class_name = result.class_name
        # 累加计数
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    logger.info(f"分类统计完成: {class_counts}")
    return class_counts