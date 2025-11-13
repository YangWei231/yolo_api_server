# services/detector.py
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from .high_res_processor import HighResProcessor
from config import *
from utils import get_logger

logger = get_logger(__name__)

class YoloDetector:
    def __init__(self):
        # 加载YOLO模型
        self.model = self._load_model()
        # 初始化高分辨率处理器（复用模型实例）
        self.high_res_processor = HighResProcessor(self.model)

    def _load_model(self) -> YOLO:
        """加载YOLO模型"""
        try:
            model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"成功加载模型: {YOLO_MODEL_PATH}")
            return model
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise  # 启动时失败应终止服务

    def detect_normal(self, image: Image.Image, roi_offset: tuple[int, int] = (0, 0)) -> list[dict]:
        """常规检测（非切片模式）"""
        try:
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = self.model.predict(source=cv_img)
            
            det_results = []
            for result in results:
                if not hasattr(result, 'obb'):
                    continue
                for obb in result.obb:
                    rbox = obb.xywhr[0].tolist()
                    # 叠加ROI偏移
                    rbox[0] += roi_offset[0]
                    rbox[1] += roi_offset[1]
                    
                    det_results.append({
                        "class_name": result.names[int(obb.cls)],
                        "class_id": int(obb.cls),
                        "confidence": float(obb.conf),
                        "rbox": rbox
                    })
            return det_results
        except Exception as e:
            logger.error(f"常规检测失败: {str(e)}")
            raise

    def detect_high_res(self, image: Image.Image, roi_offset: tuple[int, int] = (0, 0), 
                       tile_size: int = DEFAULT_TILE_SIZE, strip_ratio: float = DEFAULT_STRIP_RATIO) -> tuple[list[dict], int]:
        """高分辨率检测（切片模式）"""
        # 更新切片参数
        self.high_res_processor.update_params(tile_size, strip_ratio)
        # 切片
        tiles = self.high_res_processor.slice_image(image)
        # 检测并合并结果
        results = self.high_res_processor.detect_tiles(tiles, roi_offset)
        return results, len(tiles)