# services/detector.py
from ultralytics import YOLO
from PIL import Image
import cv2
import torch
import numpy as np
from .high_res_processor import HighResProcessor
from config import *
from utils import get_logger
from ultralytics.utils.nms import TorchNMS
from ultralytics.utils.metrics import batch_probiou
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
        # 新增：对合并后的结果执行NMS去重
        if not results:
            return results, len(tiles)  # 无结果时直接返回
        
        # 1. 转换结果为张量格式
        # 提取旋转框 [x_center, y_center, width, height, angle]、置信度、类别
        rboxes = []
        scores = []
        classes = []
        for res in results:
            rboxes.append(res["rbox"])  # 旋转框参数
            scores.append(res["confidence"])  # 置信度
            classes.append(res["class_id"])  # 类别ID
        
        # 转换为torch张量
        rboxes = torch.tensor(rboxes, dtype=torch.float32)  # shape: [N, 5]
        scores = torch.tensor(scores, dtype=torch.float32)  # shape: [N]
        classes = torch.tensor(classes, dtype=torch.int64)  # shape: [N]
        
        # 2. 为不同类别添加偏移量（确保NMS仅在同类间生效）
        # 偏移量 = 类别ID * 最大可能坐标值（避免不同类别的框重叠判断）
        max_wh = torch.max(rboxes[:, :2]) * 2  # 取中心坐标最大值的2倍作为偏移基数
        c = classes.float() * max_wh  # 类别偏移量
        rboxes_with_cls = rboxes.clone()
        rboxes_with_cls[:, :2] += c.unsqueeze(1)  # 将偏移量添加到x_center和y_center
        
        # 3. 执行旋转框NMS（使用Ultralytics原生实现）
        # IoU阈值参考val.py中的0.3，可根据场景调整
        keep_indices = TorchNMS.fast_nms(
            boxes=rboxes_with_cls,
            scores=scores,
            iou_threshold=0.3,
            iou_func=batch_probiou  # 旋转框IoU计算函数
        )
        
        # 4. 过滤重复结果
        filtered_results = [results[i] for i in keep_indices.numpy()]
        
        return filtered_results, len(tiles)