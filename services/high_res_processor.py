# services/high_res_processor.py
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from config import DEFAULT_TILE_SIZE, DEFAULT_STRIP_RATIO, TILE_SIZE_RANGE, STRIP_RATIO_RANGE
from utils import get_logger

logger = get_logger(__name__)

class HighResProcessor:
    def __init__(self, model: YOLO):
        self.model = model  # 外部传入已加载的模型（避免重复加载）
        self.tile_size = DEFAULT_TILE_SIZE
        self.strip_ratio = DEFAULT_STRIP_RATIO
        self.strip = int(self.tile_size * self.strip_ratio)
        self.step = self.tile_size - self.strip

    def update_params(self, tile_size: int, strip_ratio: float) -> None:
        """更新切片参数（带范围校验）"""
        # 限制切片大小在合理范围
        self.tile_size = max(TILE_SIZE_RANGE[0], min(tile_size, TILE_SIZE_RANGE[1]))
        # 限制重叠比例在合理范围
        self.strip_ratio = max(STRIP_RATIO_RANGE[0], min(strip_ratio, STRIP_RATIO_RANGE[1]))
        # 重新计算重叠区域和步长
        self.strip = int(self.tile_size * self.strip_ratio)
        self.step = self.tile_size - self.strip
        logger.info(f"更新高分辨率参数: tile_size={self.tile_size}, strip_ratio={self.strip_ratio}")

    def slice_image(self, image: Image.Image) -> list[tuple[Image.Image, int, int]]:
        """将图像切片为子图，返回（子图, x_start, y_start）"""
        img_w, img_h = image.size
        logger.info(f"原始图像尺寸: {img_w}x{img_h}，开始切片")
        
        x_steps = self._calc_steps(img_w)
        y_steps = self._calc_steps(img_h)
        tiles = []
        
        for y in range(y_steps):
            for x in range(x_steps):
                x_start = x * self.step
                y_start = y * self.step
                x_end = min(x_start + self.tile_size, img_w)
                y_end = min(y_start + self.tile_size, img_h)
                
                # 最后一个切片不足时，向左/上偏移以保证尺寸
                if x == x_steps - 1 and (x_end - x_start) < self.tile_size:
                    x_start = img_w - self.tile_size
                    x_end = img_w
                if y == y_steps - 1 and (y_end - y_start) < self.tile_size:
                    y_start = img_h - self.tile_size
                    y_end = img_h
                
                sub_img = image.crop((x_start, y_start, x_end, y_end))
                tiles.append((sub_img, x_start, y_start))
        
        logger.info(f"切片完成，共生成 {len(tiles)} 个子图")
        return tiles

    def detect_tiles(self, tiles: list[tuple[Image.Image, int, int]], roi_offset: tuple[int, int]) -> list[dict]:
        """检测所有子图并转换坐标到原图（含ROI偏移）"""
        results = []
        for sub_img, x_start, y_start in tiles:
            # 转换为OpenCV格式（BGR）
            cv_img = cv2.cvtColor(np.array(sub_img), cv2.COLOR_RGB2BGR)
            det_results = self.model.predict(source=cv_img)
            
            for result in det_results:
                if not hasattr(result, 'obb'):
                    continue  # 跳过无旋转框的结果
                for obb in result.obb:
                    # 提取旋转框参数（中心坐标、宽高、角度）
                    rbox = obb.xywhr[0].tolist()
                    # 转换坐标到原图（子图偏移 + ROI偏移）
                    rbox[0] += x_start + roi_offset[0]  # x_center
                    rbox[1] += y_start + roi_offset[1]  # y_center
                    
                    results.append({
                        "class_name": result.names[int(obb.cls)],
                        "class_id": int(obb.cls),
                        "confidence": float(obb.conf),
                        "rbox": rbox
                    })
        return results

    def _calc_steps(self, dimension: int) -> int:
        """计算某一维度（宽/高）的切片数量"""
        if dimension <= self.tile_size:
            return 1
        # 向上取整公式：(总数 - 单次量 + 步长 - 1) // 步长
        return (dimension - self.tile_size + self.step - 1) // self.step