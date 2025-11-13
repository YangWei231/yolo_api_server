# 满足高分辨率图像切片划窗检测需求，可将超高分辨率目标切片后分别通过YOLO进行检测而后输出检测结果。
# 切片之间可能有重复检测，因此还需要做一个NMS。

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import io
import base64
import requests
from ultralytics import YOLO
import os
import logging
from typing import Optional, List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="YOLO OBB旋转框检测API", 
              description="支持超高分辨率图像切片检测的YOLO旋转框检测服务，支持ROI区域裁剪")

class HighResImageProcessor:
    def __init__(self, model, tile_size=1024, strip_ratio=0.5):
        """
        初始化高分辨率图像处理工具
        :param model: 已加载的YOLO模型
        :param tile_size: 子图尺寸（n x n）
        :param strip_ratio: 重叠比例（strip = tile_size * strip_ratio）
        """
        self.model = model
        self.tile_size = tile_size
        self.strip = int(tile_size * strip_ratio)  # 重叠区域大小
        self.step = tile_size - self.strip  # 每次滑动的步长

    def slice_image(self, image: Image.Image):
        """
        将高分辨率图像切片为子图
        :return: 子图列表及对应的坐标偏移量 [(sub_image, x_start, y_start), ...]
        """
        img_width, img_height = image.size
        logger.info(f"原始图像分辨率：{img_height}x{img_width}")
        tiles = []

        # 计算x和y方向的切片数量
        x_steps = self._calculate_steps(img_width)
        y_steps = self._calculate_steps(img_height)

        for y in range(y_steps):
            for x in range(x_steps):
                # 计算当前子图的起始坐标
                x_start = x * self.step
                y_start = y * self.step

                # 确保最后一个子图不超出原图范围
                x_end = min(x_start + self.tile_size, img_width)
                y_end = min(y_start + self.tile_size, img_height)

                # 如果是最后一个子图且尺寸不足，调整起始坐标
                if x == x_steps - 1 and x_end - x_start < self.tile_size:
                    x_start = img_width - self.tile_size
                    x_end = img_width

                if y == y_steps - 1 and y_end - y_start < self.tile_size:
                    y_start = img_height - self.tile_size
                    y_end = img_height

                # 裁剪子图
                sub_img = image.crop((x_start, y_start, x_end, y_end))
                tiles.append((sub_img, x_start, y_start))

        return tiles

    def _calculate_steps(self, dimension):
        """计算某个维度（宽/高）需要的切片数量"""
        if dimension <= self.tile_size:
            return 1
        return (dimension - self.tile_size + self.step - 1) // self.step  # 向上取整

    def detect_tiles(self, tiles, roi_offset=(0, 0)):
        """
        检测所有子图并转换坐标到原图
        :return: 合并后的检测结果列表
        """
        all_results = []

        for sub_img, x_start, y_start in tiles:
            # 转换为numpy数组进行检测
            cv_image = cv2.cvtColor(np.array(sub_img), cv2.COLOR_RGB2BGR)
            results = self.model.predict(source=cv_image)

            for result in results:
                if not hasattr(result, 'obb'):  # 没有检测到旋转框
                    continue

                # 提取旋转框参数：xywhr（中心坐标、宽、高、旋转角度（弧度））
                for obb in result.obb:
                    rbox = obb.xywhr[0].tolist()
                    
                    # 转换坐标到子图在原图中的位置
                    rbox[0] += x_start + roi_offset[0]  # x_center
                    rbox[1] += y_start + roi_offset[1]  # y_center
                    
                    all_results.append({
                        "class_name": result.names[int(obb.cls)],
                        "class_id": int(obb.cls),
                        "confidence": float(obb.conf),
                        "rbox": rbox
                    })

        return all_results

    def process_high_res_image(self, image: Image.Image, roi_offset=(0, 0)):
        """完整处理流程：切片->检测->合并结果"""
        logger.info("开始处理高分辨率图像")
        tiles = self.slice_image(image)
        logger.info(f"图像切片完成，共生成 {len(tiles)} 个子图")
        results = self.detect_tiles(tiles, roi_offset)
        logger.info(f"所有子图检测完成，共检测到 {len(results)} 个目标")
        return results

# 加载YOLO OBB模型
try:
    model_path = "/home/wsl/Workspace/remote_qwen_lamp/HF_process_w_yolo12/yolo11s-obb.pt"
    model = YOLO(model_path)
    # 初始化高分辨率处理器
    high_res_processor = HighResImageProcessor(model)
    logger.info(f"成功加载YOLO OBB模型和高分辨率处理器: {model_path}")
except Exception as e:
    logger.error(f"加载模型失败: {str(e)}")
    raise

# 定义请求模型
class DetectionRequest(BaseModel):
    image_data: Optional[str] = None  # Base64编码的图像数据
    image_url: Optional[str] = None   # 图像URL
    roi: Optional[List[int]] = None   # 感兴趣区域 [x1, y1, x2, y2]，可选
    high_res: bool = False            # 是否启用高分辨率模式
    tile_size: int = 1024             # 子图尺寸，仅在高分辨率模式下有效
    strip_ratio: float = 0.5          # 重叠比例，仅在高分辨率模式下有效

# 定义检测结果模型
class DetectionResult(BaseModel):
    """单个目标的旋转框检测结果"""
    class_name: str
    class_id: int
    confidence: float
    # 旋转框参数：[x_center, y_center, width, height, angle]（角度单位：弧度）
    rbox: List[float]  # 坐标相对于原始图像

class DetectionResponse(BaseModel):
    success: bool
    message: str = ""
    results: List[DetectionResult] = []
    roi_used: bool = False  # 是否使用了ROI区域
    high_res_used: bool = False  # 是否使用了高分辨率模式
    tiles_count: int = 0    # 切片数量，仅在高分辨率模式下有效

# 工具函数：加载图像
def load_image_from_base64(base64_str: str) -> Image.Image:
    try:
        if base64_str.startswith(('data:image/', 'data:application/')):
            base64_str = base64_str.split(',')[1]
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"从Base64加载图像失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"无效的图像数据: {str(e)}")

def load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        logger.error(f"从URL加载图像失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"无法从URL加载图像: {str(e)}")

# 工具函数：裁剪图像
def crop_image(image: Image.Image, roi: List[int]) -> tuple[Image.Image, tuple[int, int]]:
    try:
        if len(roi) != 4:
            raise ValueError("ROI必须包含4个整数: [x1, y1, x2, y2]")
        x1, y1, x2, y2 = roi
        width, height = image.size
        # 确保坐标有效且在图像范围内
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image, (x1, y1)  # 返回裁剪图和原始坐标偏移
    except Exception as e:
        logger.error(f"裁剪图像失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"裁剪图像失败: {str(e)}")

# 处理旋转框检测结果（常规模式）
def process_obb_detection(image: Image.Image, roi_offset: tuple = (0, 0)) -> List[DetectionResult]:
    """处理旋转框检测，返回结果（坐标转换为原始图像）"""
    try:
        # 转换为OpenCV格式
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 执行OBB检测
        results = model.predict(source=cv_image)
        
        detection_results = []
        for result in results:
            if not hasattr(result, 'obb'):
                continue  # 无旋转框结果则跳过
            for obb in result.obb:
                # 提取旋转框参数：xywhr
                rbox = obb.xywhr[0].tolist()
                
                # 如果使用了ROI，将中心坐标转换回原始图像
                if roi_offset:
                    rbox[0] += roi_offset[0]  # x_center += ROI的x1偏移
                    rbox[1] += roi_offset[1]  # y_center += ROI的y1偏移
                
                detection_results.append(DetectionResult(
                    class_name=result.names[int(obb.cls)],
                    class_id=int(obb.cls),
                    confidence=float(obb.conf),
                    rbox=rbox
                ))
        
        return detection_results
    except Exception as e:
        logger.error(f"旋转框检测处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检测处理失败: {str(e)}")

# 检测接口
@app.post("/detect", response_model=DetectionResponse, summary="旋转框目标检测接口")
async def detect(request: DetectionRequest):
    try:
        # 加载图像
        if request.image_data:
            image = load_image_from_base64(request.image_data)
        elif request.image_url:
            image = load_image_from_url(request.image_url)
        else:
            raise HTTPException(status_code=400, detail="必须提供image_data或image_url")
        
        roi_used = False
        roi_offset = (0, 0)
        high_res_used = request.high_res
        tiles_count = 0
        
        # 裁剪ROI（如果提供）
        if request.roi:
            image, roi_offset = crop_image(image, request.roi)
            roi_used = True
        
        # 执行检测
        if high_res_used:
            # 更新高分辨率处理器参数
            high_res_processor.tile_size = max(256, min(request.tile_size, 4096))  # 限制合理范围
            high_res_processor.strip_ratio = max(0.1, min(request.strip_ratio, 0.9))  # 限制合理范围
            high_res_processor.strip = int(high_res_processor.tile_size * high_res_processor.strip_ratio)
            high_res_processor.step = high_res_processor.tile_size - high_res_processor.strip
            
            # 切片检测
            tiles = high_res_processor.slice_image(image)
            tiles_count = len(tiles)
            raw_results = high_res_processor.detect_tiles(tiles, roi_offset)
            
            # 转换为DetectionResult对象
            results = [DetectionResult(**item) for item in raw_results]
        else:
            # 常规检测
            results = process_obb_detection(image, roi_offset)
        breakpoint()
        return DetectionResponse(
            success=True,
            message=f"成功检测到{len(results)}个目标（{'高分辨率切片' if high_res_used else '常规'}模式）",
            results=results,
            roi_used=roi_used,
            high_res_used=high_res_used,
            tiles_count=tiles_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检测接口错误: {str(e)}")
        return DetectionResponse(
            success=False,
            message=f"检测失败: {str(e)}"
        )

# 文件上传接口
@app.post("/detect/file", response_model=DetectionResponse, summary="通过文件上传进行旋转框检测")
async def detect_file(
    file: UploadFile = File(...),
    roi: Optional[str] = None,  # 格式: "x1,y1,x2,y2"
    high_res: bool = False,
    tile_size: int = 1024,
    strip_ratio: float = 0.5
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        roi_used = False
        roi_offset = (0, 0)
        high_res_used = high_res
        tiles_count = 0
        
        if roi:
            try:
                roi_coords = list(map(int, roi.split(',')))
                image, roi_offset = crop_image(image, roi_coords)
                roi_used = True
            except Exception as e:
                logger.warning(f"解析ROI参数失败: {str(e)}，将使用完整图像")
        
        # 执行检测
        if high_res_used:
            # 更新高分辨率处理器参数
            high_res_processor.tile_size = max(256, min(tile_size, 4096))
            high_res_processor.strip_ratio = max(0.1, min(strip_ratio, 0.9))
            high_res_processor.strip = int(high_res_processor.tile_size * high_res_processor.strip_ratio)
            high_res_processor.step = high_res_processor.tile_size - high_res_processor.strip
            
            # 切片检测
            tiles = high_res_processor.slice_image(image)
            tiles_count = len(tiles)
            raw_results = high_res_processor.detect_tiles(tiles, roi_offset)
            results = [DetectionResult(**item) for item in raw_results]
        else:
            # 常规检测
            results = process_obb_detection(image, roi_offset)
        
        return DetectionResponse(
            success=True,
            message=f"成功检测到{len(results)}个目标（{'高分辨率切片' if high_res_used else '常规'}模式）",
            results=results,
            roi_used=roi_used,
            high_res_used=high_res_used,
            tiles_count=tiles_count
        )
    except Exception as e:
        logger.error(f"文件检测接口错误: {str(e)}")
        return DetectionResponse(
            success=False,
            message=f"文件检测失败: {str(e)}"
        )

# 健康检查接口
@app.get("/health", summary="健康检查接口")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None, 
        "high_res_processor_loaded": high_res_processor is not None,
        "timestamp": str(os.popen('date').read())
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("启动YOLO OBB旋转框检测API服务（支持高分辨率图像）")
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)