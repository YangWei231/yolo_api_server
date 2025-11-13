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
from typing import Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(title="YOLO OBB旋转框检测API", description="用于大模型调用的YOLO旋转框检测服务，支持ROI区域裁剪")

# 加载YOLO OBB模型（切换为旋转框模型）
try:
    # 模型路径：使用yolos-obb.ot（建议确认格式，通常为.pt）
    model_path = "/home/wsl/Workspace/remote_qwen_lamp/HF_process_w_yolo12/yolo11s-obb.pt"
    model = YOLO(model_path)
    logger.info(f"成功加载YOLO OBB模型: {model_path}")
except Exception as e:
    logger.error(f"加载YOLO OBB模型失败: {str(e)}")
    raise

# 定义请求模型（与之前一致）
class DetectionRequest(BaseModel):
    image_data: Optional[str] = None  # Base64编码的图像数据
    image_url: Optional[str] = None   # 图像URL
    roi: Optional[list[int]] = None   # 感兴趣区域 [x1, y1, x2, y2]，可选

# 定义检测结果模型（适配旋转框）
class DetectionResult(BaseModel):
    """单个目标的旋转框检测结果"""
    class_name: str
    class_id: int
    confidence: float
    # 旋转框参数：[x_center, y_center, width, height, angle]（角度单位：弧度）
    rbox: list[float]  # 坐标相对于原始图像

class DetectionResponse(BaseModel):
    success: bool
    message: str = ""
    results: list[DetectionResult] = []
    roi_used: bool = False  # 是否使用了ROI区域

# 工具函数：加载图像（与之前一致）
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

# 工具函数：裁剪图像（与之前一致）
def crop_image(image: Image.Image, roi: list[int]) -> tuple[Image.Image, tuple[int, int]]:
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

# 核心修改：处理旋转框检测结果
def process_obb_detection(image: Image.Image, roi_offset: tuple = (0, 0)) -> list[DetectionResult]:
    """处理旋转框检测，返回结果（坐标转换为原始图像）"""
    try:
        # 转换为OpenCV格式
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 执行OBB检测（YOLO OBB模型输出存放在result.obb中）
        results = model.predict(source=cv_image)
        
        detection_results = []
        for result in results:
            # 旋转框数据在result.obb中（替代原result.boxes）
            if not hasattr(result, 'obb'):
                continue  # 无旋转框结果则跳过
            for obb in result.obb:
                # 提取旋转框参数：xywhr（中心坐标、宽、高、旋转角度（弧度））
                # 格式：[x_center, y_center, width, height, angle]
                rbox = obb.xywhr[0].tolist()
                
                # 如果使用了ROI，将中心坐标转换回原始图像（宽高和角度不受偏移影响）
                if roi_offset:
                    rbox[0] += roi_offset[0]  # x_center += ROI的x1偏移
                    rbox[1] += roi_offset[1]  # y_center += ROI的y1偏移
                
                detection_results.append(DetectionResult(
                    class_name=result.names[int(obb.cls)],
                    class_id=int(obb.cls),
                    confidence=float(obb.conf),
                    rbox=rbox  # 存储旋转框参数
                ))
        
        return detection_results
    except Exception as e:
        logger.error(f"旋转框检测处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检测处理失败: {str(e)}")

# 检测接口（使用旋转框处理函数）
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
        
        # 裁剪ROI（如果提供）
        if request.roi:
            image, roi_offset = crop_image(image, request.roi)
            roi_used = True
        
        # 执行旋转框检测
        results = process_obb_detection(image, roi_offset)
        
        return DetectionResponse(
            success=True,
            message=f"成功检测到{len(results)}个目标（旋转框）",
            results=results,
            roi_used=roi_used
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检测接口错误: {str(e)}")
        return DetectionResponse(
            success=False,
            message=f"检测失败: {str(e)}"
        )

# 文件上传接口（适配旋转框）
@app.post("/detect/file", response_model=DetectionResponse, summary="通过文件上传进行旋转框检测")
async def detect_file(
    file: UploadFile = File(...),
    roi: Optional[str] = None  # 格式: "x1,y1,x2,y2"
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        roi_used = False
        roi_offset = (0, 0)
        
        if roi:
            try:
                roi_coords = list(map(int, roi.split(',')))
                image, roi_offset = crop_image(image, roi_coords)
                roi_used = True
            except Exception as e:
                logger.warning(f"解析ROI参数失败: {str(e)}，将使用完整图像")
        
        results = process_obb_detection(image, roi_offset)
        
        return DetectionResponse(
            success=True,
            message=f"成功检测到{len(results)}个目标（旋转框）",
            results=results,
            roi_used=roi_used
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
    return {"status": "healthy", "model_loaded": model is not None, "timestamp": str(os.popen('date').read())}

if __name__ == "__main__":
    import uvicorn
    logger.info("启动YOLO OBB旋转框检测API服务")
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)