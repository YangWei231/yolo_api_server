# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from services import YoloDetector
from schemas import DetectionRequest, DetectionResponse, DetectionResult
from utils import (
    load_from_base64, load_from_url, load_from_file,
    crop_roi, get_logger,count_detection_classes
)
import os
from typing import Optional, List

# 初始化日志
logger = get_logger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="YOLO OBB旋转框检测API",
    description="支持超高分辨率图像切片检测，支持ROI区域裁剪"
)

# 初始化检测服务（全局单例，避免重复加载模型）
detector = YoloDetector()

# 检测接口（支持Base64/URL输入）
@app.post("/detect", response_model=DetectionResponse, summary="旋转框目标检测接口")
async def detect(request: DetectionRequest):
    try:
        # 加载图像
        if request.image_data:
            image = load_from_base64(request.image_data)
        elif request.image_url:
            image = load_from_url(request.image_url)
        else:
            raise HTTPException(status_code=400, detail="必须提供image_data或image_url")
        
        # 处理ROI
        roi_used = False
        roi_offset = (0, 0)
        if request.roi:
            image, roi_offset = crop_roi(image, request.roi)
            roi_used = True
        
        # 执行检测（常规/高分辨率模式）
        if request.high_res:
            results_raw, tiles_count = detector.detect_high_res(
                image, roi_offset, request.tile_size, request.strip_ratio
            )
            high_res_used = True
        else:
            results_raw = detector.detect_normal(image, roi_offset)
            tiles_count = 0
            high_res_used = False
        
        # 转换为响应模型
        results = [DetectionResult(**item) for item in results_raw]
        class_counts = count_detection_classes(results)
        return DetectionResponse(
            success=True,
            message=f"检测完成，共{len(results)}个目标（{'高分辨率' if high_res_used else '常规'}模式）",
            results=results,
            roi_used=roi_used,
            high_res_used=high_res_used,
            tiles_count=tiles_count,
            class_counts=class_counts  # 新增：返回统计结果
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检测接口错误: {str(e)}")
        return DetectionResponse(success=False, message=f"检测失败: {str(e)}")

# 文件上传检测接口
@app.post("/detect/file", response_model=DetectionResponse, summary="文件上传检测接口")
async def detect_file(
    file: UploadFile = File(...),
    roi: Optional[str] = None,
    high_res: bool = False,
    tile_size: int = 1024,
    strip_ratio: float = 0.5
):
    try:
        # 加载文件图像
        file_content = await file.read()
        image = load_from_file(file_content)
        
        # 处理ROI
        roi_used = False
        roi_offset = (0, 0)
        if roi:
            try:
                roi_coords = list(map(int, roi.split(',')))
                image, roi_offset = crop_roi(image, roi_coords)
                roi_used = True
            except Exception as e:
                logger.warning(f"ROI解析失败，使用原图: {str(e)}")
        
        # 执行检测
        if high_res:
            results_raw, tiles_count = detector.detect_high_res(
                image, roi_offset, tile_size, strip_ratio
            )
            high_res_used = True
        else:
            results_raw = detector.detect_normal(image, roi_offset)
            tiles_count = 0
            high_res_used = False
        
        results = [DetectionResult(**item) for item in results_raw]
        class_counts = count_detection_classes(results)
        return DetectionResponse(
            success=True,
            message=f"文件检测完成，共{len(results)}个目标（{'高分辨率' if high_res_used else '常规'}模式）",
            results=results,
            roi_used=roi_used,
            high_res_used=high_res_used,
            tiles_count=tiles_count,
            class_counts=class_counts  # 新增：返回统计结果
        )
    except Exception as e:
        logger.error(f"文件检测接口错误: {str(e)}")
        return DetectionResponse(success=False, message=f"文件检测失败: {str(e)}")

# 健康检查接口
@app.get("/health", summary="服务健康检查")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "timestamp": os.popen('date').read().strip()
    }

# 启动服务
if __name__ == "__main__":
    import uvicorn
    logger.info("启动YOLO检测API服务（支持高分辨率图像）")
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)