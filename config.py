# config.py
import os
from typing import Tuple

# 模型配置
YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH", 
    "/home/wsl/Workspace/remote_qwen_lamp/HF_process_w_yolo12/yolo11s-obb.pt"
)

# 高分辨率处理默认参数
DEFAULT_TILE_SIZE = 1024
DEFAULT_STRIP_RATIO = 0.5  # 重叠比例
TILE_SIZE_RANGE: Tuple[int, int] = (256, 4096)  # 切片大小允许范围
STRIP_RATIO_RANGE: Tuple[float, float] = (0.1, 0.9)  # 重叠比例允许范围

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")