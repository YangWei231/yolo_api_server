# YOLO OBB 旋转框检测API服务（支持高分辨率图像）

基于FastAPI和YOLO OBB模型的目标检测服务，支持超高分辨率图像切片检测、ROI区域裁剪，并提供检测结果的分类统计功能，适用于大模型调用或独立的目标检测场景。


## 核心功能

- **旋转框检测**：基于YOLO OBB模型，支持带角度的旋转矩形框检测（输出格式：[x_center, y_center, width, height, angle]）。
- **高分辨率处理**：自动将超大图像切片为子图（可配置切片大小和重叠比例），检测后合并结果。
- **ROI区域裁剪**：支持指定感兴趣区域（ROI），仅检测区域内目标，提高效率。
- **多输入方式**：支持Base64编码图像、图像URL、文件上传三种输入方式。
- **分类统计**：自动统计检测结果中每个类别的数量，便于快速分析。


## 项目结构

```
yolo_server_hf/
├── main.py               # 应用入口（FastAPI实例、路由注册）
├── config.py             # 配置中心（模型路径、默认参数等）
├── schemas/              # 数据模型（请求/响应结构）
│   ├── request.py        # 检测请求模型
│   └── response.py       # 检测响应模型
├── services/             # 核心业务逻辑
│   ├── detector.py       # 检测服务（常规检测+高分辨率检测）
│   └── high_res_processor.py  # 高分辨率图像切片逻辑
└── utils/                # 工具函数
    ├── image_loader.py   # 图像加载（Base64/URL/文件）
    ├── image_utils.py    # 图像预处理（ROI裁剪等）
    ├── logger.py         # 日志配置
    └── analysis_utils.py # 检测结果统计（分类计数）
```


## 环境依赖

- Python 3.8+
- 核心依赖库：
  - `fastapi`：API服务框架
  - `uvicorn`：ASGI服务器
  - `ultralytics`：YOLO模型推理
  - `opencv-python`：图像处理
  - `pillow`：图像加载与裁剪
  - `requests`：URL图像加载

安装依赖：
```bash
pip install fastapi uvicorn ultralytics opencv-python pillow requests pydantic
```


## 配置说明

修改 `config.py` 配置核心参数：

| 参数名              | 说明                                  | 默认值                                      |
|---------------------|---------------------------------------|---------------------------------------------|
| `YOLO_MODEL_PATH`   | YOLO OBB模型路径（.pt文件）           | `/home/wsl/.../yolo11s-obb.pt`（需自行修改） |
| `DEFAULT_TILE_SIZE` | 高分辨率模式下默认切片大小            | 1024（像素）                                |
| `DEFAULT_STRIP_RATIO` | 切片重叠比例（0~1）                  | 0.5（即重叠50%）                             |
| `TILE_SIZE_RANGE`   | 切片大小允许范围（最小值，最大值）    | (256, 4096)                                 |
| `STRIP_RATIO_RANGE` | 重叠比例允许范围（最小值，最大值）    | (0.1, 0.9)                                  |
| `LOG_LEVEL`         | 日志级别                              | "INFO"                                      |


## 启动服务

```bash
# 直接启动（默认端口8001，主机0.0.0.0）
python main.py

# 自定义端口和主机
PORT=8002 HOST=127.0.0.1 python main.py
```

服务启动后，可通过以下地址访问API文档：
- Swagger UI：`http://localhost:8001/docs`（交互式API测试界面）
- 健康检查：`http://localhost:8001/health`（验证服务是否正常运行）


## API接口使用指南

### 1. 通用检测接口（支持Base64/URL输入）

**接口地址**：`POST /detect`

**请求体参数**：

| 参数名        | 类型    | 说明                                  | 是否必填 |
|---------------|---------|---------------------------------------|----------|
| `image_data`  | string  | Base64编码的图像数据（如`data:image/png;base64,...`） | 二选一   |
| `image_url`   | string  | 图像URL（需服务可访问）               | 二选一   |
| `roi`         | list    | 感兴趣区域，格式：[x1, y1, x2, y2]    | 可选     |
| `high_res`    | bool    | 是否启用高分辨率切片模式              | 可选（默认false） |
| `tile_size`   | int     | 切片大小（仅高分辨率模式有效）        | 可选（默认1024） |
| `strip_ratio` | float   | 切片重叠比例（仅高分辨率模式有效）    | 可选（默认0.5） |

**响应示例**：
```json
{
  "success": true,
  "message": "检测完成，共5个目标（常规模式）",
  "results": [
    {
      "class_name": "car",
      "class_id": 2,
      "confidence": 0.92,
      "rbox": [100.5, 200.3, 50.2, 30.1, 0.1]
    }
  ],
  "roi_used": false,
  "high_res_used": false,
  "tiles_count": 0,
  "class_counts": {
    "car": 3,
    "person": 2
  }
}
```


### 2. 文件上传检测接口

**接口地址**：`POST /detect/file`

**请求参数**：
- `file`：上传的图像文件（multipart/form-data格式）
- `roi`：可选，感兴趣区域（格式："x1,y1,x2,y2"，如"100,200,500,600"）
- `high_res`：是否启用高分辨率模式（默认false）
- `tile_size`：切片大小（默认1024）
- `strip_ratio`：重叠比例（默认0.5）

**curl示例**：
```bash
curl -X POST "http://localhost:8001/detect/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg" \
  -F "high_res=true" \
  -F "tile_size=2048"
```


### 3. 健康检查接口

**接口地址**：`GET /health`

**响应示例**：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-12 10:00:00"
}
```


## 字段说明

- **检测结果（results）**：
  - `class_name`：目标类别名称（如"car"）
  - `class_id`：目标类别ID（对应YOLO模型的类别定义）
  - `confidence`：检测置信度（0~1）
  - `rbox`：旋转框参数（[x_center, y_center, width, height, angle]），其中`angle`单位为弧度。

- **统计结果（class_counts）**：键为类别名称，值为该类别的检测数量。


## 常见问题

1. **模型加载失败**：检查`config.py`中`YOLO_MODEL_PATH`是否指向正确的`.pt`模型文件，确保模型存在且格式正确。
2. **图像加载失败**：Base64编码需去除前缀（或服务自动处理），URL需确保服务可访问，文件需为合法图像格式（如jpg、png）。
3. **高分辨率模式速度慢**：可增大`tile_size`或减小`strip_ratio`减少切片数量（但可能影响检测完整性）。
4. **ROI参数无效**：确保`roi`格式为[x1, y1, x2, y2]，且坐标在图像范围内（x1 < x2，y1 < y2）。


## 许可证

MIT License