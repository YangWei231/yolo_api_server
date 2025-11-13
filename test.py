import requests
import base64
from PIL import Image
import io

# 1. 读取本地图像并转为Base64
image_path = "/home/wsl/Workspace/remote_qwen_lamp/HF_process_w_yolo12/0449.png"  # 替换为你的图像路径
with open(image_path, "rb") as f:
    image_data = f.read()
base64_str = base64.b64encode(image_data).decode("utf-8")  # 转为Base64字符串

# 2. 构造请求数据
url = "http://localhost:8001/detect"
payload = {
    "image_data": base64_str,  # 传入Base64数据
    # "roi": [100, 100, 500, 500]  # 可选：添加ROI区域
    "high_res" : True,
}

# 3. 发送请求
response = requests.post(url, json=payload)

# 4. 解析响应
result = response.json()
if result["success"]:
    print(f"检测到{len(result['results'])}个目标：")
    print("统计结果如下：", result["class_counts"])
    for obj in result["results"]:
        print(f"{obj['class_name']} (置信度：{obj['confidence']:.2f})，坐标：{obj['rbox']}")
else:
    print("检测失败：", result["message"])