import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict

class DetectionVisualizer:
    def __init__(self):
        """初始化检测结果可视化工具"""
        self.color_map = {}  # 类别颜色映射 {class_name: (R, G, B)}
        self.font = self._load_font()  # 加载字体

    def _load_font(self, font_size=20):
        """加载支持中文的字体"""
        try:
            # 尝试加载系统中文字体（可根据系统修改路径）
            return ImageFont.truetype("simhei.ttf", font_size)  # Windows
        except:
            try:
                return ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)  # MacOS
            except:
                try:
                    return ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", font_size)  # Linux
                except:
                    #  fallback to default font
                    return ImageFont.load_default()

    def _get_color(self, class_name: str) -> tuple:
        """为每个类别生成固定颜色"""
        if class_name not in self.color_map:
            self.color_map[class_name] = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200)
            )
        return self.color_map[class_name]

    def rbox_to_polygon(self, rbox: List[float]) -> List[tuple]:
        """
        将旋转框(rbox)转换为多边形坐标
        rbox格式: [x_center, y_center, width, height, angle(弧度)]
        """
        x_center, y_center, width, height, angle = rbox
        
        # 计算旋转矩形的四个顶点
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # 半宽和半高
        half_w = width / 2
        half_h = height / 2
        
        # 四个顶点坐标（相对于中心）
        points = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]
        
        # 旋转并平移
        rotated_points = []
        for x, y in points:
            new_x = x * cos_theta - y * sin_theta + x_center
            new_y = x * sin_theta + y * cos_theta + y_center
            rotated_points.append((new_x, new_y))
            
        return rotated_points

    def save_results_to_json(self, results: List[Dict], output_path: str) -> None:
        """保存检测结果到JSON文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"检测结果已保存至: {os.path.abspath(output_path)}")

    def visualize_results(self, image_path: str, results: List[Dict], output_path: str) -> None:
        """
        在图像上可视化检测结果
        :param image_path: 原始图像路径
        :param results: 检测结果列表
        :param output_path: 可视化结果保存路径
        """
        print(f"开始可视化 {len(results)} 个检测结果...")
        
        # 加载原始图像
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            
            for result in results:
                rbox = result["rbox"]
                class_name = result["class_name"]
                confidence = result["confidence"]
                
                # 转换旋转框为多边形
                polygon = self.rbox_to_polygon(rbox)
                
                # 获取颜色
                color = self._get_color(class_name)
                
                # 绘制旋转框
                draw.polygon(polygon, outline=color, width=3)
                
                # 绘制标签
                label = f"{class_name} {confidence:.2f}"
                label_x, label_y = polygon[0]  # 以第一个点为标签起点
                
                # 绘制标签背景
                label_bbox = draw.textbbox((label_x, label_y), label, font=self.font)
                draw.rectangle(label_bbox, fill=(255, 255, 255, 180))  # 半透明白色背景
                
                # 绘制标签文字
                draw.text((label_x, label_y), label, font=self.font, fill=color)
            
            # 保存可视化结果
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            print(f"可视化结果已保存至: {os.path.abspath(output_path)}")