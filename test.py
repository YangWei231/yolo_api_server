import os
from services.detector import YoloDetector
from utils.image_loader import load_from_file
from utils.detection_visualizer import DetectionVisualizer

def main():
    # 初始化检测器和可视化工具
    detector = YoloDetector()
    visualizer = DetectionVisualizer()
    
    # 测试图像路径
    test_image_path = "/home/wsl/Workspace/remote_qwen_lamp/HF_process_w_yolo12/1176.png"  # 替换为你的测试图像路径
    file_name = os.path.basename(test_image_path)
    if not os.path.exists(test_image_path):
        print(f"错误: 图像文件 {test_image_path} 不存在")
        return
    
    # 创建输出目录
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载图像
        with open(test_image_path, "rb") as f:
            image = load_from_file(f.read())
        
        # 执行高分辨率检测
        results, tiles_count = detector.detect_high_res(
            image,
            tile_size=1024,
            strip_ratio=0.5
        )
        
        print(f"检测完成，共发现 {len(results)} 个目标，切片数量: {tiles_count}")
        
        # 保存结果到JSON
        json_output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.json")
        visualizer.save_results_to_json(results, json_output_path)
        
        # 可视化结果并保存
        viz_output_path = os.path.join(output_dir, f"{file_name}")
        visualizer.visualize_results(test_image_path, results, viz_output_path)
        
    except Exception as e:
        print(f"检测过程出错: {str(e)}")

if __name__ == "__main__":
    main()