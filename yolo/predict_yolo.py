from ultralytics import YOLO
import cv2
import os
import math
from pathlib import Path
import numpy as np

model = YOLO("results/ver5-detectron2-3n5squares/runs/weights/best.pt")

# 设置图片文件夹和输出文件夹
image_folder = "datasets/hard_test"
output_folder = "C:/Users/hsung/Pictures"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取所有图片文件
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
image_files = [f for f in os.listdir(image_folder) 
               if Path(f).suffix.lower() in image_extensions]

# 批量处理
batch_size = 4
batch = image_files[:batch_size]

# 推理并收集结果
results_list = []
for image_file in batch:
    image_path = os.path.join(image_folder, image_file)
    
    # 加载原图
    original_image = cv2.imread(image_path)
    if original_image is None:
        continue
    
    # YOLO 检测
    results = model(image_path, conf=0.6)
    result = results[0]
    annotated_frame = result.plot()
    results_list.append(annotated_frame)
    print(f"Processed: {image_file}")

# 马赛克拼接
num_images = len(results_list)
grid_cols = math.ceil(math.sqrt(num_images))
grid_rows = math.ceil(num_images / grid_cols)

# 获取单张图片尺寸
img_h, img_w = results_list[0].shape[:2]

# 创建拼接后的画布
mosaic_h = grid_rows * img_h
mosaic_w = grid_cols * img_w
mosaic = 255 * np.ones((mosaic_h, mosaic_w, 3), dtype=np.uint8)

# 填充图片到马赛克

for idx, img in enumerate(results_list):
    row = idx // grid_cols
    col = idx % grid_cols
    y_start = row * img_h
    x_start = col * img_w
    mosaic[y_start:y_start+img_h, x_start:x_start+img_w] = img

# 保存结果
output_path = os.path.join(output_folder, "yolo_mosaic_5.png")
cv2.imwrite(output_path, mosaic)
print(f"马赛克拼接结果已保存到: {output_path}")
print("处理完成！")

