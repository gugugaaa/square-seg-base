from ultralytics import YOLO
import cv2
import os
from pathlib import Path

model = YOLO("yolo12n.pt")

# 设置图片文件夹和输出文件夹
image_folder = "C:/Users/hsung/Pictures/test"
output_folder = "C:/Users/hsung/Pictures/test/output"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取所有图片文件
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
image_files = [f for f in os.listdir(image_folder) 
               if Path(f).suffix.lower() in image_extensions]

# 处理每张图片
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    # 加载原图
    original_image = cv2.imread(image_path)
    if original_image is None:
        continue
    
    # YOLO 检测
    results = model(image_path, conf=0.6)
    result = results[0]
    annotated_frame = result.plot()
    
    # 并列合并：原图 + 检测图
    combined = cv2.hconcat([original_image, annotated_frame])
    
    # 保存结果
    output_path = os.path.join(output_folder, f"combined_{Path(image_file).stem}.png")
    cv2.imwrite(output_path, combined)
    print(f"Saved: {output_path}")

print("处理完成！")

